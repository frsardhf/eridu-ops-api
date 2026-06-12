"""Coarse-to-fine alpha-masked template matcher (zero-mean NCC).

Replaces CLIP icon classification with exact-asset matching: every SchaleDB
sprite ships as a uniform 146x116 BGRA canvas, and the game renders that same
canvas inside each inventory cell (canvas width ≈ 1.01-1.03 x cell width,
calibrated per inventory type). Matching the actual pixels — masked to the
sprite's opaque region — separates tier/colour siblings that collapse in CLIP
embedding space (validated 250/250 on marked ground truth vs 239/250 for
CLIP + correction passes).

Method notes:
- TM_CCOEFF_NORMED (zero-mean) is essential. Plain TM_CCORR_NORMED saturates
  near 1.0 for bright templates over the bright rarity backgrounds and scored
  46.8% on the same ground truth.
- The mask drops the bottom RIBBON_FRAC of the canvas (quantity ribbon is
  drawn over the sprite) and, for equipment, a top-left block (tier badge).
- Per-cell position search (±MARGIN px) is required: the ratio-derived grid
  bounds overestimate the cell pitch by ~8px/column, so sprite positions
  drift relative to the computed cell origins.

Three stages keep it fast enough for the API path (correlation work scales
with image_area x template_area, so each downscale level costs ~1/16 of the
one above):
  1. prefilter — all candidates at quarter resolution → PREFILTER_TOP_K;
  2. coarse — survivors at half resolution → COARSE_TOP_K;
  3. refine — shortlist only, full resolution, all calibrated scales.
On the marked ground truth the true item ranked #1 at half resolution in all
250 cells, so the shortlists carry a wide safety margin.
"""
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
ICON_CACHE_DIR = os.path.join(CACHE_DIR, 'icons')

# Sprite canvas width as a fraction of cell_w (CCOEFF scale-sweep calibration
# on FHD screenshots; the sweep peaks sharply, unlike CCORR's flat ridge).
CANVAS_SCALES = {'items': (1.00, 1.01, 1.02), 'equipment': (1.01, 1.02, 1.03)}
RIBBON_FRAC = 0.22   # bottom canvas fraction hidden by the quantity ribbon
BADGE_W     = 0.26   # equipment tier-badge block (fraction of canvas w/h)
BADGE_H     = 0.24
MARGIN_X    = 20     # search window expansion around the computed cell box
MARGIN_Y    = 16
PREFILTER_DOWNSCALE = 0.25
PREFILTER_TOP_K     = 20
COARSE_DOWNSCALE = 0.5
COARSE_TOP_K     = 5

# {inv_type: [(item_id, bgr_canvas, alpha_canvas), ...]}
_SPRITES: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]] = {}
# {(inv_type, round(cell_w)): bank}
_BANKS: Dict[Tuple[str, int], dict] = {}


def _load_sprites(inv_type: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    if inv_type in _SPRITES:
        return _SPRITES[inv_type]
    index_path = os.path.join(CACHE_DIR, f'icon_index_{inv_type}.json')
    sprites = []
    if os.path.exists(index_path):
        with open(index_path, encoding='utf-8') as fh:
            index = json.load(fh)['items']
        icon_dir = os.path.join(ICON_CACHE_DIR, inv_type)
        for item_id, info in index.items():
            raw = cv2.imread(os.path.join(icon_dir, info.get('filename', '')),
                             cv2.IMREAD_UNCHANGED)
            if raw is None or raw.ndim != 3 or raw.shape[2] != 4:
                continue
            sprites.append((item_id, raw[:, :, :3], raw[:, :, 3]))
    _SPRITES[inv_type] = sprites
    return sprites


def _make_view(bgr: np.ndarray, alpha: np.ndarray, width: int, inv_type: str,
               erode: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Resize the canvas to `width` px and build the hygiene mask.

    Erosion (anti-aliased edge leak) is only applied at full resolution: at
    the reduced prefilter/coarse sizes it can wipe out thin-artwork sprites
    (item 5010's mask died this way and the item became unmatchable).
    """
    ch, cw = bgr.shape[:2]
    h = int(width * ch / cw)
    if h < 8 or width < 8:
        return None
    tmpl = cv2.resize(bgr, (width, h), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alpha, (width, h), interpolation=cv2.INTER_AREA)
    mask = (a > 128).astype(np.uint8) * 255
    if erode:
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    mask[int(h * (1 - RIBBON_FRAC)):, :] = 0
    if inv_type == 'equipment':
        mask[:int(h * BADGE_H), :int(width * BADGE_W)] = 0
    if cv2.countNonZero(mask) < 12:
        return None
    return tmpl, mask


def get_bank(inv_type: str, cell_w: float) -> dict:
    """Template bank for one (inventory type, cell width): coarse half-res
    views at the middle scale + full-res views at every calibrated scale."""
    key = (inv_type, int(round(cell_w)))
    bank = _BANKS.get(key)
    if bank is not None:
        return bank

    scales = CANVAS_SCALES[inv_type]
    mid = scales[len(scales) // 2]
    pre, coarse, full = [], {}, {}
    pre_byes, coarse_byes = [], []
    for item_id, bgr, alpha in _load_sprites(inv_type):
        views = []
        for frac in scales:
            v = _make_view(bgr, alpha, int(cell_w * frac), inv_type)
            if v is not None:
                views.append(v)
        if not views:
            continue  # unmatchable at full res — nothing we can do
        full[item_id] = views

        # An item without a usable view at a reduced stage is never dropped —
        # it skips the stage and auto-advances to the next one (a "bye").
        pre_view = _make_view(bgr, alpha,
                              int(cell_w * mid * PREFILTER_DOWNSCALE),
                              inv_type, erode=False)
        if pre_view is not None:
            pre.append((item_id, pre_view[0], pre_view[1]))
        else:
            pre_byes.append(item_id)
        cv_view = _make_view(bgr, alpha,
                             int(cell_w * mid * COARSE_DOWNSCALE),
                             inv_type, erode=False)
        if cv_view is not None:
            coarse[item_id] = cv_view
        else:
            coarse_byes.append(item_id)

    bank = {'pre': pre, 'coarse': coarse, 'full': full,
            'pre_byes': pre_byes, 'coarse_byes': coarse_byes}
    _BANKS[key] = bank
    return bank


def _best_ccoeff(window: np.ndarray, tmpl: np.ndarray,
                 mask: np.ndarray) -> float:
    th, tw = tmpl.shape[:2]
    if th >= window.shape[0] or tw >= window.shape[1]:
        return -2.0
    res = cv2.matchTemplate(window, tmpl, cv2.TM_CCOEFF_NORMED, mask=mask)
    res = np.nan_to_num(res, nan=-2.0, posinf=-2.0, neginf=-2.0)
    return float(res.max())


def prefilter_scores(window: np.ndarray, bank: dict) -> List[Tuple[float, str]]:
    """Stage 1: score every candidate at quarter resolution, best-first."""
    small = cv2.resize(window, None,
                       fx=PREFILTER_DOWNSCALE, fy=PREFILTER_DOWNSCALE,
                       interpolation=cv2.INTER_AREA)
    scores = [(_best_ccoeff(small, tmpl, mask), item_id)
              for item_id, tmpl, mask in bank['pre']]
    scores.sort(reverse=True)
    return scores


def coarse_scores(window: np.ndarray, bank: dict,
                  item_ids: List[str]) -> List[Tuple[float, str]]:
    """Stage 2: score the prefilter survivors at half resolution, best-first."""
    small = cv2.resize(window, None, fx=COARSE_DOWNSCALE, fy=COARSE_DOWNSCALE,
                       interpolation=cv2.INTER_AREA)
    scores = []
    for item_id in item_ids:
        view = bank['coarse'].get(item_id)
        if view is not None:
            scores.append((_best_ccoeff(small, view[0], view[1]), item_id))
    scores.sort(reverse=True)
    return scores


def refine(window: np.ndarray, bank: dict,
           item_ids: List[str]) -> List[Tuple[float, str]]:
    """Stage 2: full-resolution, all calibrated scales, shortlist only."""
    scores = []
    for item_id in item_ids:
        best = -2.0
        for tmpl, mask in bank['full'].get(item_id, ()):
            best = max(best, _best_ccoeff(window, tmpl, mask))
        scores.append((best, item_id))
    scores.sort(reverse=True)
    return scores


def match_window(window: np.ndarray, bank: dict
                 ) -> Tuple[Optional[str], float, float]:
    """Return (item_id, score, top1-top2 margin) for one cell window."""
    pre = prefilter_scores(window, bank)
    survivors = [iid for _, iid in pre[:PREFILTER_TOP_K]] + bank['pre_byes']
    coarse = coarse_scores(window, bank, survivors)
    shortlist = [iid for _, iid in coarse[:COARSE_TOP_K]]
    shortlist += [iid for iid in survivors
                  if iid in bank['coarse_byes'] and iid not in shortlist]
    refined = refine(window, bank, shortlist)
    if not refined:
        return None, -2.0, 0.0
    best_score, best_id = refined[0]
    margin = best_score - refined[1][0] if len(refined) > 1 else 1.0
    return best_id, best_score, margin


def cell_window(grid: np.ndarray, cell_w: float, row_bounds: List[int],
                row: int, col: int) -> np.ndarray:
    """Expanded search window around one grid cell, clamped to the grid."""
    gh, gw = grid.shape[:2]
    x0, y0 = int(col * cell_w), row_bounds[row]
    x1, y1 = int((col + 1) * cell_w), row_bounds[row + 1]
    return grid[max(0, y0 - MARGIN_Y):min(gh, y1 + MARGIN_Y),
                max(0, x0 - MARGIN_X):min(gw, x1 + MARGIN_X)]


def warm() -> None:
    """Preload sprite canvases (the disk-I/O part of bank building)."""
    for inv_type in ('items', 'equipment'):
        n = len(_load_sprites(inv_type))
        print(f'[inventory_parser] NCC sprites loaded for {inv_type} ({n})')
