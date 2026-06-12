"""Coarse-to-fine alpha-masked template matcher (zero-mean NCC, GEMM-based).

Replaces CLIP icon classification with exact-asset matching: every SchaleDB
sprite ships as a uniform 146x116 BGRA canvas, and the game renders that same
canvas inside each inventory cell (canvas width ≈ 1.01-1.03 x cell width,
calibrated per inventory type). Matching the actual pixels — masked to the
sprite's opaque region — separates tier/colour siblings that collapse in CLIP
embedding space (validated 250/250 on marked ground truth vs 239/250 for
CLIP + correction passes).

Method notes:
- Zero-mean NCC is essential. Plain (non-zero-mean) normalised correlation
  saturates near 1.0 for bright templates over the bright rarity backgrounds
  and scored 46.8% on the same ground truth.
- The mask drops the bottom RIBBON_FRAC of the canvas (quantity ribbon is
  drawn over the sprite) and, for equipment, a top-left block (tier badge).
- Per-cell position search (±MARGIN px) is required: the ratio-derived grid
  bounds overestimate the cell pitch by ~8px/column, so sprite positions
  drift relative to the computed cell origins.

Implementation: instead of per-candidate cv2.matchTemplate calls (which cost
~2s/cell on a budget VPS vCPU), each stage scores ALL candidates over a grid
of positions with three BLAS matrix multiplies. For mask M, template T with
masked mean t̄, and patch P: since Σ M·(T−t̄) = 0, the zero-mean masked
correlation numerator Σ M(P−p̄)(T−t̄) equals P · (M·(T−t̄)) — the patch mean
drops out — and the patch variance term Σ M(P−p̄)² comes from P²·M and P·M.
Stacking candidates row-wise turns each stage into patches @ stack.T.

Stages (each level keeps the per-candidate hygiene masks):
  1. quarter resolution, full position extent (stride 2) → PREFILTER_TOP_K;
  2. half resolution, local search around the stage-1 peak → COARSE_TOP_K;
  3. full resolution, all calibrated scales, local search + 1px polish.
"""
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
ICON_CACHE_DIR = os.path.join(CACHE_DIR, 'icons')
# Hand-verified template overrides (BGRA, same filename as the cached icon).
# For items whose SchaleDB sprite drifted from the in-game render (e.g. JP
# cover art on EN clients: gifts 5009/5023), debug_make_override.py crops the
# real render and re-uses the original alpha as the mask. Committed in git —
# survives cache wipes and re-downloads.
OVERRIDE_DIR = os.path.join(BASE_DIR, 'assets', 'icon_overrides')

# Sprite canvas width as a fraction of cell_w (CCOEFF scale-sweep calibration
# on FHD screenshots; the sweep peaks sharply, unlike CCORR's flat ridge).
CANVAS_SCALES = {'items': (1.00, 1.01, 1.02), 'equipment': (1.01, 1.02, 1.03)}
RIBBON_FRAC = 0.22   # bottom canvas fraction hidden by the quantity ribbon
BADGE_W     = 0.26   # equipment tier-badge block (fraction of canvas w/h)
BADGE_H     = 0.24
MARGIN_X    = 20     # search window expansion around the computed cell box
MARGIN_Y    = 16
PREFILTER_TOP_K = 20
COARSE_TOP_K    = 5

_Q = 0.25    # prefilter level downscale
_H = 0.5     # coarse level downscale
# No blur at the reduced levels: NCC peaks there are ~1px wide, so the
# prefilter scores DENSELY (stride 1 — still one GEMM) instead of widening
# peaks with smoothing. Blurring was tried and hurt: the window blends
# sprite+background at the sprite border while the (alpha-premultiplied)
# template blends sprite-only, and that asymmetry decorrelates exactly the
# borderline cells.
_Q_BLUR = 0.0
_H_BLUR = 0.0
_EPS = 1e-6

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
        override_dir = os.path.join(OVERRIDE_DIR, inv_type)
        for item_id, info in index.items():
            filename = info.get('filename', '')
            path = os.path.join(icon_dir, filename)
            override = os.path.join(override_dir,
                                    os.path.splitext(filename)[0] + '.png')
            if os.path.exists(override):
                path = override
            raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if raw is None or raw.ndim != 3 or raw.shape[2] != 4:
                continue
            sprites.append((item_id, raw[:, :, :3], raw[:, :, 3]))
    _SPRITES[inv_type] = sprites
    return sprites


def _make_view(bgr: np.ndarray, alpha: np.ndarray, width: int, inv_type: str,
               erode: bool = True,
               blur: float = 0.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Resize the canvas to `width` px and build the hygiene mask.

    Erosion (anti-aliased edge leak) is only applied at full resolution: at
    the reduced prefilter/coarse sizes it can wipe out thin-artwork sprites
    (item 5010's mask died this way and the item became unmatchable).
    `blur` widens the correlation peak at reduced levels so a strided
    position grid cannot fall into the valley between score samples.
    """
    ch, cw = bgr.shape[:2]
    h = int(width * ch / cw)
    if h < 8 or width < 8:
        return None
    tmpl = cv2.resize(bgr, (width, h), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alpha, (width, h), interpolation=cv2.INTER_AREA)
    if blur > 0:
        # Alpha-premultiplied blur: canvases carry garbage colours under
        # alpha=0, and a plain blur smears that into the masked border
        # pixels. Premultiplying keeps only sprite colours blending.
        af = (a.astype(np.float32) / 255.0)[:, :, None]
        prem = cv2.GaussianBlur(tmpl.astype(np.float32) * af, (0, 0), blur)
        wsum = cv2.GaussianBlur(af, (0, 0), blur)
        if wsum.ndim == 2:
            wsum = wsum[:, :, None]
        tmpl = (prem / np.maximum(wsum, 1e-3)).clip(0, 255).astype(np.uint8)
    mask = (a > 128).astype(np.uint8)
    if erode:
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    mask[int(h * (1 - RIBBON_FRAC)):, :] = 0
    if inv_type == 'equipment':
        mask[:int(h * BADGE_H), :int(width * BADGE_W)] = 0
    if int(mask.sum()) < 12:
        return None
    return tmpl, mask


def _stack_level(views: List[Tuple[str, np.ndarray, np.ndarray]]) -> Optional[dict]:
    """Stack (item_id, tmpl, mask) views of ONE common size into GEMM arrays.

    Stores tvec = M·(T − t̄) per candidate (t̄ = masked mean over all masked
    pixel-channels), its norm, and the flattened mask — everything the
    masked zero-mean correlation needs at score time.
    """
    if not views:
        return None
    h, w = views[0][1].shape[:2]
    ids, tvecs, masks, tnorms, msums = [], [], [], [], []
    for item_id, tmpl, mask in views:
        t = tmpl.astype(np.float32)
        m3 = np.repeat(mask[:, :, None], 3, axis=2).astype(np.float32)
        msum = float(m3.sum())
        tmean = float((t * m3).sum()) / max(msum, _EPS)
        tvec = (m3 * (t - tmean)).ravel()
        tnorm = float(np.sqrt((tvec * (t.ravel() - tmean)).sum()))
        if tnorm < _EPS:
            continue
        ids.append(item_id)
        tvecs.append(tvec)
        masks.append(m3.ravel())
        tnorms.append(tnorm)
        msums.append(msum)
    if not ids:
        return None
    return {
        'ids': ids,
        'idx': {iid: i for i, iid in enumerate(ids)},
        'T': np.stack(tvecs),            # (K, D)
        'M': np.stack(masks),            # (K, D)
        'tnorm': np.array(tnorms, dtype=np.float32),
        'msum': np.array(msums, dtype=np.float32),
        'w': w, 'h': h,
    }


def get_bank(inv_type: str, cell_w: float) -> dict:
    """Template bank for one (inventory type, cell width): stacked GEMM
    arrays at quarter/half resolution (middle scale) and full resolution
    (every calibrated scale)."""
    key = (inv_type, int(round(cell_w)))
    bank = _BANKS.get(key)
    if bank is not None:
        return bank

    scales = CANVAS_SCALES[inv_type]
    mid = scales[len(scales) // 2]
    sprites = _load_sprites(inv_type)

    def collect(width: int, erode: bool, blur: float = 0.0):
        views, missing = [], []
        for item_id, bgr, alpha in sprites:
            v = _make_view(bgr, alpha, width, inv_type, erode=erode, blur=blur)
            if v is not None:
                views.append((item_id, v[0], v[1]))
            else:
                missing.append(item_id)
        return views, missing

    q_views, q_byes = collect(int(cell_w * mid * _Q), erode=False, blur=_Q_BLUR)
    h_views, h_byes = collect(int(cell_w * mid * _H), erode=False, blur=_H_BLUR)
    full = []
    for frac in scales:
        f_views, _ = collect(int(cell_w * frac), erode=True)
        lvl = _stack_level(f_views)
        if lvl is not None:
            full.append(lvl)

    bank = {
        'q': _stack_level(q_views), 'q_byes': q_byes,
        'h': _stack_level(h_views), 'h_byes': h_byes,
        'f': full,
    }
    _BANKS[key] = bank
    return bank


def _grid_positions(win_w: int, win_h: int, tw: int, th: int,
                    stride: int, center: Optional[Tuple[int, int]] = None,
                    radius: int = 0) -> List[Tuple[int, int]]:
    """Valid template top-left positions, either the full extent (stride
    sweep) or a local neighbourhood around `center`."""
    max_x, max_y = win_w - tw, win_h - th
    if max_x < 0 or max_y < 0:
        return []
    if center is None:
        xs = range(0, max_x + 1, stride)
        ys = range(0, max_y + 1, stride)
    else:
        cx, cy = center
        xs = range(max(0, cx - radius), min(max_x, cx + radius) + 1, stride)
        ys = range(max(0, cy - radius), min(max_y, cy + radius) + 1, stride)
    return [(x, y) for y in ys for x in xs]


def _score_level(window: np.ndarray, level: dict, cand_rows: np.ndarray,
                 positions: List[Tuple[int, int]]):
    """Masked zero-mean NCC of every candidate row at every position.

    Returns (scores (npos, K), positions) — three GEMMs over the stacked
    patch matrix; see module docstring for the algebra.
    """
    tw, th = level['w'], level['h']
    P = np.empty((len(positions), th * tw * 3), dtype=np.float32)
    for i, (x, y) in enumerate(positions):
        P[i] = window[y:y + th, x:x + tw].reshape(-1)

    T = level['T'][cand_rows]
    M = level['M'][cand_rows]
    msum = level['msum'][cand_rows]
    tnorm = level['tnorm'][cand_rows]

    num = P @ T.T                            # Σ M(P-p̄)(T-t̄) per pos/cand
    sP = P @ M.T                             # Σ M·P
    sP2 = (P * P) @ M.T                      # Σ M·P²
    varP = np.maximum(sP2 - (sP * sP) / msum[None, :], _EPS)
    return num / (np.sqrt(varP) * tnorm[None, :]), positions


def _best_per_candidate(scores: np.ndarray, positions) -> Tuple[np.ndarray, list]:
    best_pos_i = scores.argmax(axis=0)
    best = scores[best_pos_i, np.arange(scores.shape[1])]
    return best, [positions[i] for i in best_pos_i]


def _resize(window: np.ndarray, f: float, blur: float = 0.0) -> np.ndarray:
    small = cv2.resize(window, None, fx=f, fy=f,
                       interpolation=cv2.INTER_AREA)
    if blur > 0:
        small = cv2.GaussianBlur(small, (0, 0), blur)
    return small.astype(np.float32)


def prefilter_scores(window: np.ndarray, bank: dict) -> List[Tuple[float, str]]:
    """Stage 1 ranking of every candidate (quarter res), best-first.

    Exposed separately for the eval harness; match_window reuses it.
    """
    ranked, _ = _prefilter(window, bank)
    return ranked


def _prefilter(window: np.ndarray, bank: dict):
    lvl = bank['q']
    if lvl is None:
        return [], {}
    small = _resize(window, _Q, blur=_Q_BLUR)
    positions = _grid_positions(small.shape[1], small.shape[0],
                                lvl['w'], lvl['h'], stride=1)
    if not positions:
        return [], {}
    rows = np.arange(len(lvl['ids']))
    scores, positions = _score_level(small, lvl, rows, positions)
    best, best_pos = _best_per_candidate(scores, positions)
    ranked = sorted(zip(best.tolist(), lvl['ids']), reverse=True)
    pos_by_id = {iid: best_pos[i] for i, iid in enumerate(lvl['ids'])}
    return ranked, pos_by_id


def _stage_union(window: np.ndarray, lvl: dict, item_ids: List[str],
                 centers_by_id: Dict[str, Tuple[int, int]],
                 radius: int, stride: int):
    """Score `item_ids` over the UNION of their per-candidate local grids in
    one GEMM batch.

    Every candidate gets a dense neighbourhood around its OWN previous-stage
    peak — a wrong top-1 can never starve the true candidate of its best
    position (the failure mode of a single shared center). Candidates without
    a center (stage byes) trigger a full-extent stride-2 sweep added to the
    union. Returns (ranked [(score, id)], {id: best_position}).
    """
    rows, kept = [], []
    for iid in item_ids:
        ri = lvl['idx'].get(iid)
        if ri is not None:
            rows.append(ri)
            kept.append(iid)
    if not rows:
        return [], {}

    seen, positions = set(), []
    need_fallback = False
    for iid in kept:
        center = centers_by_id.get(iid)
        if center is None:
            need_fallback = True
            continue
        for p in _grid_positions(window.shape[1], window.shape[0],
                                 lvl['w'], lvl['h'], stride=stride,
                                 center=center, radius=radius):
            if p not in seen:
                seen.add(p)
                positions.append(p)
    if need_fallback or not positions:
        for p in _grid_positions(window.shape[1], window.shape[0],
                                 lvl['w'], lvl['h'], stride=2):
            if p not in seen:
                seen.add(p)
                positions.append(p)
    if not positions:
        return [], {}

    scores, positions = _score_level(window, lvl, np.array(rows), positions)
    best, best_pos = _best_per_candidate(scores, positions)
    order = np.argsort(best)[::-1]
    ranked = [(float(best[i]), kept[i]) for i in order]
    pos_by_id = {kept[i]: best_pos[i] for i in range(len(kept))}
    return ranked, pos_by_id


def match_window(window: np.ndarray, bank: dict
                 ) -> Tuple[Optional[str], float, float]:
    """Return (item_id, score, top1-top2 margin) for one cell window."""
    ranked, q_pos = _prefilter(window, bank)
    survivors = [iid for _, iid in ranked[:PREFILTER_TOP_K]] + bank['q_byes']
    if not survivors:
        return None, -2.0, 0.0

    # ── stage 2: half res, union of per-candidate local searches ─────────
    shortlist, h_pos = survivors, {}
    if bank['h'] is not None:
        h_centers = {iid: (q_pos[iid][0] * 2, q_pos[iid][1] * 2)
                     for iid in survivors if iid in q_pos}
        h_ranked, h_pos = _stage_union(_resize(window, _H, blur=_H_BLUR),
                                       bank['h'], survivors, h_centers,
                                       radius=3, stride=1)
        if h_ranked:
            shortlist = [iid for _, iid in h_ranked[:COARSE_TOP_K]]
            shortlist += [iid for iid in bank['h_byes']
                          if iid in survivors and iid not in shortlist]

    # ── stage 3: full res, every scale, local search + 1px polish ────────
    winf = window.astype(np.float32)
    f_centers = {iid: (h_pos[iid][0] * 2, h_pos[iid][1] * 2)
                 for iid in shortlist if iid in h_pos}
    best_by_id: Dict[str, float] = {}
    best_overall = (-2.0, None, None, None)   # score, id, pos, level
    for lvl_f in bank['f']:
        f_ranked, f_pos = _stage_union(winf, lvl_f, shortlist, f_centers,
                                       radius=4, stride=2)
        for score, iid in f_ranked:
            best_by_id[iid] = max(best_by_id.get(iid, -2.0), score)
        if f_ranked and f_ranked[0][0] > best_overall[0]:
            best_overall = (f_ranked[0][0], f_ranked[0][1],
                            f_pos.get(f_ranked[0][1]), lvl_f)
    if not best_by_id:
        return None, -2.0, 0.0

    # 1px polish around the winning stride-2 peak
    score, iid, pos, lvl_f = best_overall
    if pos is not None:
        p_ranked, _ = _stage_union(winf, lvl_f, [iid], {iid: pos},
                                   radius=1, stride=1)
        if p_ranked:
            best_by_id[iid] = max(best_by_id[iid], p_ranked[0][0])

    finals = sorted(((s, iid) for iid, s in best_by_id.items()), reverse=True)
    best_score, best_id = finals[0]
    margin = best_score - finals[1][0] if len(finals) > 1 else 1.0
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
