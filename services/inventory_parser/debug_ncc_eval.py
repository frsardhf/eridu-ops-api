"""Evaluate alpha-masked template matching against the marked ground truth.

For every grid cell, slides each candidate sprite (full 146x116 canvas at a
small set of calibrated scales, alpha-masked) over an expanded cell window
with masked TM_CCOEFF_NORMED (zero-mean — plain TM_CCORR_NORMED saturates on
bright flat templates and scored 46.8% here) and takes the best score; the
argmax candidate is the prediction. Compares three matchers on the marked
ground truth:

    raw CLIP top-1        — parsed from detected_ids_diag.txt
    CLIP + cheats (prod)  — detected_ids_pristine.txt
    masked NCC            — this matcher

Mask hygiene: alpha binarised >128 and eroded 1px (anti-aliased edge leak),
bottom RIBBON_FRAC of the canvas dropped (quantity ribbon), and for equipment
the top-left badge block dropped (tier label overlay).

Usage (from services/inventory_parser/ with .venv active):
    python debug_ncc_eval.py
"""
import json
import os
import re
import time

import cv2
import numpy as np

from pipeline import _extract_grid, CACHE_DIR, ICON_CACHE_DIR

ITEMS_DIR     = '/Users/frsardhf/Downloads/Images/Items'
EQUIPMENT_DIR = '/Users/frsardhf/Downloads/Images/Equipment'
GT_PATH       = '/Users/frsardhf/Downloads/Images/detected_ids.txt'
PRISTINE_PATH = '/Users/frsardhf/Downloads/Images/detected_ids_pristine.txt'
DIAG_PATH     = '/Users/frsardhf/Downloads/Images/detected_ids_diag.txt'

DIRS = {'items': ITEMS_DIR, 'equipment': EQUIPMENT_DIR}

# Sprite canvas width as a fraction of cell_w. Calibrated per type with a
# CCOEFF scale sweep on known cells (peaks: items 1.01, equipment 1.02-1.03).
CANVAS_SCALES = {'items': (1.00, 1.01, 1.02), 'equipment': (1.01, 1.02, 1.03)}
RIBBON_FRAC  = 0.22   # bottom fraction of the canvas hidden by the qty ribbon
BADGE_W      = 0.26   # equipment tier-badge block (fraction of canvas w/h)
BADGE_H      = 0.24
MARGIN_X     = 20     # search window expansion around the cell
MARGIN_Y     = 16


def parse_id_grid(path):
    """Parse detected_ids-format txt → {(name, row, col): id}, plus file order."""
    out, files = {}, []
    name = inv_type = None
    row = 0
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^#\s+(\S+)\s+\[(\w+)\]', line)
            if m:
                name, inv_type = m.group(1), m.group(2)
                files.append((name, inv_type))
                row = 0
                continue
            for col, tok in enumerate(line.split()):
                if tok != '-':
                    out[(name, row, col)] = tok
            row += 1
    return out, files


def parse_diag_top1(path):
    """Parse the diag file → {(name, row, col): raw_clip_top1_id}."""
    out, name = {}, None
    pat = re.compile(r'r(\d+)c(\d+)\s+final=\s*\S+\s+top1=(\S+?)\(')
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            m = re.match(r'^=== (\S+) \[', line.strip())
            if m:
                name = m.group(1)
                continue
            m = pat.search(line)
            if m and name:
                out[(name, int(m.group(1)), int(m.group(2)))] = m.group(3)
    return out


def build_templates(inv_type: str, cell_w: float):
    """Pre-resize every candidate sprite + hygiene mask at each canvas scale.

    Returns {item_id: [(tmpl, mask), ...]} — one entry per scale.
    """
    index_path = os.path.join(CACHE_DIR, f'icon_index_{inv_type}.json')
    with open(index_path, encoding='utf-8') as fh:
        index = json.load(fh)['items']

    templates = {}
    for item_id, info in index.items():
        raw = cv2.imread(os.path.join(ICON_CACHE_DIR, inv_type, info['filename']),
                         cv2.IMREAD_UNCHANGED)
        if raw is None or raw.ndim != 3 or raw.shape[2] != 4:
            continue
        ch, cw = raw.shape[:2]
        views = []
        for frac in CANVAS_SCALES[inv_type]:
            w = int(cell_w * frac)
            h = int(w * ch / cw)
            tmpl = cv2.resize(raw[:, :, :3], (w, h), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(raw[:, :, 3], (w, h), interpolation=cv2.INTER_AREA)
            mask = (alpha > 128).astype(np.uint8) * 255
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
            mask[int(h * (1 - RIBBON_FRAC)):, :] = 0
            if inv_type == 'equipment':
                mask[:int(h * BADGE_H), :int(w * BADGE_W)] = 0
            if cv2.countNonZero(mask) < 50:
                continue
            views.append((tmpl, mask))
        if views:
            templates[item_id] = views
    return templates


def main() -> None:
    gt, files = parse_id_grid(GT_PATH)
    prod, _ = parse_id_grid(PRISTINE_PATH)
    clip_top1 = parse_diag_top1(DIAG_PATH)

    ncc_pred = {}
    ncc_margin = {}
    t0 = time.time()

    templates_cache = {}
    for name, inv_type in files:
        with open(os.path.join(DIRS[inv_type], name), 'rb') as fh:
            meta = _extract_grid(fh.read(), inv_type)
        grid, rows, cols, cell_w, row_bounds = meta
        gh, gw = grid.shape[:2]

        if inv_type not in templates_cache:
            templates_cache[inv_type] = build_templates(inv_type, cell_w)
        templates = templates_cache[inv_type]

        for row in range(rows):
            for col in range(cols):
                if (name, row, col) not in gt:
                    continue
                cx0, cy0 = int(col * cell_w), row_bounds[row]
                cx1, cy1 = int((col + 1) * cell_w), row_bounds[row + 1]
                ex0, ey0 = max(0, cx0 - MARGIN_X), max(0, cy0 - MARGIN_Y)
                ex1, ey1 = min(gw, cx1 + MARGIN_X), min(gh, cy1 + MARGIN_Y)
                window = grid[ey0:ey1, ex0:ex1]

                scores = []
                for item_id, views in templates.items():
                    best = -2.0
                    for tmpl, mask in views:
                        th, tw = tmpl.shape[:2]
                        if th >= window.shape[0] or tw >= window.shape[1]:
                            continue
                        res = cv2.matchTemplate(window, tmpl,
                                                cv2.TM_CCOEFF_NORMED, mask=mask)
                        res = np.nan_to_num(res, nan=-2.0, posinf=-2.0,
                                            neginf=-2.0)
                        best = max(best, float(res.max()))
                    scores.append((best, item_id))
                scores.sort(reverse=True)
                ncc_pred[(name, row, col)] = scores[0][1]
                ncc_margin[(name, row, col)] = scores[0][0] - scores[1][0]

        done = sum(1 for k in ncc_pred if k[0] == name)
        print(f'[ncc] {name}: {done} cells  ({time.time() - t0:.0f}s elapsed)')

    # ── scoring ──────────────────────────────────────────────────────────
    def score(pred):
        hits = sum(1 for k, true_id in gt.items() if pred.get(k) == true_id)
        return hits, len(gt)

    for label, pred in (('raw CLIP top-1', clip_top1),
                        ('CLIP + cheats (prod)', prod),
                        ('masked NCC', ncc_pred)):
        hits, total = score(pred)
        print(f'{label:>22}: {hits}/{total}  ({hits / total:.1%})')

    errs = [(k, gt[k], ncc_pred.get(k)) for k in sorted(gt)
            if ncc_pred.get(k) != gt[k]]
    if errs:
        print('\nmasked NCC errors:')
        for (name, r, c), true_id, pred_id in errs:
            print(f'  {name} r{r}c{c}: true={true_id} pred={pred_id} '
                  f'margin={ncc_margin.get((name, r, c), 0):.4f}')
    margins = np.array(list(ncc_margin.values()))
    print(f'\nNCC top1-top2 margins: min={margins.min():.4f} '
          f'p5={np.percentile(margins, 5):.4f} median={np.median(margins):.4f}')
    print(f'total time: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
