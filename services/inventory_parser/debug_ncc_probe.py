"""Calibration probe v2 for masked template matching.

All SchaleDB sprites share a 146x116 BGRA canvas, so the game renders that
canvas at ONE fixed scale per cell. This probe slides the full-canvas sprite
(alpha-masked) over an expanded slot window at a fine scale sweep and reports
the best (canvas_scale, x, y, score) per known cell. Convergence across cells
means the production matcher can use fixed geometry.

Usage: python debug_ncc_probe.py
"""
import json
import os

import cv2
import numpy as np

from pipeline import _extract_grid, CACHE_DIR, ICON_CACHE_DIR

ITEMS_DIR     = '/Users/frsardhf/Downloads/Images/Items'
EQUIPMENT_DIR = '/Users/frsardhf/Downloads/Images/Equipment'
MARGIN = 30  # expansion around the slot so an oversized canvas still fits

# (folder, filename, inv_type, row, col, true_id)
PROBES = [
    (ITEMS_DIR, 'Screenshot_2026-06-09_165712.png', 'items', 0, 0, '10'),
    (ITEMS_DIR, 'Screenshot_2026-06-09_165712.png', 'items', 1, 2, '103'),
    (ITEMS_DIR, 'Screenshot_2026-06-09_165712.png', 'items', 3, 3, '132'),
    (ITEMS_DIR, 'Screenshot_2026-06-09_165754.png', 'items', 2, 1, '3010'),
    (ITEMS_DIR, 'Screenshot_2026-06-09_165754.png', 'items', 3, 4, '3030'),
    (ITEMS_DIR, 'Screenshot_2026-06-09_165842.png', 'items', 3, 4, '5999'),
    (EQUIPMENT_DIR, 'Screenshot_2026-06-09_165909.png', 'equipment', 0, 0, '1001'),
    (EQUIPMENT_DIR, 'Screenshot_2026-06-09_165909.png', 'equipment', 2, 2, '2004'),
    (EQUIPMENT_DIR, 'Screenshot_2026-06-09_165916.png', 'equipment', 4, 4, '6005'),
]


def load_sprite_canvas(inv_type: str, item_id: str):
    """Return (bgr, alpha) of the FULL 146x116 sprite canvas."""
    index_path = os.path.join(CACHE_DIR, f'icon_index_{inv_type}.json')
    with open(index_path, encoding='utf-8') as fh:
        index = json.load(fh)['items']
    info = index.get(item_id)
    if info is None:
        return None, None
    path = os.path.join(ICON_CACHE_DIR, inv_type, info['filename'])
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 3 or raw.shape[2] != 4:
        return None, None
    return raw[:, :, :3], raw[:, :, 3]


def main() -> None:
    grid_cache = {}
    for folder, name, inv_type, row, col, true_id in PROBES:
        key = (folder, name)
        if key not in grid_cache:
            with open(os.path.join(folder, name), 'rb') as fh:
                grid_cache[key] = _extract_grid(fh.read(), inv_type)
        grid, rows, cols, cell_w, row_bounds = grid_cache[key]
        gh, gw = grid.shape[:2]

        cx0, cy0 = int(col * cell_w), row_bounds[row]
        cx1, cy1 = int((col + 1) * cell_w), row_bounds[row + 1]
        ex0, ey0 = max(0, cx0 - MARGIN), max(0, cy0 - MARGIN)
        ex1, ey1 = min(gw, cx1 + MARGIN), min(gh, cy1 + MARGIN)
        window = grid[ey0:ey1, ex0:ex1]

        bgr, alpha = load_sprite_canvas(inv_type, true_id)
        if bgr is None:
            print(f'{name} r{row}c{col}: sprite for {true_id} not found')
            continue
        ch, cw = bgr.shape[:2]

        best = None
        for frac in np.arange(0.60, 1.21, 0.02):
            w = int(cell_w * frac)
            h = int(w * ch / cw)
            if w >= window.shape[1] or h >= window.shape[0] or h < 8:
                continue
            tmpl = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(window, tmpl, cv2.TM_CCORR_NORMED, mask=mask)
            res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if best is None or max_val > best[0]:
                best = (max_val, frac, max_loc[0], max_loc[1], w, h)

        score, frac, bx, by, w, h = best
        # Position of the canvas top-left relative to the CELL origin
        rx, ry = ex0 + bx - cx0, ey0 + by - cy0
        print(f'{name} r{row}c{col} id={true_id:>5}: cell_w={cell_w:.0f} '
              f'slot_h={cy1-cy0}  canvas_scale={frac:.2f} ({w}x{h}px)  '
              f'offset_in_cell=({rx},{ry})  score={score:.4f}')


if __name__ == '__main__':
    main()
