"""Build a template override from a verified in-game render.

For items whose SchaleDB sprite drifted from what the game actually draws
(e.g. JP cover art on EN clients — gifts 5009/5023), this crops the sprite
canvas out of a screenshot cell whose true item ID is known, resamples it to
the 146x116 canvas, attaches the ORIGINAL sprite's alpha (the artwork shape
is identical, only surface art differs), and writes a BGRA PNG to
assets/icon_overrides/<inv_type>/. ncc_matcher prefers overrides at load.

Usage (from services/inventory_parser/ with .venv active):
    python debug_make_override.py <screenshot> <items|equipment> <row> <col> <item_id>
e.g.
    python debug_make_override.py assets/Screenshot9.png items 3 0 5023
"""
import json
import os
import sys

import cv2
import numpy as np

import ncc_matcher as m
from pipeline import _extract_grid


def main() -> None:
    if len(sys.argv) != 6:
        print(__doc__)
        sys.exit(1)
    img_path, inv_type = sys.argv[1], sys.argv[2]
    row, col, item_id = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]

    with open(img_path, 'rb') as fh:
        grid, rows, cols, cell_w, rb = _extract_grid(fh.read(), inv_type)
    win = m.cell_window(grid, cell_w, rb, row, col).astype(np.float32)
    bank = m.get_bank(inv_type, cell_w)

    # Locate the canvas: dense stride-2 sweep per scale, then 1px polish.
    best = None   # (score, lvl, (x, y))
    for lvl in bank['f']:
        if item_id not in lvl['idx']:
            continue
        ranked, pos = m._stage_union(win, lvl, [item_id], {}, radius=0, stride=2)
        if not ranked:
            continue
        p_ranked, p_pos = m._stage_union(win, lvl, [item_id],
                                         {item_id: pos[item_id]},
                                         radius=2, stride=1)
        score = p_ranked[0][0] if p_ranked else ranked[0][0]
        at = p_pos.get(item_id, pos[item_id])
        if best is None or score > best[0]:
            best = (score, lvl, at)
    if best is None:
        print(f'no full-res view for {item_id}')
        sys.exit(1)

    score, lvl, (x, y) = best
    crop = win[y:y + lvl['h'], x:x + lvl['w']].astype(np.uint8)
    canvas_bgr = cv2.resize(crop, (146, 116), interpolation=cv2.INTER_AREA)

    index = json.load(open(os.path.join(m.CACHE_DIR,
                                        f'icon_index_{inv_type}.json')))['items']
    filename = index[item_id]['filename']
    sprite = cv2.imread(os.path.join(m.ICON_CACHE_DIR, inv_type, filename),
                        cv2.IMREAD_UNCHANGED)
    out = np.dstack([canvas_bgr, sprite[:, :, 3]])

    out_dir = os.path.join(m.OVERRIDE_DIR, inv_type)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(filename)[0] + '.png')
    cv2.imwrite(out_path, out)
    print(f'{item_id}: matched score={score:.4f} at ({x},{y}) '
          f'scale_w={lvl["w"]}px → {out_path}')


if __name__ == '__main__':
    main()
