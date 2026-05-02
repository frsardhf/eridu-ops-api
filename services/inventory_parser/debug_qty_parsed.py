"""
Verify the parsed quantities for each cell after the regex fix.
Usage: python debug_qty_parsed.py items <img1> [img2 ...]
"""
import os
import sys

import cv2

from pipeline import (
    _compute_grid_bounds,
    _detect_row_boundaries,
    _read_quantity,
    QTY_CROP_TOP_FRAC, QTY_CROP_OVERFLOW,
    QTY_CROP_LEFT_FRAC, QTY_CROP_RIGHT_FRAC,
)


def dump(img_path: str, label: str) -> None:
    print(f"\n{'='*60}\n{label}: {os.path.basename(img_path)}\n{'='*60}")
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"  ERROR: cannot read {img_path}")
        return

    h, w = image.shape[:2]
    right_half = image[:, w // 2:]
    left, top, right_px, bottom = _compute_grid_bounds(right_half)
    grid = right_half[top:bottom, left:right_px]
    gh, gw = grid.shape[:2]
    cell_w = gw / 5
    row_bounds = _detect_row_boundaries(grid, n_rows=4)

    for row in range(4):
        line_parts = []
        for col in range(5):
            x0 = int(col * cell_w);  x1 = int((col + 1) * cell_w)
            y0 = row_bounds[row];    y1 = row_bounds[row + 1]
            sh, sw = y1 - y0, x1 - x0
            qty_top = y0 + int(sh * QTY_CROP_TOP_FRAC)
            qty_bot = min(y1 + int(sh * QTY_CROP_OVERFLOW), grid.shape[0])
            qty_left = x0 + int(sw * QTY_CROP_LEFT_FRAC)
            qty_right = min(x0 + int(sw * QTY_CROP_RIGHT_FRAC), grid.shape[1])
            qty_crop = grid[qty_top:qty_bot, qty_left:qty_right]
            qty, conf = _read_quantity(qty_crop)
            line_parts.append(f"r{row}c{col}={qty:>5}({conf:.2f})")
        print("  " + "  ".join(line_parts))


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python debug_qty_parsed.py <items|equipment> <img1> [img2 ...]")
        sys.exit(1)
    for i, p in enumerate(sys.argv[2:], 1):
        dump(p, f"img{i}")


if __name__ == '__main__':
    main()
