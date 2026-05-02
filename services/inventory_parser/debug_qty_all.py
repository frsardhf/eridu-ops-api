"""
One-off debug — dumps quantity crops + OCR detections for ALL 20 cells
across one or more screenshots, to diagnose mis-reads.

Usage:
  python debug_qty_all.py items /path/to/img1.png /path/to/img2.png ...
"""
import os
import sys
import re

import cv2
import numpy as np

from pipeline import (
    _compute_grid_bounds,
    _detect_row_boundaries,
    _get_or_load_easyocr,
    QTY_CROP_TOP_FRAC, QTY_CROP_OVERFLOW,
    QTY_CROP_LEFT_FRAC, QTY_CROP_RIGHT_FRAC,
    _QTY_OCR_SCALE,
)


def dump_for_screenshot(img_path: str, out_dir: str, label: str) -> None:
    print(f"\n{'='*70}\n{label}: {os.path.basename(img_path)}\n{'='*70}")
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"  ERROR: cv2 could not read {img_path}")
        return

    h, w = image.shape[:2]
    right_half = image[:, w // 2:]
    left, top, right_px, bottom = _compute_grid_bounds(right_half)
    grid = right_half[top:bottom, left:right_px]
    gh, gw = grid.shape[:2]
    cell_w = gw / 5
    row_bounds = _detect_row_boundaries(grid, n_rows=4)

    reader = _get_or_load_easyocr()
    os.makedirs(out_dir, exist_ok=True)

    for row in range(4):
        for col in range(5):
            x0 = int(col * cell_w);  x1 = int((col + 1) * cell_w)
            y0 = row_bounds[row];    y1 = row_bounds[row + 1]
            sh, sw = y1 - y0, x1 - x0
            if sh <= 0 or sw <= 0:
                continue

            qty_top = y0 + int(sh * QTY_CROP_TOP_FRAC)
            qty_bot = min(y1 + int(sh * QTY_CROP_OVERFLOW), grid.shape[0])
            qty_left = x0 + int(sw * QTY_CROP_LEFT_FRAC)
            qty_right = min(x0 + int(sw * QTY_CROP_RIGHT_FRAC), grid.shape[1])
            qty_crop = grid[qty_top:qty_bot, qty_left:qty_right]
            if qty_crop.size == 0:
                continue

            # Match the live OCR pipeline preprocessing
            scaled = cv2.resize(
                qty_crop,
                (qty_crop.shape[1] * _QTY_OCR_SCALE, qty_crop.shape[0] * _QTY_OCR_SCALE),
                interpolation=cv2.INTER_CUBIC,
            )
            _gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
            _blur = cv2.GaussianBlur(_gray, (0, 0), 1.5)
            _sharp = cv2.addWeighted(_gray, 1.8, _blur, -0.8, 0)
            scaled = cv2.cvtColor(_sharp, cv2.COLOR_GRAY2BGR)

            results = reader.readtext(
                scaled,
                allowlist='0123456789kKmMxX×',
                detail=1,
            )

            # Save the raw + scaled crops
            cell_label = f"{label}_r{row}c{col}"
            cv2.imwrite(os.path.join(out_dir, f"{cell_label}_raw.png"), qty_crop)
            cv2.imwrite(os.path.join(out_dir, f"{cell_label}_scaled.png"), scaled)

            # Print OCR detections w/ bboxes
            print(f"\n  [{cell_label}]  raw={qty_crop.shape[1]}x{qty_crop.shape[0]}  "
                  f"scaled={scaled.shape[1]}x{scaled.shape[0]}")
            if not results:
                print("    (no OCR detections)")
                continue
            for r in results:
                bbox, text, conf = r
                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                bx, by = int(min(xs)), int(min(ys))
                bw, bh = int(max(xs) - min(xs)), int(max(ys) - min(ys))
                print(f"    text={text!r:>10}  conf={conf:.3f}  "
                      f"bbox=(x={bx} y={by} w={bw} h={bh})")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python debug_qty_all.py <items|equipment> <img1> [img2 ...]")
        sys.exit(1)
    inv_type = sys.argv[1]
    paths = sys.argv[2:]

    out_dir = os.path.join(os.path.dirname(os.path.abspath(paths[0])), 'debug_qty_out')
    print(f"Writing debug output to: {out_dir}")

    for i, path in enumerate(paths, 1):
        dump_for_screenshot(path, out_dir, f"img{i}")

    print(f"\nDone. {out_dir}/")


if __name__ == '__main__':
    main()
