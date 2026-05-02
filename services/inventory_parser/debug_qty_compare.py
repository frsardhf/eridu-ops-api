"""
Compare OCR output WITH vs WITHOUT the unsharp mask preprocessing,
to see if sharpening helps or hurts overall accuracy.
Usage: python debug_qty_compare.py items <img1> [img2 ...]
"""
import os
import re
import sys

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


def read_quantity_variant(roi_bgr: np.ndarray, sharpen: bool):
    reader = _get_or_load_easyocr()
    h, w = roi_bgr.shape[:2]
    if h == 0 or w == 0:
        return 0, 0.0, ''

    scaled = cv2.resize(
        roi_bgr,
        (w * _QTY_OCR_SCALE, h * _QTY_OCR_SCALE),
        interpolation=cv2.INTER_CUBIC,
    )

    if sharpen:
        _gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        _blur = cv2.GaussianBlur(_gray, (0, 0), 1.5)
        _sharp = cv2.addWeighted(_gray, 1.8, _blur, -0.8, 0)
        scaled = cv2.cvtColor(_sharp, cv2.COLOR_GRAY2BGR)

    results = reader.readtext(scaled, allowlist='0123456789kKmMxX×', detail=1)
    if not results:
        return 0, 0.0, ''

    img_h = scaled.shape[0]

    def _qty_priority(result):
        bbox = result[0]
        cy = sum(pt[1] for pt in bbox) / len(bbox)
        is_bottom = int(cy > img_h * 0.45)
        text = result[1].upper()
        has_x = int(bool(re.match(r'^[X×]', text)))
        raw = re.sub(r'^[X×]+', '', text)
        raw = re.sub(r'[KM]$', '', raw)
        n_digits = sum(c.isdigit() for c in raw)
        return (is_bottom, has_x, n_digits, float(result[2]))

    best = max(results, key=_qty_priority)
    text: str = best[1].upper()
    conf: float = float(best[2])
    raw_text = text

    # Apply the new fix
    m = re.match(r'^.{0,2}[X×]', text)
    text = text[m.end():] if m else re.sub(r'^[X×]+', '', text)

    multiplier = 1
    if text.endswith('M'):
        multiplier = 1_000_000; text = text[:-1]
    elif text.endswith('K'):
        multiplier = 1_000; text = text[:-1]

    digits = re.sub(r'[^0-9]', '', text)
    if not digits:
        return 0, conf, raw_text
    try:
        quantity = int(digits) * multiplier
    except ValueError:
        quantity = 0
    return quantity, conf, raw_text


def dump(img_path: str, label: str) -> None:
    print(f"\n{'='*78}\n{label}: {os.path.basename(img_path)}\n{'='*78}")
    print(f"{'cell':<8}{'sharp_qty':<12}{'sharp_raw':<14}{'plain_qty':<12}{'plain_raw':<14}{'diff'}")
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        return

    h, w = image.shape[:2]
    right_half = image[:, w // 2:]
    left, top, right_px, bottom = _compute_grid_bounds(right_half)
    grid = right_half[top:bottom, left:right_px]
    gh, gw = grid.shape[:2]
    cell_w = gw / 5
    row_bounds = _detect_row_boundaries(grid, n_rows=4)

    for row in range(4):
        for col in range(5):
            x0 = int(col * cell_w);  x1 = int((col + 1) * cell_w)
            y0 = row_bounds[row];    y1 = row_bounds[row + 1]
            sh, sw = y1 - y0, x1 - x0
            qty_top = y0 + int(sh * QTY_CROP_TOP_FRAC)
            qty_bot = min(y1 + int(sh * QTY_CROP_OVERFLOW), grid.shape[0])
            qty_left = x0 + int(sw * QTY_CROP_LEFT_FRAC)
            qty_right = min(x0 + int(sw * QTY_CROP_RIGHT_FRAC), grid.shape[1])
            qty_crop = grid[qty_top:qty_bot, qty_left:qty_right]

            sharp_q, _, sharp_raw = read_quantity_variant(qty_crop, sharpen=True)
            plain_q, _, plain_raw = read_quantity_variant(qty_crop, sharpen=False)
            diff = '<-- DIFF' if sharp_q != plain_q else ''
            print(f"r{row}c{col}    {sharp_q:<12}{sharp_raw:<14}{plain_q:<12}{plain_raw:<14}{diff}")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python debug_qty_compare.py <items|equipment> <img1> [img2 ...]")
        sys.exit(1)
    for i, p in enumerate(sys.argv[2:], 1):
        dump(p, f"img{i}")


if __name__ == '__main__':
    main()
