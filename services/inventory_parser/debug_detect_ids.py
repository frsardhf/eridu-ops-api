"""Detection-only ground-truth dump — NO Gemini calls.

Runs the production icon path (_extract_grid + _classify_and_correct with
qty_lookup=None) on all screenshots in ITEMS_DIR and EQUIPMENT_DIR and writes
the detected item IDs to a single txt for manual ground-truth marking:

    # <screenshot filename> [items|equipment]
    10 11 12 13 100
    ...            <- 5 space-separated IDs per grid row, '-' = empty/dropped cell

A companion *_diag.txt logs, per cell, the NCC score and top1-top2 margin so
weak matches stand out. Useful when validating a new device/resolution: dump,
hand-mark the wrong IDs, then score with debug_c2f_eval.py.

Usage (from services/inventory_parser/ with .venv active):
    python debug_detect_ids.py
"""
import glob
import os

import ncc_matcher
from pipeline import _classify_and_correct, _extract_grid

ITEMS_DIR     = '/Users/frsardhf/Downloads/Images/Items'
EQUIPMENT_DIR = '/Users/frsardhf/Downloads/Images/Equipment'
# NOTE: detected_ids.txt is the hand-marked ground truth — never write there.
OUT_PATH      = '/Users/frsardhf/Downloads/Images/detected_ids_pred.txt'
DIAG_PATH     = '/Users/frsardhf/Downloads/Images/detected_ids_pred_diag.txt'


def main() -> None:
    out_lines, diag_lines = [], []

    for folder, inv_type in ((ITEMS_DIR, 'items'), (EQUIPMENT_DIR, 'equipment')):
        files = sorted(glob.glob(os.path.join(folder, '*.png')))
        if not files:
            print(f'[gt] no screenshots in {folder}')
            continue

        for fp in files:
            name = os.path.basename(fp)
            with open(fp, 'rb') as fh:
                img_bytes = fh.read()

            meta = _extract_grid(img_bytes, inv_type)
            if meta is None:
                out_lines.append(f'# {name} [{inv_type}] — GRID NOT FOUND')
                print(f'[gt] {name}: grid not found')
                continue
            grid, rows, cols, cell_w, row_bounds = meta

            final = _classify_and_correct(grid, rows, cols, cell_w, row_bounds,
                                          inv_type, None)
            final_map = {(r['row'], r['col']): r for r in final}

            bank = ncc_matcher.get_bank(inv_type, cell_w)
            out_lines.append(f'# {name} [{inv_type}]')
            diag_lines.append(f'\n=== {name} [{inv_type}] ===')
            for row in range(rows):
                vals = []
                for col in range(cols):
                    r = final_map.get((row, col))
                    vals.append(r['itemId'] if r else '-')
                    win = ncc_matcher.cell_window(grid, cell_w, row_bounds,
                                                  row, col)
                    iid, score, margin = ncc_matcher.match_window(win, bank)
                    diag_lines.append(
                        f'  r{row}c{col}  final={r["itemId"] if r else "-":>6} '
                        f' ncc={iid}({score:.4f})  margin={margin:.4f}')
                out_lines.append(' '.join(vals))

            print(f'[gt] {name}: {len(final_map)} cells detected')

    with open(OUT_PATH, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out_lines) + '\n')
    with open(DIAG_PATH, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(diag_lines) + '\n')
    print(f'\n[gt] wrote {OUT_PATH}')
    print(f'[gt] wrote {DIAG_PATH}')


if __name__ == '__main__':
    main()
