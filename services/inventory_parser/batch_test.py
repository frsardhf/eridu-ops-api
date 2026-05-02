"""Batch test — run pipeline on all screenshots and report sort-order violations."""
import glob
import os
import sys

from pipeline import parse_inventory

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')


def test_all():
    screenshots = sorted(glob.glob(os.path.join(ASSETS_DIR, 'Screenshot*.png')))
    print(f'Found {len(screenshots)} screenshots\n')

    total_violations = 0
    total_items = 0

    for spath in screenshots:
        fname = os.path.basename(spath)
        with open(spath, 'rb') as f:
            data = f.read()
        results = parse_inventory(data, 'items')
        if not results:
            print(f'{fname}: NO RESULTS')
            continue

        ordered = sorted(results, key=lambda r: r['row'] * 5 + r['col'])
        ids = [int(r['itemId']) for r in ordered]
        n = len(ids)
        total_items += n

        violations = []
        for i in range(1, n):
            if ids[i] <= ids[i - 1]:
                violations.append((i - 1, i, ids[i - 1], ids[i]))

        total_violations += len(violations)
        status = f'{len(violations)} violations' if violations else 'OK'
        print(f'{fname}: {n}/20 items, {status}')
        for prev_i, cur_i, prev_id, cur_id in violations:
            r_prev = ordered[prev_i]
            r_cur = ordered[cur_i]
            print(f'  [{r_prev["row"]},{r_prev["col"]}] id={prev_id} >= '
                  f'[{r_cur["row"]},{r_cur["col"]}] id={cur_id}')

    print(f'\nTotal: {total_items} items, {total_violations} violations')


if __name__ == '__main__':
    test_all()
