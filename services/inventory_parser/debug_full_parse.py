"""
Run the full parse_inventory pipeline on one or more screenshots and print the
sorted result table (itemId, rarity, quantity, row, col).

Usage (from services/inventory_parser/ with .venv active):
    python debug_full_parse.py /path/to/*.png items
    python debug_full_parse.py /path/to/img7.png /path/to/img8.png items
"""
import sys
import os

from pipeline import parse_inventory


def run_one(img_path: str, inv_type: str) -> None:
    print(f'\n{"="*60}')
    print(f' {os.path.basename(img_path)}  [{inv_type}]')
    print(f'{"="*60}')
    with open(img_path, 'rb') as f:
        img_bytes = f.read()

    results = parse_inventory(img_bytes, inv_type)
    if not results:
        print('  (no results)')
        return

    # Sort by grid position
    results.sort(key=lambda r: r['row'] * 5 + r['col'])
    prev_id = None
    for r in results:
        mono = ''
        if prev_id is not None:
            if int(r['itemId']) > prev_id:
                mono = ' ↑'
            elif int(r['itemId']) < prev_id:
                mono = ' ↓'
            else:
                mono = ' ='
        print(f"  r{r['row']}c{r['col']}  id={r['itemId']:>8s}  {r['rarity']:<4}  "
              f"qty={r.get('quantity', '?'):>8}  conf={r.get('confidence', 0):.3f}{mono}")
        prev_id = int(r['itemId'])


def main() -> None:
    if len(sys.argv) < 3:
        print('Usage: python debug_full_parse.py <img1.png> [img2.png ...] <items|equipment>')
        sys.exit(1)

    inv_type = sys.argv[-1]
    if inv_type not in ('items', 'equipment'):
        print(f'ERROR: last arg must be items or equipment, got: {inv_type!r}')
        sys.exit(1)

    for img_path in sys.argv[1:-1]:
        if not os.path.exists(img_path):
            print(f'SKIP: not found: {img_path}')
            continue
        run_one(img_path, inv_type)


if __name__ == '__main__':
    main()
