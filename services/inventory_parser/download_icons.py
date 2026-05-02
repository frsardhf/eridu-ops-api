"""Fetch item/equipment data from schaledb and build icon_index_*.json.

Run before embed.py whenever new items are added to the game:

    python download_icons.py items
    python download_icons.py equipment
    python download_icons.py          # both

Outputs written to cache/:
    icon_index_{type}.json            — {item_id: {filename, rarity}}
    icons/{type}/{filename}.webp      — icon images (used by embed.py)
"""

import json
import os
import sys
import time
from urllib import request, error

BASE_DIR  = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, 'cache')

SCHALEDB_BASE_URL = 'https://schaledb.com/data/en'
SCHALEDB_IMG_URL  = 'https://schaledb.com/images'

_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; eridu-ops-inventory-parser/1.0)',
    'Accept': 'application/json, image/webp, */*',
}

# Mirror of pipeline.py ITEM_INCLUDE_FILTER / EQUIPMENT_INCLUDE_FILTER.
# Keep these in sync with src/types/resource.ts MATERIAL / EQUIPMENT constants.
ITEM_INCLUDE_FILTER = {
    'category':    {'CharacterExpGrowth', 'Favor'},
    'subcategory': {'Artifact', 'CDItem', 'BookItem'},
    'id':          {5, 23, 2000, 2001, 2002, 9999},
}
EQUIPMENT_INCLUDE_FILTER = {
    'category':    {'Exp', 'WeaponExpGrowthA', 'WeaponExpGrowthB', 'WeaponExpGrowthC', 'WeaponExpGrowthZ'},
    'recipecost':  {1500, 10000, 25000, 50000, 75000, 100000, 125000, 150000, 175000},
}


def _open_url(url: str, timeout: int = 20):
    req = request.Request(url, headers=_HEADERS)
    return request.urlopen(req, timeout=timeout)


def _fetch_json(url: str) -> dict:
    try:
        with _open_url(url) as response:
            return json.loads(response.read().decode('utf-8'))
    except (error.URLError, json.JSONDecodeError) as exc:
        print(f'ERROR: Failed to fetch {url}: {exc}')
        return {}


def _download_icon(url: str, dest_path: str) -> bool:
    """Download a single icon. Returns True on success, False on failure."""
    if os.path.exists(dest_path):
        return True  # already cached
    try:
        with _open_url(url, timeout=15) as response:
            data = response.read()
        with open(dest_path, 'wb') as fh:
            fh.write(data)
        return True
    except error.URLError as exc:
        print(f'  WARN: failed to download {url}: {exc}')
        return False


def _item_passes_filter(item: dict, inventory_type: str) -> bool:
    """Return True if the item should be included in the icon index."""
    if inventory_type == 'items':
        f = ITEM_INCLUDE_FILTER
        return (
            item.get('Category') in f['category']
            or item.get('SubCategory') in f['subcategory']
            or item.get('Id') in f['id']
        )
    # equipment
    f = EQUIPMENT_INCLUDE_FILTER
    return (
        item.get('Category') in f['category']
        or item.get('RecipeCost') in f['recipecost']
    )


def _icon_filename(item: dict, inventory_type: str) -> str:
    """Return the .webp filename for the icon (without directory)."""
    icon = item['Icon']
    if inventory_type == 'equipment' and item.get('Tier', 0) != 0:
        icon = f'{icon}_piece'
    return f'{icon}.webp'


def _icon_url(filename: str, inventory_type: str) -> str:
    kind = 'equipment' if inventory_type == 'equipment' else 'item'
    return f'{SCHALEDB_IMG_URL}/{kind}/icon/{filename}'


def download(inv_type: str) -> None:
    print(f'\n=== Downloading icons for: {inv_type} ===')

    icons_dir = os.path.join(CACHE_DIR, 'icons', inv_type)
    os.makedirs(icons_dir, exist_ok=True)

    endpoint = f'{SCHALEDB_BASE_URL}/{"equipment" if inv_type == "equipment" else "items"}.json'
    print(f'Fetching {endpoint} ...')
    raw = _fetch_json(endpoint)
    if not raw:
        print('ERROR: empty response from schaledb — aborting.')
        sys.exit(1)

    # Filter to only the items we care about
    filtered = {k: v for k, v in raw.items() if _item_passes_filter(v, inv_type)}
    print(f'Filtered {len(filtered)} / {len(raw)} items')

    # Download icons
    ok = fail = skip = 0
    for item_id, item in filtered.items():
        filename = _icon_filename(item, inv_type)
        dest = os.path.join(icons_dir, filename)
        if os.path.exists(dest):
            skip += 1
            continue
        url = _icon_url(filename, inv_type)
        if _download_icon(url, dest):
            ok += 1
        else:
            fail += 1
        time.sleep(0.05)  # be polite to schaledb

    print(f'Icons: {ok} downloaded, {skip} already cached, {fail} failed')

    # Write icon index
    index_path = os.path.join(CACHE_DIR, f'icon_index_{inv_type}.json')
    index_data = {
        'generatedAt': time.time(),
        'items': {
            item_id: {
                'filename': _icon_filename(item, inv_type),
                'rarity':   item.get('Rarity', 'N'),
            }
            for item_id, item in filtered.items()
        },
    }
    with open(index_path, 'w', encoding='utf-8') as fh:
        json.dump(index_data, fh, ensure_ascii=False, indent=2)
    print(f'Index saved → {index_path}  ({len(index_data["items"])} entries)')


def main() -> None:
    types_arg = sys.argv[1:] if len(sys.argv) > 1 else ['items', 'equipment']
    for t in types_arg:
        if t not in ('items', 'equipment'):
            print(f'ERROR: unknown type "{t}". Use: items, equipment')
            sys.exit(1)
        download(t)
    print('\nDone. Now run: python embed.py items && python embed.py equipment')


if __name__ == '__main__':
    main()
