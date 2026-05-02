"""
Debug script — run from services/inventory_parser/ with .venv active.
Usage: python debug_parse.py /path/to/screenshot.png items|equipment
"""
import json
import os
import sys

import cv2
import numpy as np

# ── import pipeline helpers ──────────────────────────────────────────────────
from pipeline import (
    CACHE_DIR, ICON_INDEX_ITEMS, ICON_INDEX_EQUIPMENT,
    CLIP_INPUT_SIZE, EMBED_SCORE_THRESHOLD,
    ICON_CROP_TOP, ICON_CROP_BOT, ICON_CROP_LEFT, ICON_CROP_RIGHT,
    _compute_grid_bounds, _find_panel_bounds,
    _classify_rarity, _load_embeddings, _load_rarities,
    _classify_icon_embed, _get_embed_scores,
    _extract_icon_crop,
    GRID_FALLBACK_LEFT_RATIO, GRID_FALLBACK_RIGHT_RATIO,
    GRID_FALLBACK_TOP_RATIO, GRID_FALLBACK_BOTTOM_RATIO,
)

# Alias kept for backward compat within this script
_icon_crop_from_slot = _extract_icon_crop


def check_icon_index(inv_type: str) -> int:
    path = ICON_INDEX_ITEMS if inv_type == 'items' else ICON_INDEX_EQUIPMENT
    if not os.path.exists(path):
        print(f'  [MISSING] {path}')
        return 0
    with open(path) as f:
        data = json.load(f)
    items = data.get('items', {})
    count = len(items)
    print(f'  [OK] {path}  →  {count} entries')
    return count


def check_embed_model(inv_type: str) -> bool:
    emb_path    = os.path.join(CACHE_DIR, f'icon_embeddings_{inv_type}.npy')
    labels_path = os.path.join(CACHE_DIR, f'icon_embeddings_{inv_type}_labels.json')
    missing = [p for p in (emb_path, labels_path) if not os.path.exists(p)]
    if missing:
        print(f'  [MISSING] embeddings not found — run: python embed.py {inv_type}')
        for p in missing:
            print(f'    missing: {os.path.basename(p)}')
        return False
    with open(labels_path) as f:
        labels = json.load(f)
    print(f'  [OK] {os.path.basename(emb_path)}  →  {len(labels)} classes')
    return True


def save_debug_images(image: np.ndarray, right_half: np.ndarray,
                      left: int, top: int, right: int, bottom: int,
                      out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Full image with right-half boundary
    h, w = image.shape[:2]
    vis_full = image.copy()
    cv2.line(vis_full, (w // 2, 0), (w // 2, h), (0, 255, 255), 2)
    cv2.imwrite(os.path.join(out_dir, '1_full_with_split.png'), vis_full)

    # Right half with detected grid rect
    vis_half = right_half.copy()
    cv2.rectangle(vis_half, (left, top), (right, bottom), (0, 255, 0), 3)
    cv2.putText(vis_half, f'grid: ({left},{top}) -> ({right},{bottom})',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(out_dir, '2_right_half_grid.png'), vis_half)

    # Grid crop
    grid = right_half[top:bottom, left:right]
    if grid.size > 0:
        cv2.imwrite(os.path.join(out_dir, '3_grid_crop.png'), grid)

        # Each cell
        rows, cols = 4, 5
        gh, gw = grid.shape[:2]
        cell_w, cell_h = gw / cols, gh / rows
        vis_cells = grid.copy()
        for r in range(rows):
            for c in range(cols):
                x0, y0 = int(c * cell_w), int(r * cell_h)
                x1, y1 = int((c + 1) * cell_w), int((r + 1) * cell_h)
                cv2.rectangle(vis_cells, (x0, y0), (x1, y1), (0, 200, 255), 1)
                cv2.putText(vis_cells, f'{r},{c}', (x0 + 2, y0 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(out_dir, '4_grid_cells.png'), vis_cells)

    print(f'  Debug images saved to: {out_dir}/')


def main() -> None:
    if len(sys.argv) < 3:
        print('Usage: python debug_parse.py <screenshot.png> <items|equipment>')
        sys.exit(1)

    img_path = sys.argv[1]
    inv_type = sys.argv[2]

    if not os.path.exists(img_path):
        print(f'ERROR: file not found: {img_path}')
        sys.exit(1)

    print(f'\n{"="*60}')
    print(f' Debug parse: {os.path.basename(img_path)} [{inv_type}]')
    print(f'{"="*60}\n')

    # ── 1. Icon index ─────────────────────────────────────────────────────────
    print('1. Icon index')
    n_items = check_icon_index(inv_type)
    if n_items == 0:
        print('  WARNING: no index — run warmup first (python app.py then Ctrl-C)')

    # ── 1b. Embedding model ───────────────────────────────────────────────────
    print('\n1b. Embedding model')
    has_cnn = check_embed_model(inv_type)

    # ── 2. Load image ────────────────────────────────────────────────────────
    print('\n2. Loading image')
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f'  ERROR: cv2 could not read {img_path}')
        sys.exit(1)
    h, w = image.shape[:2]
    print(f'  Size: {w} x {h} px')

    # ── 3. Right-half crop ───────────────────────────────────────────────────
    print('\n3. Right-half crop')
    right_half = image[:, w // 2:]
    rh, rw = right_half.shape[:2]
    print(f'  Right half: {rw} x {rh} px  (x offset: {w // 2})')

    # ── 4. Panel detection ───────────────────────────────────────────────────
    print('\n4. Panel detection (largest bright contour)')
    panel = _find_panel_bounds(right_half)
    if panel:
        px, py, pw, ph = panel
        print(f'  Found: x={px}, y={py}, w={pw}, h={ph}')
    else:
        print('  Not found → using fallback ratios')
        print(f'    left={GRID_FALLBACK_LEFT_RATIO:.2f}  right={GRID_FALLBACK_RIGHT_RATIO:.2f}')
        print(f'    top ={GRID_FALLBACK_TOP_RATIO:.2f}  bottom={GRID_FALLBACK_BOTTOM_RATIO:.2f}')

    # ── 5. Grid bounds ───────────────────────────────────────────────────────
    print('\n5. Grid bounds')
    left, top, right_px, bottom = _compute_grid_bounds(right_half)
    print(f'  left={left}  top={top}  right={right_px}  bottom={bottom}')
    grid_w = right_px - left
    grid_h = bottom - top
    print(f'  Grid size: {grid_w} x {grid_h} px')
    if grid_w <= 0 or grid_h <= 0:
        print('  ERROR: invalid grid — nothing to parse')
        sys.exit(1)

    cell_w = grid_w / 5
    cell_h = grid_h / 4
    print(f'  Cell size: {cell_w:.1f} x {cell_h:.1f} px  (5 cols x 4 rows)')

    # ── 6. Save debug images ─────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(os.path.abspath(img_path)), 'debug_out')
    print('\n6. Saving debug images')
    save_debug_images(image, right_half, left, top, right_px, bottom, out_dir)

    # ── 7. Sample cell 0,0 — rarity + icon crop ──────────────────────────────
    print('\n7. Sampling cell (row=0, col=0)')
    grid = right_half[top:bottom, left:right_px]
    slot = grid[0:int(cell_h), 0:int(cell_w)]
    rarity, conf = _classify_rarity(slot)
    print(f'  Rarity: {rarity}  (confident={conf})')

    # Icon crop — mirrors pipeline.py square-normalise + ICON_CROP_* fractions.
    sh, sw = slot.shape[:2]
    icon_crop = _icon_crop_from_slot(slot)
    icon_resized = cv2.resize(icon_crop, (CLIP_INPUT_SIZE, CLIP_INPUT_SIZE))
    cv2.imwrite(os.path.join(out_dir, '5_icon_query_r0c0.png'), icon_resized)
    print(f'  Icon crop: {icon_crop.shape[1]}x{icon_crop.shape[0]} px  '
          f'→ resized to {CLIP_INPUT_SIZE}x{CLIP_INPUT_SIZE} for CLIP')
    print(f'  Crop fractions: top={ICON_CROP_TOP}  bot={ICON_CROP_BOT}  '
          f'left={ICON_CROP_LEFT}  right={ICON_CROP_RIGHT}')

    # Quantity crop + contour debug
    qty_crop = slot[int(sh * 0.74):int(sh * 0.99), int(sw * 0.40):int(sw * 0.98)]
    cv2.imwrite(os.path.join(out_dir, '6_qty_crop_r0c0.png'), qty_crop)
    gray_q = cv2.cvtColor(qty_crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_q, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(out_dir, '7_qty_thresh_r0c0.png'), thresh)
    qh, qw = thresh.shape
    min_area = max(20.0, 0.005 * qw * qh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [(x, y, bw, bh) for cnt in contours
             for x, y, bw, bh in [cv2.boundingRect(cnt)]
             if bw * bh >= min_area]
    boxes.sort(key=lambda b: b[0])
    print(f'  Quantity ROI: {qw}x{qh} px  |  {len(boxes)} digit contours  '
          f'(min_area={min_area:.1f})')
    for i, (x, y, bw, bh) in enumerate(boxes):
        print(f'    contour {i}: x={x} y={y} w={bw} h={bh} area={bw*bh}')
    vis_qty = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    for x, y, bw, bh in boxes:
        cv2.rectangle(vis_qty, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(out_dir, '8_qty_contours_r0c0.png'), vis_qty)

    # ── 8. Embedding top-3 for cell (0,0) ────────────────────────────────────
    print('\n8. Embedding top-3 for cell (0,0)')
    if not has_cnn:
        print('  (skipped — no embed model)')
    else:
        net, embeddings, labels = _load_embeddings(inv_type)
        rarities = _load_rarities(inv_type)
        cos_scores = _get_embed_scores(icon_crop, net, embeddings)
        top3 = sorted(enumerate(cos_scores), key=lambda x: -x[1])[:3]
        for idx, sc in top3:
            item_id  = labels.get(str(idx), '?')
            item_rar = rarities.get(str(idx), '?')
            flag = '  ← MATCH' if sc >= EMBED_SCORE_THRESHOLD else ''
            print(f'  [{idx:>4}] item_id={item_id:>8s}  rarity={item_rar:<3}  cosine={sc:.4f}{flag}')
        # Show rarity-aware final selection
        final_id, final_score = _classify_icon_embed(
            icon_crop, net, embeddings, labels, rarities, rarity,
        )
        print(f'  → rarity-aware pick: item_id={final_id}  '
              f'(cell rarity={rarity}, score={final_score:.4f})')
        best_score = top3[0][1] if top3 else 0.0
        if best_score < EMBED_SCORE_THRESHOLD:
            print(f'\n  NOTE: best cosine {best_score:.4f} < threshold {EMBED_SCORE_THRESHOLD}')
            print('  Possible causes:')
            print('   • Cell (0,0) is empty or a partial item')
            print('   • Grid crop is misaligned (check debug_out/4_grid_cells.png)')
            print('   • Re-run: python embed.py ' + inv_type)

    # ── 9. All 20 cells quick scan ───────────────────────────────────────────
    print('\n9. Scanning all 20 cells')
    if not has_cnn:
        print('  (skipped — no embed model)')
    else:
        net, embeddings, labels = _load_embeddings(inv_type)
        rarities = _load_rarities(inv_type)
        hits = 0
        for row in range(4):
            row_parts = []
            for col in range(5):
                x0 = int(col * cell_w);  x1 = int((col + 1) * cell_w)
                y0 = int(row * cell_h);  y1 = int((row + 1) * cell_h)
                slot2 = grid[y0:y1, x0:x1]
                if slot2.size == 0:
                    row_parts.append('EMPTY ')
                    continue
                rarity2, _ = _classify_rarity(slot2)
                icon2 = _icon_crop_from_slot(slot2)
                if icon2.size == 0:
                    row_parts.append('TINY  ')
                    continue
                item_id2, score2 = _classify_icon_embed(
                    icon2, net, embeddings, labels, rarities, rarity2,
                )
                hit = item_id2 is not None and score2 >= EMBED_SCORE_THRESHOLD
                hits += hit
                row_parts.append(
                    f'{rarity2:<3}{"✓" if hit else "✗"}(s={score2:.2f})'
                )
            print('  row', row, ':', '  '.join(row_parts))
        print(f'\n  Total matched: {hits}/20')

    print(f'\nDone. Open {out_dir}/ to inspect the grid crop images.')


if __name__ == '__main__':
    main()
