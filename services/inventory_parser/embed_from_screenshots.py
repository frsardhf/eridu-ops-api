"""Enhance CLIP embeddings by blending in-game screenshot crops with sprite embeddings.

Approach:
1. Run the existing pipeline on each screenshot in assets/ to get item predictions.
2. Trust predictions that are monotonically consistent (ascending ID order) — these
   are high-confidence ground truth labels.
3. Crop the icon region from each trusted cell and compute its CLIP embedding.
4. For each item, average all in-game embeddings, then blend 50/50 with the
   original sprite embedding.  Items without any in-game sample keep the
   sprite-only embedding unchanged.

Usage:
    python embed_from_screenshots.py items

Prerequisites:
    - Run `python embed.py items` first (generates sprite-based embeddings)
    - Place screenshots as assets/Screenshot0.png, Screenshot1.png, …
"""

import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from transformers import CLIPModel

BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
CLIP_INPUT_SIZE = 224

_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# Must match pipeline.py values
ICON_CROP_TOP   = 0.04
ICON_CROP_BOT   = 0.78
ICON_CROP_LEFT  = 0.28
ICON_CROP_RIGHT = 0.96

# Blend weight: 0.0 = sprite only, 1.0 = in-game only
INGAME_BLEND = 0.5


def _bgr_to_clip_tensor(bgr: np.ndarray) -> torch.Tensor:
    bgr = cv2.resize(bgr, (CLIP_INPUT_SIZE, CLIP_INPUT_SIZE))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _CLIP_MEAN) / _CLIP_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


def _icon_crop(slot: np.ndarray) -> np.ndarray:
    """Extract the icon region from a grid cell, matching pipeline.py logic."""
    sh, sw = slot.shape[:2]
    side = min(sh, sw)
    if sh > sw:
        trim = (sh - sw) // 2
        slot = slot[trim:trim + side, :]
    elif sw > sh:
        trim = (sw - sh) // 2
        slot = slot[:, trim:trim + side]
    return slot[int(side * ICON_CROP_TOP):int(side * ICON_CROP_BOT),
                int(side * ICON_CROP_LEFT):int(side * ICON_CROP_RIGHT)]


def _extract_grid_cells(image_path: str):
    """Extract 20 icon crops from a screenshot. Returns list of BGR arrays."""
    from pipeline import _compute_grid_bounds

    img = cv2.imread(image_path)
    if img is None:
        return []
    h, w = img.shape[:2]
    right_half = img[:, w // 2:]
    left, top, right, bottom = _compute_grid_bounds(right_half)
    grid = right_half[top:bottom, left:right]
    gh, gw = grid.shape[:2]
    cell_w, cell_h = gw / 5, gh / 4

    crops = []
    for row in range(4):
        for col in range(5):
            x0 = int(col * cell_w)
            y0 = int(row * cell_h)
            x1 = int((col + 1) * cell_w)
            y1 = int((row + 1) * cell_h)
            slot = grid[y0:y1, x0:x1]
            ic = _icon_crop(slot)
            crops.append(ic if ic.size > 0 else None)
    return crops


def _get_trusted_labels(image_path: str, inv_type: str):
    """Run the pipeline and return labels for cells that are monotonically consistent."""
    from pipeline import parse_inventory

    with open(image_path, 'rb') as f:
        data = f.read()
    results = parse_inventory(data, inv_type)
    if not results:
        return [None] * 20

    # Sort by grid position
    ordered = sorted(results, key=lambda r: r['row'] * 5 + r['col'])
    ids = [int(r['itemId']) for r in ordered]

    # Mark each cell as trusted if it's consistent with ascending sort order.
    # A cell is trusted if:
    #   - It's greater than the previous trusted cell (or first cell)
    #   - AND less than the next trusted cell (or last cell)
    # First pass: find longest ascending subsequence using greedy forward scan.
    n = len(ids)
    trusted = [False] * n

    # Simple approach: mark cells that don't violate monotonicity with neighbours
    for i in range(n):
        prev_ok = (i == 0) or (ids[i] > ids[i - 1])
        next_ok = (i == n - 1) or (ids[i] < ids[i + 1])
        trusted[i] = prev_ok and next_ok

    labels = []
    for i in range(n):
        if trusted[i]:
            labels.append(str(ids[i]))
        else:
            labels.append(None)

    # Pad to 20 if fewer results
    while len(labels) < 20:
        labels.append(None)

    return labels


def enhance(inv_type: str) -> None:
    sprite_emb_path = os.path.join(CACHE_DIR, f'icon_embeddings_{inv_type}.npy')
    labels_path = os.path.join(CACHE_DIR, f'icon_embeddings_{inv_type}_labels.json')

    if not os.path.exists(sprite_emb_path):
        print(f'ERROR: Run `python embed.py {inv_type}` first.')
        sys.exit(1)

    sprite_emb = np.load(sprite_emb_path)
    with open(labels_path) as f:
        labels_map = json.load(f)  # {str(row_idx): item_id}

    # Reverse: item_id -> row_idx
    id_to_idx = {v: int(k) for k, v in labels_map.items()}

    print(f'\n=== In-game embedding enhancement for: {inv_type} ===')
    print(f'Loading CLIP model: {CLIP_MODEL_NAME}')
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    model.eval()

    # Collect in-game embeddings per item_id
    ingame_vecs: dict[str, list[np.ndarray]] = {}

    screenshots = sorted(glob.glob(os.path.join(ASSETS_DIR, 'Screenshot*.png')))
    print(f'Found {len(screenshots)} screenshots')

    for spath in screenshots:
        fname = os.path.basename(spath)
        print(f'  Processing {fname}...')

        trusted_labels = _get_trusted_labels(spath, inv_type)
        crops = _extract_grid_cells(spath)

        trusted_count = sum(1 for l in trusted_labels if l is not None)
        print(f'    Trusted cells: {trusted_count}/20')

        with torch.no_grad():
            for i, (crop, label) in enumerate(zip(crops, trusted_labels)):
                if label is None or crop is None:
                    continue
                if label not in id_to_idx:
                    continue

                tensor = _bgr_to_clip_tensor(crop)
                vision_out = model.vision_model(pixel_values=tensor)
                pooled = vision_out.pooler_output
                feat = model.visual_projection(pooled)
                vec = feat.squeeze().numpy().astype(np.float32)

                if label not in ingame_vecs:
                    ingame_vecs[label] = []
                ingame_vecs[label].append(vec)

    # Blend in-game embeddings with sprite embeddings
    enhanced = sprite_emb.copy()
    enhanced_count = 0

    for item_id, vecs in ingame_vecs.items():
        idx = id_to_idx.get(item_id)
        if idx is None:
            continue

        ingame_avg = np.mean(vecs, axis=0)
        ingame_avg = ingame_avg / (np.linalg.norm(ingame_avg) + 1e-10)

        sprite_vec = sprite_emb[idx]

        # Blend: weighted average of sprite and in-game embeddings
        blended = (1 - INGAME_BLEND) * sprite_vec + INGAME_BLEND * ingame_avg
        blended = blended / (np.linalg.norm(blended) + 1e-10)

        enhanced[idx] = blended
        enhanced_count += 1

    # Save enhanced embeddings (overwrite)
    np.save(sprite_emb_path, enhanced)
    print(f'\nEnhanced {enhanced_count}/{len(labels_map)} item embeddings')
    print(f'Items without in-game samples (sprite-only): {len(labels_map) - enhanced_count}')
    print(f'Saved → {sprite_emb_path}')


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('items', 'equipment'):
        print('Usage: python embed_from_screenshots.py items|equipment')
        sys.exit(1)

    enhance(sys.argv[1])
