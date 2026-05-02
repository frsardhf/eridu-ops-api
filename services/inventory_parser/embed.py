"""Generate CLIP embeddings for icon matching.

Run after icons have been downloaded (app.py warmup downloads them automatically):

    python embed.py items
    python embed.py equipment

Outputs written to cache/:
    icon_embeddings_{type}.npy             — L2-normalised embeddings, shape (N, 512)
    icon_embeddings_{type}_labels.json     — {str(row_index): item_id}
    icon_embeddings_{type}_rarities.json   — {str(row_index): rarity}
    icon_embeddings_{type}_shapes.json     — {str(row_index): {circularity, aspect_ratio}}

No checkpoint (.pt) required — uses the pretrained CLIP ViT-B/32 model which
encodes visual semantics (shape, colour, text) independently of the background.
"""

import json
import os
import sys

import cv2
import numpy as np
import torch
from transformers import CLIPModel

BASE_DIR  = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
ICONS_DIR = os.path.join(CACHE_DIR, 'icons')

INDEX_PATHS = {
    'items':     os.path.join(CACHE_DIR, 'icon_index_items.json'),
    'equipment': os.path.join(CACHE_DIR, 'icon_index_equipment.json'),
}

CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
CLIP_INPUT_SIZE = 224

# CLIP preprocessing constants (RGB order)
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# Neutral grey background — composited under transparent icon pixels.
NEUTRAL_BG_BGR = (128, 128, 128)
# Dark background that approximates the game's parallelogram slot background.
DARK_BG_BGR = (45, 45, 45)

# Rarity-specific backgrounds — sampled from actual in-game icon crop regions.
# Using the item's own rarity background reduces the domain gap between reference
# icons and in-game crops, which is critical for ViT-L/14 which is more sensitive
# to background colour than ViT-B/32.
RARITY_BG_BGR = {
    'N':   (210, 230, 235),    # beige / cream
    'R':   (235, 210, 185),    # warm beige / light orange
    'SR':  (130, 200, 248),    # blue
    'SSR': (240, 180, 210),    # pink / purple
}


def _composite_on_bg(raw: np.ndarray, bg_bgr: tuple) -> np.ndarray:
    """Composite a BGRA icon onto bg_bgr and return BGR at CLIP_INPUT_SIZE."""
    if raw is None:
        return np.full((CLIP_INPUT_SIZE, CLIP_INPUT_SIZE, 3), bg_bgr, dtype=np.uint8)

    if raw.ndim == 3 and raw.shape[2] == 4:
        bgr   = raw[:, :, :3].astype(np.float32)
        alpha = raw[:, :, 3:].astype(np.float32) / 255.0
        bg    = np.full_like(bgr, bg_bgr, dtype=np.float32)
        bgr   = (bgr * alpha + bg * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    elif raw.ndim == 3 and raw.shape[2] == 3:
        bgr = raw
    else:
        bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    return cv2.resize(bgr, (CLIP_INPUT_SIZE, CLIP_INPUT_SIZE))


def _composite_on_grey(raw: np.ndarray) -> np.ndarray:
    return _composite_on_bg(raw, NEUTRAL_BG_BGR)


def _brightness_adjust(bgr: np.ndarray, factor: float) -> np.ndarray:
    """Scale pixel values by factor, clipping to [0, 255]."""
    return np.clip(bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _bgr_to_clip_tensor(bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR uint8 HWC image to a CLIP-normalised NCHW tensor."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _CLIP_MEAN) / _CLIP_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)  # 1×3×H×W


def _compute_icon_shape(raw: np.ndarray) -> dict:
    """Return {'circularity': float, 'aspect_ratio': float} from the icon's alpha mask.

    circularity = 4π·area / perimeter² → 1.0 for a perfect circle, ~0 for a line.
    aspect_ratio = bounding_box_w / bounding_box_h.
    Falls back to {1.0, 1.0} when the icon has no usable mask.
    """
    if raw is None:
        return {'circularity': 1.0, 'aspect_ratio': 1.0}

    if raw.ndim == 3 and raw.shape[2] == 4:
        alpha = raw[:, :, 3]
        _, binary = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {'circularity': 1.0, 'aspect_ratio': 1.0}

    c = max(contours, key=cv2.contourArea)
    area      = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    _, _, w, h = cv2.boundingRect(c)

    circularity  = float(4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 1.0
    aspect_ratio = float(w / h) if h > 0 else 1.0
    return {'circularity': round(circularity, 4), 'aspect_ratio': round(aspect_ratio, 4)}


def embed(inv_type: str) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

    emb_path      = os.path.join(CACHE_DIR, f'icon_embeddings_{inv_type}.npy')
    labels_path   = os.path.join(CACHE_DIR, f'icon_embeddings_{inv_type}_labels.json')
    rarities_path = os.path.join(CACHE_DIR, f'icon_embeddings_{inv_type}_rarities.json')
    shapes_path   = os.path.join(CACHE_DIR, f'icon_embeddings_{inv_type}_shapes.json')

    print(f'\n=== Embedding pipeline for: {inv_type} ===')
    print(f'Loading CLIP model: {CLIP_MODEL_NAME}')

    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    model.eval()

    # --- Load icon index ---
    index_path = INDEX_PATHS.get(inv_type)
    if not index_path or not os.path.exists(index_path):
        print(f'ERROR: icon index not found: {index_path}')
        print(f'       Start app.py once so warmup downloads the icons, then Ctrl-C.')
        sys.exit(1)

    with open(index_path, encoding='utf-8') as fh:
        raw = json.load(fh)
    index = raw.get('items', {})   # {item_id: {filename, rarity}}

    icon_dir = os.path.join(ICONS_DIR, inv_type)
    ordered_ids:    list[str]         = []
    ordered_rar:    list[str]         = []   # rarity per row index
    ordered_shapes: list[dict]        = []   # shape descriptors per row index
    embeddings:     list[np.ndarray]  = []

    # Multi-view augmentation: embed each icon from 5 views and average.
    # Views: (1) grey bg, (2) dark bg — approximates the game slot background,
    #        (3) horizontal flip on grey, (4) brightness −15%, (5) brightness +15%.
    # Averaging before L2-norm reduces the domain gap between reference icons
    # (composited on grey) and in-game crops (dark parallelogram background), and
    # adds robustness to appearance variations without changing the embedding dimension.
    print(f'Computing embeddings for {len(index)} icons (5-view average)...')
    with torch.no_grad():
        for item_id, info in index.items():
            fpath = os.path.join(icon_dir, info.get('filename', ''))
            raw_img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
            bgr_grey = _composite_on_grey(raw_img)
            bgr_dark = _composite_on_bg(raw_img, DARK_BG_BGR)
            views = [
                bgr_grey,
                bgr_dark,
                cv2.flip(bgr_grey, 1),
                _brightness_adjust(bgr_grey, 0.85),
                _brightness_adjust(bgr_grey, 1.15),
            ]
            vecs = []
            for view in views:
                tensor = _bgr_to_clip_tensor(view)
                vision_out = model.vision_model(pixel_values=tensor)
                pooled = vision_out.pooler_output
                feat   = model.visual_projection(pooled)
                vecs.append(feat.squeeze().numpy().astype(np.float32))
            vec = np.mean(vecs, axis=0)  # average before L2-norm
            ordered_ids.append(item_id)
            ordered_rar.append(info.get('rarity', 'Unknown'))
            ordered_shapes.append(_compute_icon_shape(raw_img))
            embeddings.append(vec)

    # Stack and L2-normalise
    emb_matrix = np.stack(embeddings, axis=0).astype(np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-10
    emb_matrix = emb_matrix / norms

    np.save(emb_path, emb_matrix)
    print(f'Embeddings saved → {emb_path}  shape={emb_matrix.shape}')

    labels_map = {str(i): item_id for i, item_id in enumerate(ordered_ids)}
    with open(labels_path, 'w', encoding='utf-8') as fh:
        json.dump(labels_map, fh)
    print(f'Labels saved      → {labels_path}  ({len(labels_map)} classes)')

    rarities_map = {str(i): rar for i, rar in enumerate(ordered_rar)}
    with open(rarities_path, 'w', encoding='utf-8') as fh:
        json.dump(rarities_map, fh)
    print(f'Rarities saved    → {rarities_path}')

    shapes_map = {str(i): s for i, s in enumerate(ordered_shapes)}
    with open(shapes_path, 'w', encoding='utf-8') as fh:
        json.dump(shapes_map, fh)
    print(f'Shapes saved      → {shapes_path}')


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('items', 'equipment'):
        print('Usage: python embed.py items|equipment')
        sys.exit(1)

    embed(sys.argv[1])
