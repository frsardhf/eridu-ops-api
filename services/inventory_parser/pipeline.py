import json
import os
import re
from typing import Dict, List, Optional, Tuple
from urllib import request, error

import cv2
import numpy as np

BASE_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
ICON_CACHE_DIR = os.path.join(CACHE_DIR, 'icons')

ICON_INDEX_ITEMS = os.path.join(CACHE_DIR, 'icon_index_items.json')
ICON_INDEX_EQUIPMENT = os.path.join(CACHE_DIR, 'icon_index_equipment.json')

SCHALEDB_BASE_URL = 'https://schaledb.com/data/en'

GRID_FALLBACK_LEFT_RATIO = 0.06
GRID_FALLBACK_RIGHT_RATIO = 0.95
# Top: just below the "List / filter" header bar (~22% of right-half height)
# Bottom: just above the "Item cannot be used here" footer bar (~78%)
# Providing list_header.png + item_cannot_use.png templates overrides these.
GRID_FALLBACK_TOP_RATIO = 0.22
GRID_FALLBACK_BOTTOM_RATIO = 0.78
GRID_ANCHOR_MARGIN = 5

# CLIP model used for icon embedding.
CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
CLIP_INPUT_SIZE = 224
# CLIP preprocessing constants (RGB order, matches openai/clip-vit-base-patch32).
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
# Minimum cosine similarity to accept an embedding match.
# CLIP similarities: correct match typically 0.75–0.95, wrong match < 0.70.
EMBED_SCORE_THRESHOLD = 0.75
# Relaxed threshold for range-restricted re-classification.  When contextual
# neighbours strongly indicate the correct item range, a lower threshold is
# acceptable because the range restriction itself provides prior information.
_RANGE_FIX_THRESHOLD = 0.70
# Even more relaxed threshold for recovering dropped favor cells.  These
# cells are completely missing from results and the page context (sort
# order, surrounding items) strongly indicates they should be favors.
_DROP_RECOVER_THRESHOLD = 0.60

# Icon crop fractions — applied to the square-normalised cell.
# Tune these four values if the icon is misaligned in the debug images.
ICON_CROP_TOP   = 0.04   # skip top (rarity border + parallelogram dead zone)
ICON_CROP_BOT   = 0.78   # bottom of icon (above quantity text)
ICON_CROP_LEFT  = 0.28   # skip left (parallelogram triangle dead zone)
ICON_CROP_RIGHT = 0.96   # right edge of icon

# Quantity crop fractions — relative to cell height/width.
QTY_CROP_TOP_FRAC   = 0.74   # start of quantity region (fraction of cell height from cell top)
QTY_CROP_OVERFLOW   = 0.06   # extend below cell bottom (fraction of cell height)
QTY_CROP_LEFT_FRAC  = 0.40   # left edge of quantity region (fraction of cell width)
QTY_CROP_RIGHT_FRAC = 0.98   # right edge of quantity region (fraction of cell width)

# Confidence blend weights (icon CLIP score vs digit OCR score).
_CONF_ICON_WEIGHT  = 0.7
_CONF_DIGIT_WEIGHT = 0.3

# Mirrors the MATERIAL / EQUIPMENT filter constants in src/types/resource.ts.
# applyFilters() uses OR logic — an item is included if it matches ANY criterion.
ITEM_INCLUDE_FILTER = {
    'category':    {'CharacterExpGrowth', 'Favor'},
    'subcategory': {'Artifact', 'CDItem', 'BookItem'},
    'id':          {23, 2000, 2001, 2002, 9999},
}
EQUIPMENT_INCLUDE_FILTER = {
    'category':    {'Exp'},
    'recipecost':  {1500, 10000, 25000, 50000, 75000, 100000, 125000, 150000, 175000},
}


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

# Shared CLIPModel instance — loaded once, reused for both items and equipment.
_CLIP_MODEL = None
# {inventory_type: (model, embeddings_matrix, labels)}
# embeddings_matrix: L2-normalised float32 array, shape (N, 512)
# labels: {str(row_index): item_id}
_EMBED_CACHE: Dict[str, Tuple[object, np.ndarray, Dict[str, str]]] = {}
# {inventory_type: {str(row_index): rarity_str}}  — loaded alongside embeddings
_RARITY_CACHE: Dict[str, Dict[str, str]] = {}
# {inventory_type: {str(row_index): {circularity, aspect_ratio}}}
_SHAPE_CACHE: Dict[str, Dict[str, dict]] = {}
# EasyOCR reader — loaded once on first quantity read.
_EASYOCR_READER = None


def get_item_icon_url(icon: str, item_type: str, tier: Optional[int] = None) -> str:
    is_equipment = item_type == 'equipment'
    icon_name = icon
    if is_equipment and tier is not None and tier != 0:
        icon_name = f'{icon}_piece'
    return f'https://schaledb.com/images/{"equipment" if is_equipment else "item"}/icon/{icon_name}.webp'


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; eridu-ops-inventory-parser/1.0)',
    'Accept': 'application/json, image/webp, */*',
}


def _open_url(url: str, timeout: int = 20):
    req = request.Request(url, headers=_HEADERS)
    return request.urlopen(req, timeout=timeout)


def _fetch_json(url: str) -> Dict:
    try:
        with _open_url(url) as response:
            payload = response.read()
            return json.loads(payload.decode('utf-8'))
    except (error.URLError, json.JSONDecodeError) as exc:
        print(f'[inventory_parser] Failed to fetch {url}: {exc}')
        return {}


def _download_to_path(url: str, dest_path: str) -> bool:
    try:
        with _open_url(url) as response:
            data = response.read()
        with open(dest_path, 'wb') as handle:
            handle.write(data)
        return True
    except error.URLError as exc:
        print(f'[inventory_parser] Failed to download {url}: {exc}')
        return False


def _get_or_load_clip_model() -> object:
    """Load CLIPModel once and cache globally (shared across item types)."""
    global _CLIP_MODEL
    if _CLIP_MODEL is None:
        import torch
        from transformers import CLIPModel as _CLIPModel
        print(f'[inventory_parser] Loading CLIP model: {CLIP_MODEL_NAME}')
        _CLIP_MODEL = _CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        _CLIP_MODEL.eval()
    return _CLIP_MODEL


def _load_embeddings(
    inventory_type: str,
) -> Tuple[Optional[object], Optional[np.ndarray], Optional[Dict[str, str]]]:
    """Load the CLIP model, precomputed embedding matrix, and label map.

    Returns (model, embeddings, labels) or (None, None, None) if embed.py
    hasn't been run yet.
    """
    if inventory_type in _EMBED_CACHE:
        return _EMBED_CACHE[inventory_type]

    emb_path    = os.path.join(CACHE_DIR, f'icon_embeddings_{inventory_type}.npy')
    labels_path = os.path.join(CACHE_DIR, f'icon_embeddings_{inventory_type}_labels.json')

    if not all(os.path.exists(p) for p in (emb_path, labels_path)):
        return None, None, None

    model      = _get_or_load_clip_model()
    embeddings = np.load(emb_path)                      # (N, 512), L2-normalised
    with open(labels_path, 'r', encoding='utf-8') as fh:
        labels = json.load(fh)

    _EMBED_CACHE[inventory_type] = (model, embeddings, labels)
    print(f'[inventory_parser] Loaded embeddings for {inventory_type} '
          f'({embeddings.shape[0]} classes, dim={embeddings.shape[1]})')
    return model, embeddings, labels


def _clip_preprocess(query_bgr: np.ndarray) -> np.ndarray:
    """Resize and CLIP-normalise a BGR crop → float32 array (1, 3, 224, 224)."""
    resized = cv2.resize(query_bgr, (CLIP_INPUT_SIZE, CLIP_INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _CLIP_MEAN) / _CLIP_STD
    return rgb.transpose(2, 0, 1)[np.newaxis, :]        # (1, 3, 224, 224)


def _get_embed_scores(
    query_bgr: np.ndarray,
    model: object,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Return cosine similarities (shape: N,) for all stored icons (debug helper)."""
    import torch
    arr    = _clip_preprocess(query_bgr)                # (1, 3, 224, 224)
    tensor = torch.from_numpy(arr)
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=tensor)
        pooled     = vision_out.pooler_output
        feat       = model.visual_projection(pooled)
    feat = feat.numpy().ravel().astype(np.float32)
    feat = feat / (np.linalg.norm(feat) + 1e-10)
    return (embeddings @ feat).astype(np.float32)       # cosine similarity per icon


def _load_rarities(inventory_type: str) -> Dict[str, str]:
    """Load {str(row_index): rarity_str} from cache, or empty dict if missing."""
    if inventory_type in _RARITY_CACHE:
        return _RARITY_CACHE[inventory_type]
    path = os.path.join(CACHE_DIR, f'icon_embeddings_{inventory_type}_rarities.json')
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as fh:
        rarities = json.load(fh)
    _RARITY_CACHE[inventory_type] = rarities
    return rarities


def _load_shapes(inventory_type: str) -> Dict[str, dict]:
    """Load {str(row_index): {circularity, aspect_ratio}} from cache, or {} if missing."""
    if inventory_type in _SHAPE_CACHE:
        return _SHAPE_CACHE[inventory_type]
    path = os.path.join(CACHE_DIR, f'icon_embeddings_{inventory_type}_shapes.json')
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as fh:
        shapes = json.load(fh)
    _SHAPE_CACHE[inventory_type] = shapes
    return shapes


# Shape-compatibility score bonus added to CLIP cosine similarity.
# Shape-matched candidates (circularity within _SHAPE_CIRC_TOL) get a small boost
# so they rank above geometrically incompatible ones when CLIP scores are close.
# Kept small (0.04) so it only breaks near-ties — never overrides a clearly better match.
_SHAPE_BONUS   = 0.015
_SHAPE_CIRC_TOL = 0.30   # max |db_circularity - query_circularity| to earn the bonus


def _compute_query_shape(icon_bgr: np.ndarray) -> dict:
    """Estimate the shape descriptor of an icon crop extracted from a screenshot.

    Uses Otsu binarisation on the grayscale crop to isolate the icon from the
    dark background, then measures contour circularity and bounding-box aspect ratio.
    """
    gray = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {'circularity': 1.0, 'aspect_ratio': 1.0}
    c = max(contours, key=cv2.contourArea)
    area      = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    _, _, w, h = cv2.boundingRect(c)
    circularity  = float(4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 1.0
    aspect_ratio = float(w / h) if h > 0 else 1.0
    return {'circularity': circularity, 'aspect_ratio': aspect_ratio}


def _classify_icon_embed(
    query_bgr: np.ndarray,
    net: object,
    embeddings: np.ndarray,
    labels: Dict[str, str],
    rarities: Optional[Dict[str, str]] = None,
    hint_rarity: Optional[str] = None,
    shapes: Optional[Dict[str, dict]] = None,
    query_shape: Optional[dict] = None,
) -> Tuple[Optional[str], float]:
    """Return (item_id, cosine_similarity) for the closest icon in the database.

    Pipeline:
    1. Shape-aware scoring — add _SHAPE_BONUS to candidates whose circularity is
       within _SHAPE_CIRC_TOL of the query.  Soft approach: no candidate excluded.
    2. Rarity reranking   — within the adjusted scores, prefer a rarity-matched
       result when it falls within 0.07 of the global best (resolves exp-book ties).
    """
    scores = _get_embed_scores(query_bgr, net, embeddings)   # cosine, range [-1, 1]

    # ── 1. Shape-aware scoring ──────────────────────────────────────────────
    # Add a small bonus to shape-compatible candidates so they rank above
    # geometrically incompatible ones when CLIP scores are close.
    # Soft approach: no candidate is excluded — the threshold check in the
    # caller still uses the real (un-boosted) CLIP similarity.
    adjusted = scores.copy()
    if shapes and query_shape:
        q_circ = query_shape['circularity']
        for i in range(len(scores)):
            db_circ = shapes.get(str(i), {}).get('circularity', 1.0)
            if abs(db_circ - q_circ) <= _SHAPE_CIRC_TOL:
                adjusted[i] += _SHAPE_BONUS

    global_best_idx   = int(adjusted.argmax())
    global_best_score = float(scores[global_best_idx])   # real CLIP score for threshold

    # ── 2. Rarity reranking ─────────────────────────────────────────────────
    if rarities and hint_rarity and hint_rarity != 'Unknown':
        rarity_indices = [i for i in range(len(adjusted))
                          if rarities.get(str(i)) == hint_rarity]
        if rarity_indices:
            rar_best_idx   = max(rarity_indices, key=lambda i: adjusted[i])
            rar_best_score = float(scores[rar_best_idx])
            # Accept rarity-filtered result if within 0.07 of the global best
            # (using real scores for the margin comparison so the shape bonus
            #  does not inflate the rarity-match acceptance threshold).
            if rar_best_score >= global_best_score - 0.07:
                return labels.get(str(rar_best_idx)), rar_best_score

    return labels.get(str(global_best_idx)), global_best_score


def warm_icon_db() -> None:
    for inv_type in ('items', 'equipment'):
        _load_embeddings(inv_type)
        _load_rarities(inv_type)
        _load_shapes(inv_type)
    _get_or_load_easyocr()


def _get_or_load_easyocr() -> object:
    """Load EasyOCR reader once and cache globally."""
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        print('[inventory_parser] Loading EasyOCR reader (en)')
        _EASYOCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    return _EASYOCR_READER


# Upscale factor applied to quantity crop before OCR.
# 3× is a sweet spot: fast enough for 20-cell grids, accurate on the game font.
# Avoid 4× — it causes EasyOCR to merge narrow digits like '1' into neighbours.
_QTY_OCR_SCALE = 3


def _read_quantity(roi_bgr: np.ndarray) -> Tuple[int, float]:
    """Read the quantity value from an item slot's bottom-right crop using EasyOCR.

    Expected text format: ×NNN[K|M] where:
      - × is the multiplication prefix (stripped)
      - NNN is 1–5 decimal digits
      - K multiplies by 1 000; M multiplies by 1 000 000

    Returns (quantity, confidence) where confidence ∈ [0, 1].
    """
    reader = _get_or_load_easyocr()
    h, w = roi_bgr.shape[:2]
    if h == 0 or w == 0:
        return 0, 0.0

    scaled = cv2.resize(
        roi_bgr,
        (w * _QTY_OCR_SCALE, h * _QTY_OCR_SCALE),
        interpolation=cv2.INTER_CUBIC,
    )

    # Sharpen: enhance digit edge contrast to reduce 1 ↔ 7 confusion.
    # Unsharp mask: sharpened = original * 1.8 - blurred * 0.8
    _gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    _blur = cv2.GaussianBlur(_gray, (0, 0), 1.5)
    _sharp = cv2.addWeighted(_gray, 1.8, _blur, -0.8, 0)
    scaled = cv2.cvtColor(_sharp, cv2.COLOR_GRAY2BGR)

    # Restrict character set to digits + known suffix/prefix characters.
    results = reader.readtext(
        scaled,
        allowlist='0123456789kKmMxX×',
        detail=1,
    )
    if not results:
        return 0, 0.0

    # Pick the best quantity detection.  Pure max-confidence is unreliable:
    # SSR rarity frames and complex icon backgrounds produce spurious reads that
    # can outscore the real quantity text.
    #
    # Priority (highest to lowest):
    #   1. Vertical position — prefer detections at the BOTTOM of the crop.
    #      When the grid is scroll-misaligned, quantity text from the row above
    #      bleeds into the top of the current cell.  The actual quantity is always
    #      at the bottom.  We discretise into top-half (0) vs bottom-half (1).
    #   2. Starts with × / X  — the game ALWAYS prefixes quantities with ×.
    #   3. Digit count after stripping ×/K/M.
    #   4. EasyOCR confidence as final tiebreaker.
    img_h = scaled.shape[0]

    def _qty_priority(result):
        # result[0] = bounding box polygon, result[1] = text, result[2] = conf
        bbox = result[0]
        # Centre-Y of the bounding box (average of all y-coordinates)
        cy = sum(pt[1] for pt in bbox) / len(bbox)
        is_bottom = int(cy > img_h * 0.45)

        text = result[1].upper()
        has_x = int(bool(re.match(r'^[X×]', text)))
        raw = re.sub(r'^[X×]+', '', text)
        raw = re.sub(r'[KM]$', '', raw)
        n_digits = sum(c.isdigit() for c in raw)
        return (is_bottom, has_x, n_digits, float(result[2]))

    best = max(results, key=_qty_priority)
    _bbox_x_min = min(pt[0] for pt in best[0])
    text: str = best[1].upper()
    conf: float = float(best[2])

    # Strip leading × / X prefix (the in-game quantity multiplier symbol).
    # Also handles icon-bleed: a curved icon edge in the top-left of the crop
    # is sometimes mis-read as a leading digit (e.g. "1x179", "3X575", "4X748").
    # If × appears within the first 3 characters, drop everything up to and
    # including it; otherwise fall back to stripping leading X/× as before.
    m = re.match(r'^.{0,2}[X×]', text)
    if m:
        text = text[m.end():]
        _x_stripped = True
    else:
        _stripped = re.sub(r'^[X×]+', '', text)
        _x_stripped = _stripped != text
        text = _stripped
    # Additional fix: × is sometimes misread as '7' at the left edge of the crop
    # (the diagonal strokes of × resemble the numeral).  Detectable when: no X
    # character appeared in the text, the text starts with '7', the bounding box
    # starts very close to x=0 (where × always sits), and removing the spurious
    # leading '7' still leaves ≥ 3 digits.  The 30 px threshold applies to the
    # 3× up-scaled crop (≈ 10 px in the original ROI).
    if not _x_stripped and text.startswith('7') and len(text) >= 4 and _bbox_x_min < 30:
        text = text[1:]

    # Parse optional K/M suffix.
    multiplier = 1
    if text.endswith('M'):
        _before_m = text[:-1]
        if len(_before_m) == 1 and _before_m.isdigit():
            # Single digit before 'M': EasyOCR almost certainly misread a trailing
            # '7' as 'M' (the two glyphs share a wide top stroke in the game font).
            # e.g. raw "X9M" → stripped "9M" → actual ×97.
            # Real ×NM quantities (millions) would show 2+ digits before the suffix.
            text = _before_m + '7'
        else:
            multiplier = 1_000_000
            text = text[:-1]
    elif text.endswith('K'):
        multiplier = 1_000
        text = text[:-1]

    digits = re.sub(r'[^0-9]', '', text)
    if not digits:
        return 0, conf

    try:
        quantity = int(digits) * multiplier
    except ValueError:
        quantity = 0

    return quantity, conf


def _match_template(image: np.ndarray, template_path: str, threshold: float = 0.7) -> Optional[Tuple[int, int, int, int]]:
    if not os.path.exists(template_path):
        return None
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < threshold:
        return None

    h, w = template.shape
    return max_loc[0], max_loc[1], w, h


def _find_panel_bounds(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w * h < 0.1 * image.shape[0] * image.shape[1]:
        return None
    return x, y, w, h


def _compute_grid_bounds(right_half: np.ndarray) -> Tuple[int, int, int, int]:
    height, width = right_half.shape[:2]

    list_header_path = os.path.join(ASSETS_DIR, 'list_header.png')
    item_cannot_path = os.path.join(ASSETS_DIR, 'item_cannot_use.png')

    list_match = _match_template(right_half, list_header_path)
    cannot_match = _match_template(right_half, item_cannot_path)

    top = int(height * GRID_FALLBACK_TOP_RATIO)
    bottom = int(height * GRID_FALLBACK_BOTTOM_RATIO)

    if list_match:
        _, y, _, h = list_match
        top = max(0, y + h + GRID_ANCHOR_MARGIN)
    if cannot_match:
        _, y, _, _ = cannot_match
        bottom = max(top + 10, y - GRID_ANCHOR_MARGIN)

    # Always derive left/right from fallback ratios — panel detection finds the
    # entire bright background (full right-half width) rather than the grid sub-panel,
    # so using it for horizontal bounds misaligns the cell grid.
    left = int(width * GRID_FALLBACK_LEFT_RATIO)
    right = int(width * GRID_FALLBACK_RIGHT_RATIO)

    panel_bounds = _find_panel_bounds(right_half)
    if panel_bounds:
        _, y, _, h = panel_bounds
        # Only accept panel bounds if the detected height is at least 70% of the
        # current grid height.  Screenshot 2's bright SSR (silver/diamond) rarity
        # frames push many pixels above the 200-threshold, causing _find_panel_bounds
        # to return a short region that is just the top bright strip.  Accepting that
        # would clip bottom upward and shift ALL cell crops, breaking both icon and
        # quantity reads.  The 70% guard rejects such spurious small panels.
        current_grid_h = bottom - top
        if h >= current_grid_h * 0.70:
            top = max(top, y)
            bottom = min(bottom, y + h)

    return left, top, right, bottom


def _rarity_from_hue(median_h: float) -> Tuple[str, bool]:
    if 90 <= median_h <= 120:
        return 'R', True
    if 15 <= median_h <= 35:
        return 'SR', True
    if 128 <= median_h <= 165:
        return 'SSR', True
    return 'Unknown', False


# Minimum fraction of a ring row that must be coloured for it to count as a
# rarity frame rather than a UI selection-glow.  Solid rarity borders span
# 80–90 % of the row; the selection glow only reaches ~30 %.
_RARITY_DENSITY_THRESHOLD = 0.50


def _classify_rarity(slot_bgr: np.ndarray) -> Tuple[str, bool]:
    h, w = slot_bgr.shape[:2]
    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)

    # Try progressively wider top/bottom strips to handle screenshots taken
    # at different scroll positions.  Well-aligned screenshots find the rarity
    # border in the narrow 10 % strip; misaligned ones need up to 25 %.
    # Left/right strips are intentionally skipped — they pick up neighbouring
    # cells' icon colours and cause false rarity classifications.
    # Icon artwork can bleed into narrow strips and produce a single row with
    # moderate saturation density (~0.67), mimicking a rarity border.  Real
    # rarity borders are multi-row and grow stronger at wider rings.
    # Guard 1: require at least 2 qualifying rows (icon bleed = 1 row).
    # Guard 2: at narrow rings (< 20%), also require peak density >= 0.75
    #          to skip low-density bleed from neighbouring cells' borders.
    _MIN_DENSE_ROWS = 2
    _DENSITY_CONFIDENT = 0.75
    _NARROW_RING_LIMIT = 0.20

    for ring_pct in (0.10, 0.15, 0.20, 0.25):
        ring = int(min(h, w) * ring_pct)
        if ring <= 0:
            continue
        for strip in (hsv[:ring, :, :], hsv[-ring:, :, :]):
            sat = strip[:, :, 1]
            hue = strip[:, :, 0]
            row_density = (sat > 40).sum(axis=1) / w
            peak_density = float(row_density.max())
            if peak_density >= _RARITY_DENSITY_THRESHOLD:
                dense_mask = row_density >= _RARITY_DENSITY_THRESHOLD
                if int(dense_mask.sum()) < _MIN_DENSE_ROWS:
                    continue   # likely icon bleed — single saturated row
                if ring_pct < _NARROW_RING_LIMIT and peak_density < _DENSITY_CONFIDENT:
                    continue   # low density at narrow ring — likely neighbour bleed
                dense_hues = hue[dense_mask][sat[dense_mask] > 40]
                if dense_hues.size > 0:
                    rarity, conf = _rarity_from_hue(float(np.median(dense_hues)))
                    if conf:
                        return rarity, conf

    return 'N', True   # no solid coloured frame → N rarity


def _apply_sort_order_constraint(
    results: List[Dict],
    id_to_rarity: Dict[str, str],
) -> List[Dict]:
    """Correct anomalous item IDs by enforcing inventory sort-order monotonicity.

    The game shows items in ascending or descending ID order (the default sorts).
    An ID that violates monotonicity with both neighbours—while those neighbours
    are mutually consistent—is replaced by the unique database item that fits the
    gap and shares the same detected rarity.  Corrections are skipped when the
    candidate set is ambiguous (0 or ≥2 matches), preventing false positives.
    Handles partial families (e.g. user only owns R/SR/SSR of a group, not N).

    Runs multiple passes until convergence so that consecutive wrong detections
    (e.g. two bad IDs in a row) are each caught after their neighbour is fixed.
    """
    if len(results) < 2:
        return results

    # Sort by grid position (row-major) — stable for the lifetime of this call
    order = sorted(range(len(results)),
                   key=lambda i: results[i]['row'] * 5 + results[i]['col'])
    s = [results[i] for i in order]
    n = len(s)

    def _unique_candidate(lo: int, hi: int, rarity: str) -> Optional[str]:
        """Return the single item_id in (lo, hi) exclusive with matching rarity, or None."""
        hits = [x for x in range(lo + 1, hi)
                if id_to_rarity.get(str(x)) == rarity]
        return str(hits[0]) if len(hits) == 1 else None

    # Determine global sort direction once (majority vote over original IDs)
    ids0 = [int(r['itemId']) for r in s]
    asc  = sum(1 for i in range(n - 1) if ids0[i + 1] > ids0[i])
    desc = sum(1 for i in range(n - 1) if ids0[i + 1] < ids0[i])
    ascending = asc >= desc

    # Iterate until no more corrections are made (max n passes)
    for _pass in range(n):
        ids = [int(r['itemId']) for r in s]
        changed = False

        # Interior positions: anomaly iff neighbours are mutually consistent but curr is out.
        # When immediate neighbours are inconsistent (one may itself be wrong), also try
        # the extended windows (i-2, i+1) and (i-1, i+2) as fallback.
        for i in range(1, n - 1):
            prev_id, curr_id, next_id = ids[i - 1], ids[i], ids[i + 1]
            rarity = s[i]['rarity']
            fix = None
            if ascending:
                if prev_id < next_id and not (prev_id < curr_id < next_id):
                    # Standard: both immediate neighbours are consistent
                    fix = _unique_candidate(prev_id, next_id, rarity)
                elif prev_id >= next_id:
                    # Neighbours inconsistent — one may be wrong; try extended windows
                    if i >= 2:
                        far_prev = ids[i - 2]
                        if far_prev < next_id and not (far_prev < curr_id < next_id):
                            fix = _unique_candidate(far_prev, next_id, rarity)
                    if fix is None and i < n - 2:
                        far_next = ids[i + 2]
                        if prev_id < far_next and not (prev_id < curr_id < far_next):
                            fix = _unique_candidate(prev_id, far_next, rarity)
            else:
                if prev_id > next_id and not (prev_id > curr_id > next_id):
                    fix = _unique_candidate(next_id, prev_id, rarity)
                elif prev_id <= next_id:
                    if i >= 2:
                        far_prev = ids[i - 2]
                        if far_prev > next_id and not (far_prev > curr_id > next_id):
                            fix = _unique_candidate(next_id, far_prev, rarity)
                    if fix is None and i < n - 2:
                        far_next = ids[i + 2]
                        if prev_id > far_next and not (prev_id > curr_id > far_next):
                            fix = _unique_candidate(far_next, prev_id, rarity)
            if fix:
                s[i]['itemId'] = fix
                ids[i] = int(fix)
                changed = True

        # Edge position 0
        rarity = s[0]['rarity']
        if ascending and ids[0] >= ids[1]:
            # Direction violation: ids[0] is too high
            lo, hi = max(0, ids[1] - 4), ids[1]
            fix = _unique_candidate(lo, hi, rarity)
            if fix:
                s[0]['itemId'] = fix
                ids[0] = int(fix)
                changed = True
        elif ascending and n >= 3 and ids[1] < ids[2] and ids[1] - ids[0] > 4:
            # ids[0] is going right direction but cross-family gap is suspicious;
            # ids[1] is confirmed by ids[2] so use it as anchor
            lo, hi = max(0, ids[1] - 4), ids[1]
            fix = _unique_candidate(lo, hi, rarity)
            if fix:
                s[0]['itemId'] = fix
                ids[0] = int(fix)
                changed = True
        elif not ascending and ids[0] <= ids[1]:
            # Direction violation: ids[0] is too low
            lo, hi = ids[1], ids[1] + 4
            fix = _unique_candidate(lo, hi, rarity)
            if fix:
                s[0]['itemId'] = fix
                ids[0] = int(fix)
                changed = True
        elif not ascending and n >= 3 and ids[1] > ids[2] and ids[0] - ids[1] > 4:
            # Descending: ids[0] is going right direction but cross-family gap suspicious
            lo, hi = ids[1], ids[1] + 4
            fix = _unique_candidate(lo, hi, rarity)
            if fix:
                s[0]['itemId'] = fix
                ids[0] = int(fix)
                changed = True

        # Edge position n-1
        rarity = s[-1]['rarity']
        if ascending and ids[-1] <= ids[-2]:
            # Direction violation: ids[-1] is too low
            lo, hi = ids[-2], ids[-2] + 4
            fix = _unique_candidate(lo, hi, rarity)
            if fix:
                s[-1]['itemId'] = fix
                changed = True
        elif ascending and n >= 3 and ids[-3] < ids[-2] and ids[-1] - ids[-2] > 4:
            # ids[-1] is going right direction but cross-family gap is suspicious;
            # ids[-2] is confirmed by ids[-3] so use it as anchor
            lo, hi = ids[-2], ids[-2] + 4
            fix = _unique_candidate(lo, hi, rarity)
            if fix:
                s[-1]['itemId'] = fix
                changed = True
        elif not ascending and ids[-1] >= ids[-2]:
            # Direction violation: ids[-1] is too high
            lo, hi = max(0, ids[-2] - 4), ids[-2]
            fix = _unique_candidate(lo, hi, rarity)
            if fix:
                s[-1]['itemId'] = fix
                changed = True
        elif not ascending and n >= 3 and ids[-3] > ids[-2] and ids[-2] - ids[-1] > 4:
            # Descending: ids[-1] going right direction but cross-family gap suspicious
            lo, hi = max(0, ids[-2] - 4), ids[-2]
            fix = _unique_candidate(lo, hi, rarity)
            if fix:
                s[-1]['itemId'] = fix
                changed = True

        if not changed:
            break

    # Write corrections back into original results list
    for orig_i in range(n):
        results[order[orig_i]] = s[orig_i]
    return results


def _id_range_group(item_id: int) -> int:
    """Map item ID to a coarse category for cross-range outlier detection.

    Groups:  0 = misc (0-99), 1 = artifacts (100-299), 2 = other (300-2999),
             3 = blu-rays (3000-3999), 4 = skill books (4000-4999),
             5 = favors/gifts (5000+).
    """
    if item_id < 100:
        return 0
    if item_id < 300:
        return 1
    if item_id < 3000:
        return 2
    if item_id < 4000:
        return 3
    if item_id < 5000:
        return 4
    return 5


def _fix_range_outliers(
    results: List[Dict],
    net,
    embeddings: np.ndarray,
    labels: Dict[str, str],
    rarities: Optional[Dict[str, str]],
    icon_crops: List[np.ndarray],
) -> List[Dict]:
    """Fix items whose ID range is inconsistent with surrounding items.

    The sort-order constraint cannot fix cross-range outliers when multiple
    candidates share the same rarity (e.g. all favor items are SR).  This pass
    detects items whose *range group* disagrees with their neighbours and
    re-classifies them with CLIP restricted to items in the expected range.
    """
    if len(results) < 3:
        return results

    order = sorted(range(len(results)),
                   key=lambda i: results[i]['row'] * 5 + results[i]['col'])
    s = [results[i] for i in order]
    crops = [icon_crops[i] for i in order]
    n = len(s)

    # Determine sort direction (same logic as _apply_sort_order_constraint)
    ids0 = [int(r['itemId']) for r in s]
    asc_count = sum(1 for i in range(n - 1) if ids0[i + 1] > ids0[i])
    desc_count = sum(1 for i in range(n - 1) if ids0[i + 1] < ids0[i])
    ascending = asc_count >= desc_count

    for _pass in range(n):
        ids = [int(r['itemId']) for r in s]
        groups = [_id_range_group(iid) for iid in ids]
        changed = False

        for i in range(n):
            # Gather neighbour groups in a ±2 window
            neighbor_groups: Dict[int, int] = {}
            for j in range(max(0, i - 2), min(n, i + 3)):
                if j == i:
                    continue
                g = groups[j]
                neighbor_groups[g] = neighbor_groups.get(g, 0) + 1

            if not neighbor_groups:
                continue

            dominant_group = max(neighbor_groups, key=neighbor_groups.get)
            dominant_count = neighbor_groups[dominant_group]
            my_count = neighbor_groups.get(groups[i], 0)

            # Skip if item matches dominant group, or no clear majority
            if groups[i] == dominant_group or dominant_count <= my_count:
                continue

            # Verify the item actually violates range-group monotonicity.
            # Only check *earlier* positions to avoid false positives when most
            # items on the page are already mis-classified (e.g. Screenshot10
            # where CLIP maps many favor items to artifacts — checking later
            # positions would incorrectly flag the few correct favor items).
            violates = False
            for j in range(max(0, i - 2), i):
                if ascending and groups[i] < groups[j]:
                    violates = True
                    break
                if not ascending and groups[i] > groups[j]:
                    violates = True
                    break

            # Special case: position 0 has no earlier neighbours.  Use a
            # page-level check — if ≥80 % of the *other* items belong to a
            # single different group, position 0 is very likely an outlier.
            # The high threshold prevents false positives on pages where
            # most items are already mis-classified (e.g. Screenshot10).
            if not violates and i == 0 and n >= 4:
                rest_groups: Dict[int, int] = {}
                for j in range(1, n):
                    g = groups[j]
                    rest_groups[g] = rest_groups.get(g, 0) + 1
                if rest_groups:
                    pg_dom = max(rest_groups, key=rest_groups.get)
                    pg_frac = rest_groups[pg_dom] / (n - 1)
                    if pg_frac >= 0.80 and groups[0] != pg_dom:
                        violates = True

            if not violates:
                continue

            # Re-classify restricted to dominant range group.
            # Try rarity-matched first; fall back to any-rarity if no
            # rarity-matched candidates exist (e.g. SSR favors mis-detected
            # as N/R — no N/R favors exist in the database).
            crop = crops[i]
            if crop is None or crop.size == 0:
                continue

            rarity = s[i]['rarity']
            scores = _get_embed_scores(crop, net, embeddings)

            # Try rarity-matched first
            rar_candidates = {idx_str: lid for idx_str, lid in labels.items()
                              if _id_range_group(int(lid)) == dominant_group
                              and (not rarities or rarities.get(idx_str) == rarity)}
            best_idx, best_score = _best_match_in_group(scores, rar_candidates)

            # Fallback: any rarity in range (handles rarity mis-detection)
            if best_idx is None or best_score < _RANGE_FIX_THRESHOLD:
                any_candidates = {idx_str: lid for idx_str, lid in labels.items()
                                  if _id_range_group(int(lid)) == dominant_group}
                best_idx, best_score = _best_match_in_group(scores, any_candidates)

            if best_idx is not None and best_score >= _RANGE_FIX_THRESHOLD:
                new_id = labels.get(str(best_idx))
                if new_id and new_id != s[i]['itemId']:
                    s[i]['itemId'] = new_id
                    changed = True

        if not changed:
            break

    # --- Second pass: run-based correction for consecutive wrong items ---
    # Detect contiguous runs of the same group.  When a short run is
    # sandwiched between runs of a different group forming a "bump" —
    # e.g. …g2,g2 | g4,g4,g4 | g3,g3,g3… in ascending order — the
    # g4→g3 transition violates monotonicity, so the g4 run is suspect.
    # Fix each item in the short run by re-classifying with CLIP
    # restricted to the surrounding group.
    ids = [int(r['itemId']) for r in s]
    groups = [_id_range_group(iid) for iid in ids]

    def _build_runs(grps):
        out = []
        ri = 0
        nn = len(grps)
        while ri < nn:
            rj = ri + 1
            while rj < nn and grps[rj] == grps[ri]:
                rj += 1
            out.append((ri, rj, grps[ri]))
            ri = rj
        return out

    runs = _build_runs(groups)

    if len(runs) >= 3:
        changed_run = True
        while changed_run:
            changed_run = False
            ids = [int(r['itemId']) for r in s]
            groups = [_id_range_group(iid) for iid in ids]
            runs = _build_runs(groups)

            for ridx in range(1, len(runs) - 1):
                r_start, r_end, r_grp = runs[ridx]
                r_len = r_end - r_start
                _, _, prev_grp = runs[ridx - 1]
                _, _, next_grp = runs[ridx + 1]

                # Detect "bump": in ascending order, prev ≤ run > next means
                # the run overshoots.  The correct group is next_grp.
                is_bump = False
                target_grp = None
                if ascending:
                    if r_grp > next_grp and prev_grp <= next_grp:
                        is_bump = True
                        target_grp = next_grp
                else:
                    if r_grp < next_grp and prev_grp >= next_grp:
                        is_bump = True
                        target_grp = next_grp

                if not is_bump or target_grp is None:
                    continue

                # Only fix short runs (≤ 3) and only when the next run is
                # at least as long — avoids over-correcting when ambiguous.
                next_len = runs[ridx + 1][1] - runs[ridx + 1][0]
                if r_len > 3 or r_len > next_len:
                    continue

                target_candidates = {idx_str: lid for idx_str, lid in labels.items()
                                     if _id_range_group(int(lid)) == target_grp}
                for pos in range(r_start, r_end):
                    crop = crops[pos]
                    if crop is None or crop.size == 0:
                        continue
                    scores = _get_embed_scores(crop, net, embeddings)
                    best_idx, best_score = _best_match_in_group(scores, target_candidates)
                    if best_idx is not None and best_score >= _RANGE_FIX_THRESHOLD:
                        new_id = labels.get(str(best_idx))
                        if new_id and new_id != s[pos]['itemId']:
                            s[pos]['itemId'] = new_id
                            changed_run = True

    for orig_i in range(n):
        results[order[orig_i]] = s[orig_i]
    return results




def _detect_row_boundaries(grid_bgr: np.ndarray, n_rows: int = 4) -> List[int]:
    """Detect actual row boundaries by finding grey horizontal gaps.

    When the game inventory is scroll-misaligned, the fixed grid boundaries
    cut through items rather than at the grey inter-cell gaps.  Returns
    *n_rows + 1* y-coordinates: [row0_top, row1_top, ..., grid_bottom].
    Falls back to equally-spaced boundaries when detection fails.
    """
    from scipy.ndimage import uniform_filter1d

    gh = grid_bgr.shape[0]
    hsv = cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2HSV)
    row_sat = np.array([float(hsv[y, :, 1].mean()) for y in range(gh)])
    smoothed = uniform_filter1d(row_sat, size=5)

    # Collect contiguous runs of low-saturation rows (< 15 -> grey strip)
    in_gap = False
    gaps: List[Tuple[int, int]] = []
    gs = 0
    for y in range(gh):
        if smoothed[y] < 15 and not in_gap:
            in_gap = True
            gs = y
        elif smoothed[y] >= 15 and in_gap:
            in_gap = False
            gaps.append((gs, y))
    if in_gap:
        gaps.append((gs, gh))

    mids = [int((s + e) / 2) for s, e in gaps]

    # Only use dynamic detection when the scroll offset is significant
    # (first gap > 15px).  Small offsets (< 15px) are just normal cell
    # borders — equal spacing works better and avoids boundary jitter
    # that can change CLIP crops at cell edges.
    _MIN_SCROLL_OFFSET = 15

    if len(mids) >= n_rows - 1 and mids[0] >= _MIN_SCROLL_OFFSET:
        # If the first gap is far from the top (> grid_height/8), it's an
        # inter-row boundary, not a top-border gap — the grid is well-aligned
        # and items start at y ~= 0.  Prepend 0 so row 0 begins at the top.
        if mids[0] > gh // 8:
            boundaries = [0] + mids[:n_rows - 1] + [gh]
        else:
            boundaries = mids[:n_rows] + [gh]

        heights = [boundaries[i + 1] - boundaries[i] for i in range(n_rows)]
        median_h = float(np.median(heights[:n_rows - 1]))
        if median_h > 0 and all(
            abs(h - median_h) < median_h * 0.35 for h in heights[:-1]
        ):
            return boundaries

    # Fallback: equal spacing
    return [int(i * gh / n_rows) for i in range(n_rows)] + [gh]


def _extract_icon_crop(slot: np.ndarray) -> np.ndarray:
    """Square-normalise a grid cell and crop to the icon region."""
    sh, sw = slot.shape[:2]
    side = min(sh, sw)
    if sh > sw:
        trim_v = (sh - sw) // 2
        slot_sq = slot[trim_v:trim_v + side, :]
    elif sw > sh:
        trim_h = (sw - sh) // 2
        slot_sq = slot[:, trim_h:trim_h + side]
    else:
        slot_sq = slot
    return slot_sq[
        int(side * ICON_CROP_TOP):int(side * ICON_CROP_BOT),
        int(side * ICON_CROP_LEFT):int(side * ICON_CROP_RIGHT),
    ]


def _extract_quantity_crop(
    grid: np.ndarray, y0: int, y1: int, x0: int, sh: int, sw: int,
) -> np.ndarray:
    """Crop the quantity-text region from the grid (may extend below cell)."""
    qty_top   = y0 + int(sh * QTY_CROP_TOP_FRAC)
    qty_bot   = min(y1 + int(sh * QTY_CROP_OVERFLOW), grid.shape[0])
    qty_left  = x0 + int(sw * QTY_CROP_LEFT_FRAC)
    qty_right = min(x0 + int(sw * QTY_CROP_RIGHT_FRAC), grid.shape[1])
    return grid[qty_top:qty_bot, qty_left:qty_right]


def _compute_confidence(icon_score: float, digit_score: float) -> float:
    """Blend icon and digit scores into a clamped confidence value."""
    return max(0.0, min(1.0,
        _CONF_ICON_WEIGHT * icon_score + _CONF_DIGIT_WEIGHT * digit_score))


def _best_match_in_group(
    scores: np.ndarray,
    candidates: Dict[str, str],
) -> Tuple[Optional[int], float]:
    """Find the embedding index with the highest CLIP score among candidates.

    ``candidates`` maps embedding-row index (str) to item-ID string.
    Returns ``(best_embedding_idx, best_score)``; idx is ``None`` when
    *candidates* is empty.
    """
    best_idx: Optional[int] = None
    best_score = -1.0
    for idx_str in candidates:
        idx = int(idx_str)
        if scores[idx] > best_score:
            best_score = scores[idx]
            best_idx = idx
    return best_idx, best_score


def _process_grid_cells(
    grid: np.ndarray,
    rows: int,
    cols: int,
    cell_w: float,
    row_bounds: List[int],
    net,
    embeddings: np.ndarray,
    labels: Dict[str, str],
    rarities: Dict[str, str],
    shapes: Optional[Dict[str, str]],
) -> Tuple[List[Dict], List[np.ndarray]]:
    """Classify every grid cell and return (results, icon_crops)."""
    results: List[Dict] = []
    icon_crops: List[np.ndarray] = []

    for row in range(rows):
        for col in range(cols):
            x0 = int(col * cell_w)
            y0 = row_bounds[row]
            x1 = int((col + 1) * cell_w)
            y1 = row_bounds[row + 1]

            slot = grid[y0:y1, x0:x1]
            if slot.size == 0:
                continue

            rarity, _ = _classify_rarity(slot)

            sh, sw = slot.shape[:2]

            icon_crop = _extract_icon_crop(slot)
            if icon_crop.size == 0:
                continue

            query_shape = _compute_query_shape(icon_crop) if shapes else None
            best_id, best_score = _classify_icon_embed(
                icon_crop, net, embeddings, labels, rarities, rarity,
                shapes=shapes, query_shape=query_shape,
            )
            if best_id is None or best_score < EMBED_SCORE_THRESHOLD:
                continue

            icon_score = best_score

            # Quantity crop — taken from the GRID (not the slot) so it can
            # overflow the cell boundary by a few pixels.
            quantity_crop = _extract_quantity_crop(grid, y0, y1, x0, sh, sw)
            quantity, digit_score = _read_quantity(quantity_crop)

            confidence = _compute_confidence(icon_score, digit_score)

            results.append({
                'row': row,
                'col': col,
                'itemId': best_id,
                'rarity': rarity,
                'quantity': int(quantity),
                'confidence': round(confidence, 4)
            })
            icon_crops.append(icon_crop)

    return results, icon_crops


def _recover_favor_items(
    results: List[Dict],
    grid: np.ndarray,
    rows: int,
    cols: int,
    cell_w: float,
    row_bounds: List[int],
    net,
    embeddings: np.ndarray,
    labels: Dict[str, str],
    rarities: Dict[str, str],
) -> Dict[tuple, np.ndarray]:
    """Recover dropped / mis-classified favor items via SSR detection passes.

    Mutates *results* in-place and returns a dict mapping ``(row, col)`` to
    the icon crop for every cell that was recovered or replaced.
    """
    recovered_crops: Dict[tuple, np.ndarray] = {}

    favor_count = sum(1 for r in results if _id_range_group(int(r['itemId'])) == 5)
    if favor_count < 2:
        return recovered_crops

    # Find the last SR favor (5000-5099) in grid order
    last_sr_favor_pos = -1
    for r in results:
        iid = int(r['itemId'])
        if 5000 <= iid < 5100:
            pos = r['row'] * 5 + r['col']
            if pos > last_sr_favor_pos:
                last_sr_favor_pos = pos

    # Collect favor item indices
    all_favor_indices: Dict[str, str] = {}
    ssr_favor_indices: Dict[str, str] = {}
    for idx_str, lid in labels.items():
        if _id_range_group(int(lid)) == 5:
            all_favor_indices[idx_str] = lid
            if int(lid) >= 5100:
                ssr_favor_indices[idx_str] = lid

    # --- Part A: SSR favor detection (cells after last SR favor) ----------
    if last_sr_favor_pos >= 0 and ssr_favor_indices:
        filled = {(r['row'], r['col']) for r in results}

        for row in range(rows):
            for col in range(cols):
                pos = row * 5 + col
                if pos <= last_sr_favor_pos:
                    continue

                x0 = int(col * cell_w)
                y0 = row_bounds[row]
                x1 = int((col + 1) * cell_w)
                y1 = row_bounds[row + 1]
                slot = grid[y0:y1, x0:x1]
                if slot.size == 0:
                    continue

                sh, sw = slot.shape[:2]
                icon_crop = _extract_icon_crop(slot)
                if icon_crop.size == 0:
                    continue

                scores = _get_embed_scores(icon_crop, net, embeddings)
                best_idx, best_score = _best_match_in_group(scores, all_favor_indices)

                cell_threshold = (_RANGE_FIX_THRESHOLD
                                  if (row, col) in filled
                                  else _DROP_RECOVER_THRESHOLD)
                if best_idx is None or best_score < cell_threshold:
                    continue

                new_id = labels[str(best_idx)]
                new_rar = rarities.get(str(best_idx), 'N')

                recovered_crops[(row, col)] = icon_crop

                if (row, col) in filled:
                    existing = next(r for r in results
                                    if r['row'] == row and r['col'] == col)
                    cur_grp = _id_range_group(int(existing['itemId']))
                    if cur_grp == 5:
                        continue  # already a favor — keep it
                    existing['itemId'] = new_id
                    existing['rarity'] = new_rar
                else:
                    quantity_crop = _extract_quantity_crop(grid, y0, y1, x0, sh, sw)
                    quantity, digit_score = _read_quantity(quantity_crop)
                    confidence = _compute_confidence(best_score, digit_score)

                    results.append({
                        'row': row,
                        'col': col,
                        'itemId': new_id,
                        'rarity': new_rar,
                        'quantity': int(quantity),
                        'confidence': round(float(confidence), 4),
                    })

    # --- Part B: recover dropped favor cells (anywhere on the page) ------
    filled = {(r['row'], r['col']) for r in results}
    for row in range(rows):
        for col in range(cols):
            if (row, col) in filled:
                continue
            adj_favor = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                for r2 in results:
                    if r2['row'] == nr and r2['col'] == nc:
                        if _id_range_group(int(r2['itemId'])) == 5:
                            adj_favor += 1
                        break
            if adj_favor < 1:
                continue

            x0 = int(col * cell_w)
            y0 = row_bounds[row]
            x1 = int((col + 1) * cell_w)
            y1 = row_bounds[row + 1]
            slot = grid[y0:y1, x0:x1]
            if slot.size == 0:
                continue
            sh, sw = slot.shape[:2]
            icon_crop = _extract_icon_crop(slot)
            if icon_crop.size == 0:
                continue

            scores = _get_embed_scores(icon_crop, net, embeddings)
            best_idx, best_score = _best_match_in_group(scores, all_favor_indices)

            if best_idx is not None and best_score >= _DROP_RECOVER_THRESHOLD:
                recovered_crops[(row, col)] = icon_crop
                rec_id = labels[str(best_idx)]
                rec_rar = rarities.get(str(best_idx), 'N')
                quantity_crop = _extract_quantity_crop(grid, y0, y1, x0, sh, sw)
                quantity, digit_score = _read_quantity(quantity_crop)
                confidence = _compute_confidence(best_score, digit_score)
                results.append({
                    'row': row,
                    'col': col,
                    'itemId': rec_id,
                    'rarity': rec_rar,
                    'quantity': int(quantity),
                    'confidence': round(float(confidence), 4),
                })

    return recovered_crops


def _enforce_group_sort_order(
    results: List[Dict],
    icon_crops: List[np.ndarray],
    recovered_crops: Dict[tuple, np.ndarray],
    net,
    embeddings: np.ndarray,
    labels: Dict[str, str],
) -> None:
    """Enforce ascending IDs within contiguous same-group runs.

    Mutates *results* in-place.  For each run of ≥3 items in the same range
    group, re-classifies violating positions with CLIP restricted to items
    in the valid ID window ``(prev_id, next_id)``.
    """
    ordered_res = sorted(results, key=lambda r: r['row'] * 5 + r['col'])

    # Build a mapping of icon crops by (row, col).
    crop_map: Dict[tuple, np.ndarray] = {}
    for i, r in enumerate(results):
        if i < len(icon_crops):
            crop_map[(r['row'], r['col'])] = icon_crops[i]
    crop_map.update(recovered_crops)

    # Identify contiguous same-group runs in grid order
    grp_seq = [_id_range_group(int(r['itemId'])) for r in ordered_res]
    group_runs: List[tuple] = []  # (start, end, group)
    ri = 0
    while ri < len(grp_seq):
        rj = ri + 1
        while rj < len(grp_seq) and grp_seq[rj] == grp_seq[ri]:
            rj += 1
        if rj - ri >= 3:
            group_runs.append((ri, rj, grp_seq[ri]))
        ri = rj

    for run_start, run_end, run_grp in group_runs:
        grp_label_idx: Dict[int, int] = {}
        for idx_str, lid in labels.items():
            if _id_range_group(int(lid)) == run_grp:
                grp_label_idx[int(idx_str)] = int(lid)

        if not grp_label_idx:
            continue

        run_len = run_end - run_start

        for _sort_pass in range(run_len):
            any_fixed = False
            run_ids = [int(ordered_res[p]['itemId'])
                       for p in range(run_start, run_end)]

            for ri in range(run_len):
                pos = run_start + ri
                r = ordered_res[pos]
                cur_id = run_ids[ri]

                has_violation = False
                lower_bound = -1
                upper_bound = 999999
                if ri > 0 and cur_id <= run_ids[ri - 1]:
                    has_violation = True
                    lower_bound = run_ids[ri - 1]
                if ri < run_len - 1 and cur_id >= run_ids[ri + 1]:
                    has_violation = True
                    upper_bound = run_ids[ri + 1]

                if not has_violation:
                    continue

                if ri > 0:
                    lower_bound = max(lower_bound, run_ids[ri - 1])
                if ri < run_len - 1:
                    upper_bound = min(upper_bound, run_ids[ri + 1])

                if lower_bound >= upper_bound:
                    continue

                crop = crop_map.get((r['row'], r['col']))
                if crop is None or crop.size == 0:
                    continue

                scores = _get_embed_scores(crop, net, embeddings)
                best_idx = None
                best_score = -1.0
                for emb_idx, item_id in grp_label_idx.items():
                    if item_id <= lower_bound or item_id >= upper_bound:
                        continue
                    if scores[emb_idx] > best_score:
                        best_score = scores[emb_idx]
                        best_idx = emb_idx

                if best_idx is not None and best_score >= _DROP_RECOVER_THRESHOLD:
                    new_id = grp_label_idx[best_idx]
                    r['itemId'] = str(new_id)
                    run_ids[ri] = new_id
                    any_fixed = True

            if not any_fixed:
                break


def parse_inventory(image_bytes: bytes, inventory_type: str) -> List[Dict]:
    if inventory_type not in ('items', 'equipment'):
        return []

    data = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return []

    height, width = image.shape[:2]
    right_half = image[:, width // 2 :]

    left, top, right, bottom = _compute_grid_bounds(right_half)
    if right <= left or bottom <= top:
        return []

    grid = right_half[top:bottom, left:right]
    grid_h, grid_w = grid.shape[:2]

    rows = 4
    cols = 5
    cell_w = grid_w / cols

    # Detect actual row boundaries (handles scroll-misaligned screenshots)
    row_bounds = _detect_row_boundaries(grid, rows)

    # When scroll-offset is detected, the last row is clipped at the grid
    # bottom.  Extend the grid downward so the quantity text isn't cut off.
    scroll_offset = row_bounds[0]
    if scroll_offset > 10:
        extended_bottom = min(bottom + scroll_offset, right_half.shape[0])
        grid = right_half[top:extended_bottom, left:right]
        grid_h = grid.shape[0]
        row_bounds[-1] = grid_h

    net, embeddings, labels = _load_embeddings(inventory_type)
    if net is None:
        print(f'[inventory_parser] No embeddings for {inventory_type} — run embed.py first')
        return []

    rarities = _load_rarities(inventory_type)
    shapes   = _load_shapes(inventory_type)

    results, icon_crops = _process_grid_cells(
        grid, rows, cols, cell_w, row_bounds,
        net, embeddings, labels, rarities, shapes,
    )

    # Build id → rarity lookup from loaded embedding metadata and apply sort-order fix
    id_to_rarity = {item_id: rarities.get(row_idx, 'Unknown')
                    for row_idx, item_id in labels.items()}
    results = _apply_sort_order_constraint(results, id_to_rarity)
    results = _fix_range_outliers(results, net, embeddings, labels, rarities,
                                  icon_crops)

    recovered_crops = _recover_favor_items(
        results, grid, rows, cols, cell_w, row_bounds,
        net, embeddings, labels, rarities,
    )
    _enforce_group_sort_order(
        results, icon_crops, recovered_crops, net, embeddings, labels,
    )

    # _recover_favor_items appends recovered cells at the end of the list.
    # Re-sort into row-major order so the frontend can use index-based layout.
    results.sort(key=lambda r: r['row'] * 5 + r['col'])

    return results
