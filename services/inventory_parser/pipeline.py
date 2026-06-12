import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import ncc_matcher

BASE_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
ICON_CACHE_DIR = os.path.join(CACHE_DIR, 'icons')


GRID_FALLBACK_LEFT_RATIO = 0.06
GRID_FALLBACK_RIGHT_RATIO = 0.95
# Top: just below the "List / filter" header bar (~22% of right-half height)
# Bottom (items): just above the "Item cannot be used here" footer bar (~78%)
# Bottom (equipment): no footer bar — grid runs to ~92% of right-half height.
#   Derivation: 4-row items has cell_h≈151px (grid top=237, bottom=842 on 1080px).
#   5 rows × 151px = 755px → correct bottom = 237+755 = 992 → ratio ≈ 0.919 → 0.92.
#   Using 0.95 inflates cell_h to ~158px, shifting row boundaries and dropping match scores.
# Providing list_header.png + item_cannot_use.png templates overrides these.
GRID_FALLBACK_TOP_RATIO = 0.22
GRID_FALLBACK_BOTTOM_RATIO = 0.78
GRID_FALLBACK_BOTTOM_RATIO_EQUIPMENT = 0.92
GRID_ANCHOR_MARGIN = 5

# Minimum refined NCC score to accept a cell as containing an item. Correct
# matches on the ground-truth set score >= 0.83, and gifts whose cached
# SchaleDB sprite still has JP cover art (localisation drift, e.g. 5009/5023)
# bottom out around 0.46 while remaining correct top-1; empty slots are flat
# background and score near 0. Below NCC_REVIEW_SCORE the match is kept but
# its confidence is capped so the FE flags it for review.
NCC_ACCEPT_SCORE = 0.40
NCC_REVIEW_SCORE = 0.70
# NCC top1-top2 margins below this cap the cell's confidence so the FE flags
# it for review (ground-truth minimum margin was 0.0154).
NCC_LOW_MARGIN = 0.012


# Confidence blend weights (icon match score vs digit OCR score).
_CONF_ICON_WEIGHT  = 0.7
_CONF_DIGIT_WEIGHT = 0.3


def warm_icon_db() -> None:
    # Preload sprite canvases for the matcher (fast — just disk I/O; no ML
    # model on this path).
    ncc_matcher.warm()
    # Initialise the Gemini SDK client at startup (cheap, no network call) so
    # the "Gemini client ready" log and any config issues (missing key, import
    # error) surface at boot instead of on the first user request. Falls back
    # silently to lazy init if the key isn't present.
    _get_gemini_client()


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Flash OCR — the sole quantity reader (Florence-2 removed). A chain of
# free-tier models is tried in order; rate limits are PER-MODEL per-project, so
# chaining multiplies daily capacity. When the whole chain is exhausted, callers
# get quantity=0 + low confidence and the user fills blanks manually.
# ─────────────────────────────────────────────────────────────────────────────
from datetime import datetime, timezone as _tz, timedelta

try:
    from zoneinfo import ZoneInfo
    _PACIFIC = ZoneInfo('America/Los_Angeles')
except Exception:  # tzdata unavailable — fixed UTC-8 fallback (ignores DST drift)
    _PACIFIC = _tz(timedelta(hours=-8))

# Ordered fallback chain of (model, daily_cap). Free-tier RPD is per-model, so a
# 429 on one model just advances to the next — combined ≈ 544 RPD/day. Caps are
# set just under each model's observed free-tier RPD ceiling. All entries are
# vision-capable text-out models confirmed available on this project's free tier.
_GEMINI_MODEL_CHAIN = [
    ('gemini-3.1-flash-lite', 490),  # 500 RPD / 15 RPM — primary
    ('gemini-2.5-flash',       18),  # 20 RPD / 5 RPM
    ('gemini-3.5-flash',       18),  # 20 RPD / 5 RPM
    ('gemini-2.5-flash-lite',  18),  # 20 RPD / 10 RPM
]
_GEMINI_CLIENT = None
# {'date': pacific_date, 'models': {model: {'success': int, 'adaptive_cap': int}}}
_gemini_state = {'date': None, 'models': {}}

# 503/500/UNAVAILABLE are transient Google-side overloads (free tier gets shed
# first under load). Unlike 429 quota errors, the same request usually succeeds
# on a short retry — so retry these a couple times before advancing to the next
# model in the chain. Worst-case added latency per model is 2+4=6s.
_GEMINI_TRANSIENT_RETRIES = 2
_GEMINI_RETRY_BACKOFF = 2.0  # seconds; doubles each attempt (2s, then 4s)


def _get_gemini_client():
    """Lazy-load the Gemini client. Returns None if no API key configured."""
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return None
        try:
            from google import genai
            _GEMINI_CLIENT = genai.Client(api_key=api_key)
            primary = _GEMINI_MODEL_CHAIN[0][0]
            print(f'[inventory_parser] Gemini client ready (primary: {primary})')
        except ImportError:
            print('[inventory_parser] google-genai not installed — skipping Gemini')
            return None
    return _GEMINI_CLIENT


def _reset_gemini_day_if_needed() -> None:
    """Clear per-model counters at Pacific midnight (Google's RPD reset boundary)."""
    today = datetime.now(_PACIFIC).date()
    if _gemini_state['date'] != today:
        _gemini_state['date'] = today
        _gemini_state['models'] = {}


def _gemini_model_state(model: str, cap: int) -> dict:
    """Get-or-create the per-model daily counter."""
    ms = _gemini_state['models'].get(model)
    if ms is None:
        ms = {'success': 0, 'adaptive_cap': cap}
        _gemini_state['models'][model] = ms
    return ms


def _build_qty_prompt(rows: int, cols: int) -> str:
    return (
        f"This is a Blue Archive game inventory grid with {rows} rows × {cols} "
        f"columns of item cells. Each cell has an icon and a quantity number "
        f"prefixed with '×' at the bottom-right.\n\n"
        f"Read the quantity number for every cell, top-to-bottom, left-to-right. "
        f"If a cell has no readable number, use 0. Return ONLY a JSON array, "
        f"no markdown fences, no explanation:\n"
        f'[{{"row":0,"col":0,"qty":1197}},{{"row":0,"col":1,"qty":607}},...]'
    )


def _build_multi_qty_prompt(n: int, rows: int, cols: int) -> str:
    return (
        f"There are {n} Blue Archive inventory grids, given as the first {n} images "
        f"in order (grid 0 to grid {n - 1}). Each grid has {rows} rows × {cols} "
        f"columns; each cell has an icon and a quantity prefixed with '×' at the "
        f"bottom-right.\n\n"
        f"Read the quantity for every cell in every grid. If a cell has no readable "
        f"number, use 0. Return ONLY a JSON array, no markdown fences, each entry "
        f"tagged with its grid index:\n"
        f'[{{"grid":0,"row":0,"col":0,"qty":1197}},{{"grid":0,"row":0,"col":1,"qty":607}},...]'
    )


def _strip_json_fence(text: str) -> str:
    text = (text or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    return text


def _parse_qty_json(text: str, model: str) -> Optional[Dict[Tuple[int, int], int]]:
    """Parse a single-grid JSON response into {(row,col): qty}, or None."""
    text = _strip_json_fence(text)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        print(f'[gemini] {model} non-JSON response: {e!r}; raw={text[:200]!r}')
        return None
    if not isinstance(parsed, list):
        print(f'[gemini] {model} expected list, got {type(parsed).__name__}')
        return None
    out: Dict[Tuple[int, int], int] = {}
    for item in parsed:
        try:
            r = int(item['row']); c = int(item['col']); q = int(item['qty'])
            out[(r, c)] = q
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _parse_multi_qty_json(text: str, model: str) -> Optional[Dict[Tuple[int, int, int], int]]:
    """Parse a multi-grid JSON response into {(grid,row,col): qty}, or None."""
    text = _strip_json_fence(text)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        print(f'[gemini] {model} non-JSON (multi-grid): {e!r}; raw={text[:200]!r}')
        return None
    if not isinstance(parsed, list):
        return None
    out: Dict[Tuple[int, int, int], int] = {}
    for item in parsed:
        try:
            g = int(item['grid']); r = int(item['row']); c = int(item['col']); q = int(item['qty'])
            out[(g, r, c)] = q
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _try_gemini_model(client, model: str, contents: list, parse_fn, ms: dict):
    """Try one model (with transient-error retries). `contents` is the full
    generate_content payload (images + prompt); `parse_fn(text, model)` turns
    the response into a result dict. Returns the parsed dict on success, or None
    to signal the caller should advance to the next model. On 429 it marks this
    model exhausted for the rest of the day."""
    response = None
    for attempt in range(_GEMINI_TRANSIENT_RETRIES + 1):
        try:
            response = client.models.generate_content(model=model, contents=contents)
            break
        except Exception as e:  # noqa: BLE001
            err_str = str(e).lower()
            # Quota / rate limit — this model is done for the day; advance chain.
            if any(s in err_str for s in ('429', 'quota', 'resource_exhausted', 'rate')):
                ms['adaptive_cap'] = ms['success']
                print(f'[gemini] {model} quota hit at {ms["success"]} reqs — advancing chain')
                return None
            # Transient server overload — retry with backoff before advancing.
            is_transient = any(s in err_str for s in
                               ('503', '500', 'unavailable', 'overloaded', 'internal'))
            if is_transient and attempt < _GEMINI_TRANSIENT_RETRIES:
                delay = _GEMINI_RETRY_BACKOFF * (2 ** attempt)
                print(f'[gemini] {model} transient error (attempt {attempt + 1}/'
                      f'{_GEMINI_TRANSIENT_RETRIES + 1}), retrying in {delay:.0f}s: {e}')
                time.sleep(delay)
                continue
            print(f'[gemini] {model} error: {e} — advancing chain')
            return None

    if response is None:
        return None
    out = parse_fn(response.text, model)
    if out is not None:
        ms['success'] += 1
    return out


def _run_gemini_chain(contents: list, parse_fn):
    """Run `contents` through the model chain, advancing on per-model exhaustion
    or failure. Returns parse_fn's result on first success, else None."""
    client = _get_gemini_client()
    if client is None:
        return None
    _reset_gemini_day_if_needed()
    for model, cap in _GEMINI_MODEL_CHAIN:
        ms = _gemini_model_state(model, cap)
        if ms['success'] >= ms['adaptive_cap']:
            continue  # this model exhausted today — try the next
        out = _try_gemini_model(client, model, contents, parse_fn, ms)
        if out is not None:
            return out
    return None  # whole chain exhausted/failed


def _grid_to_pil(grid_bgr: np.ndarray):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB))


def _gemini_read_all_quantities(
    grid_bgr: np.ndarray, rows: int, cols: int,
) -> Optional[Dict[Tuple[int, int], int]]:
    """Read every quantity in one grid via the Gemini model chain.

    Returns {(row, col): quantity} on the first model success, or ``None`` when
    the entire chain is unavailable/exhausted (callers then emit quantity=0 +
    low confidence for manual entry).
    """
    contents = [_grid_to_pil(grid_bgr), _build_qty_prompt(rows, cols)]
    return _run_gemini_chain(contents, _parse_qty_json)


def _gemini_read_all_quantities_batched(
    grids_meta: List[Tuple[np.ndarray, int, int]],
) -> Optional[Dict[Tuple[int, int, int], int]]:
    """Read quantities for up to 3 grids in ONE Gemini call (RPD-efficient).

    `grids_meta` is a list of (grid_bgr, rows, cols), all same inventory type.
    Returns {(grid_idx, row, col): quantity} or ``None`` if the chain
    failed/exhausted — callers should then retry per-grid before degrading.
    Multi-grid batching is reliable up to 3 grids (validated); beyond that the
    model starts scrambling the middle grid, so callers must cap the batch.
    """
    n = len(grids_meta)
    rows, cols = grids_meta[0][1], grids_meta[0][2]
    contents = [_grid_to_pil(g) for g, _, _ in grids_meta]
    contents.append(_build_multi_qty_prompt(n, rows, cols))
    return _run_gemini_chain(contents, _parse_multi_qty_json)


def _lookup_or_read_quantity(
    slot_bgr: np.ndarray,
    row: int,
    col: int,
    qty_lookup: Optional[Dict[Tuple[int, int], int]],
) -> Tuple[int, float]:
    """Return the Gemini-read quantity for this cell. When the cell is missing
    (entire Gemini chain exhausted/failed), return 0 with low confidence so the
    FE flags it red for manual entry — icon detection still yields the right item.
    """
    if qty_lookup is not None and (row, col) in qty_lookup:
        return qty_lookup[(row, col)], 0.95
    return 0, -1.0  # sentinel: quantity unknown → _compute_confidence forces red


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


def _compute_grid_bounds(
    right_half: np.ndarray,
    inventory_type: str = 'items',
) -> Tuple[int, int, int, int]:
    height, width = right_half.shape[:2]

    list_header_path = os.path.join(ASSETS_DIR, 'list_header.png')
    item_cannot_path = os.path.join(ASSETS_DIR, 'item_cannot_use.png')

    list_match = _match_template(right_half, list_header_path)
    cannot_match = _match_template(right_half, item_cannot_path)

    # Equipment has no "Item cannot be used here" footer, so the grid extends
    # further down.  Use a taller fallback bottom ratio for equipment.
    bottom_ratio = (GRID_FALLBACK_BOTTOM_RATIO_EQUIPMENT
                    if inventory_type == 'equipment'
                    else GRID_FALLBACK_BOTTOM_RATIO)
    top = int(height * GRID_FALLBACK_TOP_RATIO)
    bottom = int(height * bottom_ratio)

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
    # that can change cell crops at the edges.
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


def _compute_confidence(icon_score: float, digit_score: float) -> float:
    """Blend icon and digit scores into a clamped confidence value.

    A negative digit_score is a sentinel meaning "quantity unknown" (Gemini
    chain exhausted): force a red-zone confidence so the FE flags the cell for
    manual entry regardless of how confident the icon match was.
    """
    if digit_score < 0:
        return 0.2
    return max(0.0, min(1.0,
        _CONF_ICON_WEIGHT * icon_score + _CONF_DIGIT_WEIGHT * digit_score))


def _process_grid_cells_ncc(
    grid: np.ndarray,
    rows: int,
    cols: int,
    cell_w: float,
    row_bounds: List[int],
    inventory_type: str,
) -> List[Dict]:
    """Classify every grid cell with the masked-NCC matcher.

    No ID-rewriting correction passes follow (the retired CLIP pipeline
    needed four of them): the matcher separates sibling icons on its own
    (validated 250/250 on marked ground truth), and the old sort-order
    gap-filling heuristics could corrupt confident matches on sparsely-owned
    families. Low-margin cells are surfaced via confidence instead.

    Quantities are merged later by _apply_quantities (so classification can
    run while the Gemini call is in flight); until then each result carries
    its match score/margin in temporary keys.
    """
    bank = ncc_matcher.get_bank(inventory_type, cell_w)
    results: List[Dict] = []

    for row in range(rows):
        for col in range(cols):
            x0 = int(col * cell_w)
            y0 = row_bounds[row]
            x1 = int((col + 1) * cell_w)
            y1 = row_bounds[row + 1]
            slot = grid[y0:y1, x0:x1]
            if slot.size == 0:
                continue

            window = ncc_matcher.cell_window(grid, cell_w, row_bounds, row, col)
            item_id, score, margin = ncc_matcher.match_window(window, bank)
            if item_id is None or score < NCC_ACCEPT_SCORE:
                continue  # empty slot (end of inventory)

            rarity, _ = _classify_rarity(slot)
            results.append({
                'row': row,
                'col': col,
                'itemId': item_id,
                'rarity': rarity,
                '_score': score,
                '_margin': margin,
            })

    return results


def _apply_quantities(results: List[Dict],
                      qty_lookup: Optional[Dict[Tuple[int, int], int]]) -> None:
    """Merge Gemini quantities into classified results (in place) and turn
    the temporary match score/margin into the final confidence."""
    for r in results:
        if qty_lookup is not None and (r['row'], r['col']) in qty_lookup:
            quantity, digit_score = qty_lookup[(r['row'], r['col'])], 0.95
        else:
            quantity, digit_score = 0, -1.0   # unread → red for manual entry
        score = r.pop('_score')
        margin = r.pop('_margin')
        confidence = _compute_confidence(score, digit_score)
        if margin < NCC_LOW_MARGIN or score < NCC_REVIEW_SCORE:
            confidence = min(confidence, 0.5)
        r['quantity'] = int(quantity)
        r['confidence'] = round(confidence, 4)


# Tail-zone + sequence-break confidence demotion thresholds.
# Cells flagged as suspect get their confidence bumped to _SUSPECT_CONF (0.3)
# so the FE highlights them red for manual review. IDs are NOT changed —
# this is purely a UX nudge for cases the matcher can't disambiguate
# (Mode B: out-of-library item → closest visual match in library; Mode C:
# tail-cell ambiguity between real high-ID items and misclassifications).
_TAIL_ZONE_CELLS = 2          # last N grid positions get extra scrutiny
_TAIL_CONF_THRESHOLD = 0.78   # tail cells below this are demoted
_SUSPECT_CONF = 0.3           # red-zone confidence value applied to suspects


def _demote_suspect_confidences(results: List[Dict], rows: int, cols: int) -> None:
    """Mutate `results` in-place to demote confidence on cells that either
    break monotonic sort order or sit in the high-risk tail zone with
    mediocre confidence. Never changes itemId — only the confidence field.

    Sequence-break check: a cell whose neighbours are mutually monotonic
    but the cell itself violates that order is almost certainly wrong
    (Mode B's signature pattern). The favor-boundary 5100 transition does
    NOT trip this because the prev/next direction stays ascending.

    Tail-zone check: the last 2 cells of the grid are statistically the
    highest-error region (end-of-list ambiguity). Cells there with
    confidence below 0.78 get demoted as a precaution.
    """
    if len(results) < 3:
        return

    ordered = sorted(results, key=lambda r: r['row'] * cols + r['col'])
    ids = [int(r['itemId']) for r in ordered]
    n = len(ordered)

    asc = sum(1 for i in range(n - 1) if ids[i + 1] > ids[i])
    desc = sum(1 for i in range(n - 1) if ids[i + 1] < ids[i])
    ascending = asc >= desc

    last_pos = (rows - 1) * cols + (cols - 1)
    tail_threshold_pos = last_pos - _TAIL_ZONE_CELLS + 1

    for i, r in enumerate(ordered):
        suspect = False
        grid_pos = r['row'] * cols + r['col']

        # Interior sequence break: neighbours agree on direction but curr violates
        if 0 < i < n - 1:
            p, c, nx = ids[i - 1], ids[i], ids[i + 1]
            if ascending and p < nx and not (p <= c <= nx):
                suspect = True
            elif (not ascending) and p > nx and not (p >= c >= nx):
                suspect = True

        # Last-cell direction violation (no next neighbour to triangulate)
        if i == n - 1 and i > 0:
            p, c = ids[i - 1], ids[i]
            if ascending and c < p:
                suspect = True
            elif (not ascending) and c > p:
                suspect = True

        # Tail-zone strict: last N cells with mediocre confidence
        if grid_pos >= tail_threshold_pos and r['confidence'] < _TAIL_CONF_THRESHOLD:
            suspect = True

        if suspect and r['confidence'] > _SUSPECT_CONF:
            r['confidence'] = _SUSPECT_CONF


def _extract_grid(image_bytes: bytes, inventory_type: str):
    """Decode one screenshot and locate its inventory grid.

    Returns (grid_bgr, rows, cols, cell_w, row_bounds) or None if the image is
    unreadable or no grid could be located.
    """
    data = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None

    height, width = image.shape[:2]
    right_half = image[:, width // 2:]

    left, top, right, bottom = _compute_grid_bounds(right_half, inventory_type)
    if right <= left or bottom <= top:
        return None

    grid = right_half[top:bottom, left:right]
    grid_w = grid.shape[1]

    # Equipment screenshots show 5 rows; items show 4. _compute_grid_bounds uses
    # a taller fallback bottom ratio for equipment so the grid already covers it.
    rows = 5 if inventory_type == 'equipment' else 4
    cols = 5
    cell_w = grid_w / cols

    row_bounds = _detect_row_boundaries(grid, rows)
    # When scroll-offset is detected, the last row is clipped at the grid bottom;
    # extend the grid downward so the quantity text isn't cut off.
    scroll_offset = row_bounds[0]
    content_bottom = bottom
    if scroll_offset > 10:
        content_bottom = min(bottom + scroll_offset, right_half.shape[0])
        row_bounds[-1] = content_bottom - top

    # Pad the grid canvas below the content so the last row's NCC search
    # window has room: when the page renders lower than the computed bounds,
    # the bottom row's sprites extend past `content_bottom`, and matchTemplate
    # cannot reach an alignment whose template would leave the window — the
    # correct match becomes geometrically impossible. row_bounds keep the
    # unpadded boundary, so quantity crops and the image Gemini receives
    # (grid[:row_bounds[-1]]) are unaffected by the extra strip.
    padded_bottom = min(content_bottom + 40, right_half.shape[0])
    grid = right_half[top:padded_bottom, left:right]

    return grid, rows, cols, cell_w, row_bounds


def _classify_and_correct(grid, rows, cols, cell_w, row_bounds,
                          inventory_type, qty_lookup) -> List[Dict]:
    """Classify one grid's cells, then apply the demote-only suspect check.

    Returns results with 0-based row indices. `qty_lookup` is this grid's
    {(row, col): qty} (or None → quantities degrade to 0 + red confidence).
    Callers offset the row index per screenshot when combining batches.
    """
    results = _process_grid_cells_ncc(
        grid, rows, cols, cell_w, row_bounds, inventory_type)
    _apply_quantities(results, qty_lookup)
    _demote_suspect_confidences(results, rows, cols)
    results.sort(key=lambda r: r['row'] * 5 + r['col'])
    return results


def parse_inventory_batch(images: List[bytes], inventory_type: str) -> List[Dict]:
    """Parse 1-3 screenshots of the same inventory type in one pass.

    Quantities are read in a single batched Gemini call across all grids (RPD-
    efficient); on batch failure each grid is retried with an individual call
    before degrading to quantity=0. Each screenshot's rows are offset by
    grid_index × rows_per_screenshot so the FE groups them as #1, #2, #3.
    """
    if inventory_type not in ('items', 'equipment'):
        return []

    rows_per = 5 if inventory_type == 'equipment' else 4

    extracted = []  # (grid, rows, cols, cell_w, row_bounds) per readable screenshot
    for img_bytes in images:
        meta = _extract_grid(img_bytes, inventory_type)
        if meta is not None:
            extracted.append(meta)
    if not extracted:
        return []

    # Quantities (network-bound Gemini chain) and icon classification
    # (CPU-bound NCC) are independent until the final merge, so the Gemini
    # call runs in a worker thread while the main thread classifies — the
    # request takes max(gemini, ncc) instead of their sum.
    with ThreadPoolExecutor(max_workers=1) as pool:
        qty_future = pool.submit(_read_quantities, extracted)
        classified = [
            _process_grid_cells_ncc(grid, rows, cols, cell_w, row_bounds,
                                    inventory_type)
            for grid, rows, cols, cell_w, row_bounds in extracted
        ]
        per_grid = qty_future.result()

    combined: List[Dict] = []
    for gi, (grid, rows, cols, cell_w, row_bounds) in enumerate(extracted):
        res = classified[gi]
        _apply_quantities(res, per_grid[gi])
        _demote_suspect_confidences(res, rows, cols)
        offset = gi * rows_per
        for r in res:
            r['row'] += offset
        combined.extend(res)

    combined.sort(key=lambda r: r['row'] * 5 + r['col'])
    return combined


def _read_quantities(extracted) -> List[Optional[Dict[Tuple[int, int], int]]]:
    """Per-grid {(row,col): qty} via the Gemini chain. For >1 grid, one
    batched call; on batch failure each grid is retried individually before
    degrading to None (quantity=0 + red confidence downstream).

    grid[:row_bounds[-1]] strips the NCC search pad below the last row so
    Gemini sees the same image it always has (in the scroll-extended case
    row_bounds[-1] is already the full grid height, so this is a no-op).
    """
    if len(extracted) > 1:
        grids_for_gemini = [(g[:rb[-1]], r, c) for g, r, c, _, rb in extracted]
        batched = _gemini_read_all_quantities_batched(grids_for_gemini)
        if batched is not None:
            print(f'[inventory_parser] Gemini batch OCR ok '
                  f'({len(batched)} cells across {len(extracted)} grids)')
            per_grid: List[Optional[Dict[Tuple[int, int], int]]] = \
                [{} for _ in extracted]
            for (gi, r, c), q in batched.items():
                if 0 <= gi < len(extracted):
                    per_grid[gi][(r, c)] = q
            return per_grid
        print('[inventory_parser] batch OCR failed — retrying per-grid')
        return [_gemini_read_all_quantities(g[:rb[-1]], r, c)
                for g, r, c, _, rb in extracted]

    g, r, c, _, rb = extracted[0]
    single = _gemini_read_all_quantities(g[:rb[-1]], r, c)
    if single is not None:
        print(f'[inventory_parser] Gemini OCR ok ({len(single)} cells)')
    else:
        print('[inventory_parser] Gemini chain exhausted — quantities default to 0')
    return [single]


def parse_inventory(image_bytes: bytes, inventory_type: str) -> List[Dict]:
    """Single-screenshot entry point — thin wrapper over parse_inventory_batch."""
    return parse_inventory_batch([image_bytes], inventory_type)
