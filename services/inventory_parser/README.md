# Inventory Parser

Flask service that reads Blue Archive inventory screenshots into structured JSON of items + quantities. Runs as gunicorn at `127.0.0.1:5001` behind the shared nginx.

The architecture (flow diagram, OCR chain, batching rationale, API contract) lives in the [top-level README](../../README.md). This file covers the operational bits — what's in the directory, how to run it locally, and how the `assets/` folder is used.

## Endpoint

| Method | Path | Notes |
|---|---|---|
| POST | `/inventory/parse` | Accepts 1–3 PNG/JPG/WebP screenshots under `images` (or legacy single `image`) + `inventoryType` (`items` \| `equipment`). Returns `{ results: [...] }`. Details in the [top-level API reference](../../README.md#api-reference). |

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask entrypoint — single route, CORS, warmup |
| `pipeline.py` | Core engine: grid detection, cell classification, Gemini OCR chain, confidence demotion |
| `ncc_matcher.py` | Icon-ID matcher: coarse-to-fine alpha-masked template matching (`TM_CCOEFF_NORMED`) against the SchaleDB sprites — 250/250 on marked ground truth, no ML deps |
| `download_icons.py` | Fetches icon sprites from SchaleDB into `cache/icons/` — the matcher's reference data |
| `test_gemini.py` | Smoke test for the Gemini quantity path |
| `batch_test.py` | Dev runner: parses every `assets/Screenshot*.png` and reports sort-order violations |
| `debug_full_parse.py` | Pretty-print the full parse of one screenshot |
| `debug_detect_ids.py` | Detection-only dump (no Gemini) in ground-truth marking format, with NCC score/margin diag |
| `debug_c2f_eval.py` | Scores the matcher against a hand-marked ground-truth txt |
| `debug_ncc_eval.py` / `debug_ncc_probe.py` | Brute-force matcher eval / sprite scale calibration (for new devices/resolutions) |
| `gunicorn.conf.py` | `127.0.0.1:5001`, 1 worker, 300s timeout |
| `requirements.txt` | Flask, OpenCV, NumPy, SciPy, Pillow, `google-genai` |
| `cache/icon_index_*.json` + `cache/icons/` | SchaleDB sprite index + images (gitignored, fetched by `download_icons.py`) |
| `assets/` | Reference screenshots + optional grid-detection templates (see below) |

## Environment

| Variable | Required? | Notes |
|---|---|---|
| `GEMINI_API_KEY` | yes | Loaded from the repo-root `.env`. Without it, quantities default to 0 with red confidence for manual entry — icon detection still works. Get one from Google AI Studio. |

## Local development

```bash
cd services/inventory_parser
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# First-time: fetch icon sprites (the matcher's reference data, ~1 min)
python download_icons.py

# Add the Gemini key to the repo-root .env
echo "GEMINI_API_KEY=your_key_here" > ../../.env

# Run dev server
python app.py
# API at http://localhost:5001
```

Smoke test a screenshot:

```bash
curl -X POST http://localhost:5001/inventory/parse \
  -F "image=@screenshot.png" \
  -F "inventoryType=items"
```

Pretty-printed debug:

```bash
python debug_full_parse.py screenshot.png items
```

## Assets

`assets/` holds the reference data the parser is calibrated against:

- `Screenshot*.png` — representative Blue Archive inventory screenshots used by `batch_test.py` (sort-order regression check).
- `list_header.png`, `item_cannot_use.png` — optional template anchors. Drop them in to override the ratio-based grid bounds; when absent, `_compute_grid_bounds` falls back to fixed ratios that are accurate enough for FHD screenshots.

## Deployment

Production setup, systemd unit, nginx + rate limits, and the update flow live in [`../../deploy/README.md`](../../deploy/README.md). Key bits:

- `eridu-parser.service` — the gunicorn process
- `cache/icons/` + `cache/icon_index_*.json` — re-fetched by `download_icons.py` when new game items ship (no rebuild step; the matcher reads the sprites directly)
