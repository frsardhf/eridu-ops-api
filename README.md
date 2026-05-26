# eridu-ops-api

A single-endpoint REST API that reads **Blue Archive** inventory screenshots and returns a structured JSON list of detected items with quantities. Deployed at `api.eriduops.com` and called by the [eridu-ops](https://eriduops.com) frontend's Inventory Scanner feature.

## What it does

1. Accept a PNG screenshot + `inventoryType` (`items` or `equipment`) via `POST /inventory/parse`
2. Parse out up to a 5×4 or 5×5 grid of item cells from the screenshot
3. For each cell: identify the item, detect its rarity, and read the quantity number
4. Apply correction passes to fix misidentified cells
5. Return `{ results: [{ row, col, itemId, rarity, quantity, confidence }] }`

## Flow

```
Screenshot PNG
  → OpenCV: decode + crop right-half (inventory panel side)
  → Grid detection: template-match the "List" header + footer to find grid bounds
  → Row boundary detection: HSV saturation analysis to find actual row gaps
  → Gemini Flash: one API call on the full grid image → {(row,col): qty} lookup dict
        (falls back to per-cell Florence-2 if key missing or Gemini unavailable)
  → Per-cell loop (5 cols × 4–5 rows):
      ① Rarity: HSV color of cell border ring (green=R, orange=SR, pink=SSR)
      ② Icon ID: CLIP ViT-B/32 embedding → cosine similarity vs ~90 precomputed embeddings
             + shape re-ranking (circularity/aspect ratio) + rarity re-ranking
      ③ Quantity: Gemini lookup (fast path) or Florence-2 OCR on upscaled 3× crop (fallback)
  → Correction passes:
      ① Sort-order constraint (enforce monotonic IDs per sort direction)
      ② Range outlier fix (ID 3500 can't appear between IDs 200 and 210)
      ③ Favor item recovery (SSR gifts at ID 5100+ are rare, re-scan for them)
      ④ Group sort enforcement (within contiguous same-category runs)
  → JSON response
```

## Tech stack

| Layer | Tech |
|---|---|
| Web server | Flask + Flask-CORS, Gunicorn, Nginx |
| Image processing | OpenCV, NumPy |
| Icon matching | CLIP ViT-B/32 (Hugging Face Transformers + PyTorch), precomputed `.npy` embeddings |
| OCR — fast path | Gemini Flash 2.5 (`google-genai`): one call per screenshot, reads all 20 cells at once (~6s) |
| OCR — fallback | Florence-2 (`microsoft/Florence-2-base`): per-cell VLM OCR, used when Gemini key is absent, daily quota is exhausted, or any API error occurs (~240s on CPU) |
| Math helpers | SciPy (`uniform_filter1d` for row detection) |
| Icon data | Fetched from SchaleDB (`download_icons.py`), embeddings precomputed offline by `embed.py` |
| Deployment | Ubuntu 24.04 VPS, Let's Encrypt SSL, systemd service |

### OCR architecture decision

Quantity reading went through three candidates before landing on the current hybrid:

- **EasyOCR** (original): traditional detect-then-recognize pipeline. 98% accuracy on English digits in testing, but slow (~30s per screenshot on CPU due to CRAFT text detector overhead on small crops) and adds a heavy dependency (~1 GB).
- **RapidOCR / PaddleOCR-ONNX** (evaluated): DBNet detector, faster inference, but 90% accuracy on English and 43% on Japanese quantity labels — worse than EasyOCR on this task. Ruled out.
- **Florence-2** (current fallback): Microsoft's vision-language model. Handles the OCR task well and shares the PyTorch runtime already needed for CLIP, so no extra dependency. ~240s on CPU, ~20s on Apple Silicon MPS.
- **Gemini Flash 2.5** (primary fast path): sends the entire grid image to Google's API once per screenshot, returns all 20 quantities in ~6s. Free tier is sufficient for typical usage (~1500 req/day). An adaptive daily cap auto-lowers on the first 429 and resets at UTC midnight.

The hybrid keeps Florence-2 as a fully self-contained fallback so the service works without any API key — useful for local dev and as a safety net when the Gemini quota runs out.

## Key files

| File | Purpose |
|---|---|
| `app.py` | Flask entrypoint — single route, CORS, model warmup |
| `services/inventory_parser/pipeline.py` | 1500-line core engine: all detection, OCR, and correction logic |
| `services/inventory_parser/embed.py` | Offline tool: generates CLIP embeddings from icon PNGs, saves `.npy` cache |
| `services/inventory_parser/download_icons.py` | Fetches icon images from SchaleDB |
| `services/inventory_parser/test_gemini.py` | Smoke test for the Gemini fast path (single cell + full screenshot) |
| `cache/icon_embeddings_*.npy` | Precomputed 512-dim vectors — never committed to git, regenerated on deploy |
| `deploy/setup.sh` | One-shot VPS installer (idempotent) |
| `deploy/eridu-parser.service` | systemd unit file |
| `deploy/eridu-api.nginx.conf` | Nginx reverse proxy + rate limiting config |

## Local development

```bash
cd services/inventory_parser
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# First-time: fetch icons and build CLIP embeddings (~10 min, downloads ~600 MB)
python download_icons.py
python embed.py items
python embed.py equipment

# Optional: add Gemini key for the fast OCR path (falls back to Florence-2 without it)
echo "GEMINI_API_KEY=your_key_here" > ../../.env

# Run dev server
python app.py
# API available at http://localhost:5001
```

Test a screenshot:

```bash
curl -X POST http://localhost:5001/inventory/parse \
  -F "image=@screenshot.png" \
  -F "inventoryType=items"
```

Debug helper (pretty-prints results with grid position and confidence):

```bash
python debug_full_parse.py screenshot.png items
```

## Deployment

See [`deploy/README.md`](deploy/README.md) for the full VPS setup guide.

One-shot install on a fresh Ubuntu 24.04 VPS:

```bash
curl -fsSL https://raw.githubusercontent.com/frsardhf/eridu-ops-api/master/deploy/setup.sh | bash
```

Update after a code push:

```bash
ssh root@<vps-ip>
cd /opt/eridu-ops-api && git pull
systemctl restart eridu-parser
```

## API reference

### `POST /inventory/parse`

| Field | Type | Description |
|---|---|---|
| `image` | file | PNG/JPG/WebP screenshot (max 10 MB) |
| `inventoryType` | string | `"items"` or `"equipment"` |

**Response**

```json
{
  "results": [
    {
      "row": 0,
      "col": 0,
      "itemId": "2001",
      "rarity": "SR",
      "quantity": 150,
      "confidence": 0.823
    }
  ]
}
```

`confidence` is the CLIP cosine similarity score (0–1). Values below 0.8 are flagged as low confidence in the frontend.
