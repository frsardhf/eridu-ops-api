# eridu-ops-api

The backend for [eridu-ops](https://eriduops.com) at `api.eriduops.com` — two services behind one nginx: an **Inventory Scanner** that reads **Blue Archive** inventory screenshots into structured JSON of items + quantities, and the **Bond 100 Hall** (community bond-100 counts, sourced from arona.icu). Most of this README covers the scanner; the Hall is summarized at the [end](#bond-100-hall).

## What it does

1. Accept **1–3** PNG screenshots + `inventoryType` (`items` or `equipment`) via `POST /inventory/parse`
2. Parse out up to a 5×4 or 5×5 grid of item cells from each screenshot
3. For each cell: identify the item, detect its rarity, and read the quantity number
4. Apply correction passes to fix misidentified cells
5. Return `{ results: [{ row, col, itemId, rarity, quantity, confidence }] }`, with each screenshot's rows offset by its position so the frontend groups them (#1, #2, #3)

## Flow

```
1–3 screenshots (same inventory type)
  → Per screenshot: OpenCV decode + crop right-half → grid bounds → row boundaries
  → Gemini model chain: ONE batched API call across all grids → {(grid,row,col): qty}
        (tries 3.1-flash-lite → 2.5-flash → 3.5-flash → 2.5-flash-lite in order;
         on batch failure, retries each grid individually before degrading)
  → Per grid, per-cell loop (5 cols × 4–5 rows):
      ① Rarity: HSV color of cell border ring (green=R, orange=SR, pink=SSR)
      ② Icon ID: CLIP ViT-B/32 embedding → cosine similarity vs ~90 precomputed embeddings
             + shape re-ranking (circularity/aspect ratio) + rarity re-ranking
      ③ Quantity: from the Gemini lookup; if the whole chain is exhausted → 0 + red confidence (manual entry)
  → Correction passes (per grid):
      ① Sort-order constraint (enforce monotonic IDs per sort direction)
      ② Range outlier fix (ID 3500 can't appear between IDs 200 and 210)
      ③ Favor item recovery (SSR gifts at ID 5100+ are rare, re-scan for them)
      ④ Group sort enforcement (within contiguous same-category runs)
  → Offset each grid's rows by grid_index × rows_per_screenshot → JSON response
```

## Tech stack

| Layer | Tech |
|---|---|
| Web server | Flask + Flask-CORS, Gunicorn, Nginx |
| Image processing | OpenCV, NumPy |
| Icon matching | CLIP ViT-B/32 (Hugging Face Transformers + PyTorch), precomputed `.npy` embeddings |
| Quantity OCR | Gemini model chain (`google-genai`): one batched call reads all cells across 1–3 screenshots at once (~6s) |
| Math helpers | SciPy (`uniform_filter1d` for row detection) |
| Icon data | Fetched from SchaleDB (`download_icons.py`), embeddings precomputed offline by `embed.py` |
| Deployment | Ubuntu 24.04 VPS, Let's Encrypt SSL, systemd service |

### OCR architecture decision

Quantity reading is now handled entirely by a **chain of free-tier Gemini models** — no self-hosted OCR. How we got here:

- **EasyOCR / RapidOCR** (early): traditional detect-then-recognize. Decent accuracy but slow on CPU (~30s) and a heavy dependency. OCR consolidated to Gemini; the Paddle/EasyOCR fork (`eridu-api-paddle`) is retired — classic OCR's strengths (offline/on-prem, pixel-exact boxes, high-volume cost) don't apply to game-UI screenshots, where hosted VLMs read stylized fonts better.
- **Florence-2** (removed): Microsoft's VLM. Accurate (~99%) but **~4 min per screenshot on the VPS CPU** — too slow to be a usable fallback. The only way to make it fast is a GPU/Mac Mini, at which point a hosted API is simpler and better.
- **Gemini chain** (current): one API call sends the whole grid; the model returns every quantity in ~6s at ~100% digit accuracy. Models are tried in order — `3.1-flash-lite` (500 RPD) → `2.5-flash` → `3.5-flash` → `2.5-flash-lite`. Free-tier rate limits are **per-model**, so chaining sums to ~544 requests/day. Each model has its own daily counter (reset at Pacific midnight, matching Google's RPD boundary) and advances to the next on a 429.

**Direction:** Gemini-only, no offline fallback. When the entire chain is exhausted, cells return `quantity=0` with low (red) confidence so the user types the numbers manually — icon detection (local CLIP) still resolves the correct items. A fast result with a few blanks beats a 4-minute Florence wait. This also drops `timm`/`einops` and the heavy Florence weights from the deploy.

### Batching (1–3 screenshots per request)

Free-tier RPD is **per-model**, so a single screenshot still costs one request from one model. To stretch that budget, the frontend uploads up to **3** screenshots in one request and the backend reads them in a **single batched Gemini call** (all grids in one `generate_content`, each cell tagged with its grid index) — 3 screenshots → 1 request instead of 3.

The cap is **3, not higher**: a gating test showed multi-grid accuracy is 100% at N≤3 but drops to ~82% at N=5, where the model starts scrambling the *middle* grid's rows. So 3 is the validated ceiling (and conveniently matches the original EasyOCR limit). On any batch failure, each grid is retried as an individual call before degrading. Frontend downscales every screenshot to FHD width before upload, so 3 shots stay well under the 10 MB request cap and the parser stays at its tuned resolution.

## Key files

| File | Purpose |
|---|---|
| `app.py` | Flask entrypoint — single route, CORS, model warmup |
| `services/inventory_parser/pipeline.py` | ~1800-line core engine: all detection, OCR, and correction logic |
| `services/inventory_parser/embed.py` | Offline tool: generates CLIP embeddings from icon PNGs, saves `.npy` cache |
| `services/inventory_parser/embed_from_screenshots.py` | Quality booster: blends real in-game icon crops (from `assets/` screenshots whose pipeline output is monotonic = trusted) into the sprite embeddings |
| `services/inventory_parser/download_icons.py` | Fetches icon images from SchaleDB |
| `services/inventory_parser/test_gemini.py` | Smoke test for the Gemini fast path (single cell + full screenshot) |
| `services/inventory_parser/batch_test.py` | Dev runner: parses every `assets/Screenshot*.png` and reports sort-order violations |
| `cache/icon_embeddings_*.npy` | Precomputed 512-dim vectors — never committed to git, regenerated on deploy |
| `deploy/setup.sh` | One-shot VPS installer (idempotent) |
| `deploy/eridu-parser.service` | Inventory parser systemd unit |
| `deploy/eridu-bond100.service` | Bond 100 Hall systemd unit (paired with `eridu-bond100-sync.{service,timer}` for the daily refresh) |
| `deploy/eridu-api.nginx.conf` | Nginx reverse proxy + rate limiting config (both services) |

## Local development

```bash
cd services/inventory_parser
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# First-time: fetch icons and build CLIP embeddings (~10 min, downloads ~600 MB)
python download_icons.py
python embed.py items
python embed.py equipment

# Gemini API key is required for quantity OCR (without it, quantities default to 0
# for manual entry — icon detection still works). Get one from Google AI Studio.
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

Update after a code push (see [`deploy/README.md`](deploy/README.md) for the full notes, including the certbot caveat when reinstalling the nginx conf):

```bash
ssh root@<vps-ip>
cd /opt/eridu-ops-api && git pull
chown -R eridu:eridu /opt/eridu-ops-api          # git pull as root leaves new files root-owned
systemctl restart eridu-parser eridu-bond100     # bond100 SQLite cache in var/ is untouched
```

## API reference

### `POST /inventory/parse`

| Field | Type | Description |
|---|---|---|
| `images` | file (×1–3) | PNG/JPG/WebP screenshots, all the same inventory type (10 MB total request cap). Repeat the field for multiple. |
| `image` | file | Legacy single-file field — still accepted for backward compatibility. |
| `inventoryType` | string | `"items"` or `"equipment"` |

**Response**

Rows are global across the batch: screenshot #1 → rows `0..3` (items) / `0..4` (equipment), screenshot #2 → the next block, and so on. The frontend groups by `floor(row / rows_per_screenshot)`.

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

## Bond 100 Hall

A second service (gunicorn `:5002`, same nginx) backing the `/hall` page on the frontend — a community wall of how many players have reached **Bond 100** with each student, by Global server.

**Bridge model:** arona.icu is the single source of truth. A daily systemd timer runs `sync_arona.py`, which pulls arona's `rank_by_max_favor_user_info` endpoint in one call, aggregates the five Global servers into per-student counts + name lists, and caches them as JSON blobs in a small SQLite cache. The API just serves that cache — no scraping, no merge logic, no per-entry storage.

| Endpoint | Purpose |
|---|---|
| `GET /bond100/summary` | wall counts per student (+ by-server, snapshot date) |
| `GET /bond100/students/<id>/entries` | player names at bond 100 for one student |
| `POST /bond100/submissions` | "add me" — triggers an arona `/refresh` for the given friend code (rate-limited); the player appears in the next sync |

Removal is handled on arona's side (the frontend links to arona's guidelines). Friend codes are never stored — only a salted hash, for submission rate-limiting (per-code 6h cooldown + global daily cap). The arona API token (`ARONA_TOKEN`) and the daily sync are covered in [`deploy/README.md`](deploy/README.md).

Key files: `services/bond100/{sync_arona,app,repository,arona_client,db}.py`, `schema.sql`.
