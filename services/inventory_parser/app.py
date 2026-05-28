import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

# Load .env from the repo root (two dirs up from this file).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_REPO_ROOT, '.env'))

from pipeline import parse_inventory_batch, warm_icon_db

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB
CORS(app, origins=['https://eriduops.com', 'http://localhost:5173'])

MAX_SCREENSHOTS = 3  # multi-grid Gemini batching is reliable up to 3 (validated)

try:
    warm_icon_db()
except Exception as exc:
    import traceback
    print(f'[inventory_parser] Warmup failed: {exc}')
    traceback.print_exc()


@app.post('/inventory/parse')
def parse_inventory_endpoint():
    inventory_type = request.form.get('inventoryType')
    if inventory_type not in ('items', 'equipment'):
        return jsonify({'error': 'inventoryType must be items or equipment'}), 400

    # Accept up to MAX_SCREENSHOTS under 'images'; fall back to single 'image'
    # for backward compatibility with older frontends.
    files = request.files.getlist('images')
    if not files and 'image' in request.files:
        files = [request.files['image']]
    if not files:
        return jsonify({'error': 'at least one image is required'}), 400
    if len(files) > MAX_SCREENSHOTS:
        return jsonify({'error': f'at most {MAX_SCREENSHOTS} screenshots per request'}), 400

    images = [b for b in (f.read() for f in files) if b]
    if not images:
        return jsonify({'error': 'empty image'}), 400

    results = parse_inventory_batch(images, inventory_type)
    return jsonify({'results': results})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
