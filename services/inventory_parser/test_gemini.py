"""
Smoke test for Gemini Flash quantity OCR — single cell, then full screenshot.

Run from anywhere:
    /Users/frsardhf/Projects/eridu-ops-api/services/inventory_parser/.venv/bin/python \
        /Users/frsardhf/Projects/eridu-ops-api/services/inventory_parser/test_gemini.py
"""
import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from google import genai

# Load .env from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(REPO_ROOT, '.env'))

API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    print('ERROR: GEMINI_API_KEY not in env')
    sys.exit(1)

# Pick a model. gemini-2.5-flash is the current cheapest fast model with vision.
# Fall back to gemini-2.0-flash if 2.5 isn't available on your free tier yet.
MODEL = 'gemini-2.5-flash'

TEST_IMG = '/Users/frsardhf/Downloads/Images/Screenshot_2026-05-01_203906.png'

client = genai.Client(api_key=API_KEY)


def test_full_screenshot():
    """Send the whole inventory screenshot and ask for all 20 quantities."""
    print(f'\n=== Test 1: Full screenshot → JSON of all 20 quantities ===')
    img = Image.open(TEST_IMG)
    prompt = (
        "This is a Blue Archive game inventory screen. The right half contains "
        "a grid of 4 rows × 5 columns of item cells. Each cell has an icon and "
        "a quantity number prefixed with '×' at the bottom-right.\n\n"
        "Read the quantity number for every cell. Return ONLY a JSON array, "
        "no markdown, no explanation:\n"
        "[{\"row\":0,\"col\":0,\"qty\":1197},{\"row\":0,\"col\":1,\"qty\":607},...]"
    )
    t0 = time.time()
    response = client.models.generate_content(
        model=MODEL,
        contents=[img, prompt],
    )
    elapsed = time.time() - t0
    print(f'  elapsed: {elapsed:.2f}s')
    print(f'  response:')
    print(response.text[:2000])


def test_single_cell():
    """Crop a single cell and ask for just its quantity."""
    print(f'\n=== Test 2: Single cell crop → integer ===')
    image = cv2.imread(TEST_IMG, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    # rough crop of r0c0 in the right half
    cell = image[238:380, w // 2 + 60: w // 2 + 240]
    pil = Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
    prompt = (
        "Read ONLY the quantity number at the bottom-right of this game "
        "inventory cell. The number is prefixed with '×'. Return just the "
        "integer, no other text. If you can't read it, return 0."
    )
    t0 = time.time()
    response = client.models.generate_content(model=MODEL, contents=[pil, prompt])
    elapsed = time.time() - t0
    print(f'  elapsed: {elapsed:.2f}s')
    print(f'  response: {response.text.strip()!r}')


if __name__ == '__main__':
    print(f'Model: {MODEL}')
    print(f'Image: {TEST_IMG}')
    test_single_cell()
    test_full_screenshot()
