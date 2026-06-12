"""Validate the coarse-to-fine NCC matcher against the marked ground truth.

Checks, per cell:
  - whether the true item survives the half-res coarse stage (top-K coverage)
  - whether the refined top-1 equals the truth (end accuracy)
  - score/margin distributions (informs the empty-slot accept threshold)
and reports wall time per screenshot.

Usage (from services/inventory_parser/ with .venv active):
    python debug_c2f_eval.py
"""
import os
import re
import time

import numpy as np

import ncc_matcher
from pipeline import _extract_grid

GT_PATH = '/Users/frsardhf/Downloads/Images/detected_ids.txt'
DIRS = {'items': '/Users/frsardhf/Downloads/Images/Items',
        'equipment': '/Users/frsardhf/Downloads/Images/Equipment'}


def parse_gt(path):
    gt, files, name, inv = {}, [], None, None
    row = 0
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^#\s+(\S+)\s+\[(\w+)\]', line)
            if m:
                name, inv = m.group(1), m.group(2)
                files.append((name, inv))
                row = 0
                continue
            for col, tok in enumerate(line.split()):
                if tok != '-':
                    gt[(name, row, col)] = tok
            row += 1
    return gt, files


def main() -> None:
    gt, files = parse_gt(GT_PATH)
    ncc_matcher.warm()

    hits = miss = 0
    pre_ranks, scores, margins = [], [], []
    errors = []
    match_time = 0.0

    for name, inv in files:
        with open(os.path.join(DIRS[inv], name), 'rb') as fh:
            grid, rows, cols, cell_w, rb = _extract_grid(fh.read(), inv)
        bank = ncc_matcher.get_bank(inv, cell_w)

        file_match = 0.0
        for r in range(rows):
            for c in range(cols):
                true_id = gt.get((name, r, c))
                if true_id is None:
                    continue
                win = ncc_matcher.cell_window(grid, cell_w, rb, r, c)

                # Production-cost path, timed separately from the coverage
                # bookkeeping below.
                t0 = time.time()
                pred_id, score, margin = ncc_matcher.match_window(win, bank)
                file_match += time.time() - t0

                pre = ncc_matcher.prefilter_scores(win, bank)
                rank = next((i for i, (_, iid) in enumerate(pre)
                             if iid == true_id), 999)
                pre_ranks.append(rank)

                scores.append(score)
                margins.append(margin)
                if pred_id == true_id:
                    hits += 1
                else:
                    miss += 1
                    errors.append((name, r, c, true_id, pred_id, score, rank))
        match_time += file_match
        print(f'[c2f] {name}: match={file_match:.1f}s')

    ranks = np.array(pre_ranks)
    scores = np.array(scores)
    margins = np.array(margins)
    k = ncc_matcher.PREFILTER_TOP_K
    print(f'\naccuracy: {hits}/{hits + miss} ({hits / (hits + miss):.1%})')
    print(f'prefilter coverage: top-{k}={np.mean(ranks < k):.1%}  '
          f'worst rank={ranks.max()}  p99={np.percentile(ranks, 99):.0f}')
    print(f'pure match time: {match_time:.1f}s total '
          f'({match_time / (hits + miss) * 1000:.0f}ms/cell)')
    print(f'refined score: min={scores.min():.4f}  p5={np.percentile(scores, 5):.4f}  '
          f'median={np.median(scores):.4f}')
    print(f'margin: min={margins.min():.4f}  p5={np.percentile(margins, 5):.4f}  '
          f'median={np.median(margins):.4f}')
    for name, r, c, true_id, pred_id, score, rank in errors:
        print(f'  ERR {name} r{r}c{c}: true={true_id} pred={pred_id} '
              f'score={score:.4f} coarse_rank_of_true={rank}')


if __name__ == '__main__':
    main()
