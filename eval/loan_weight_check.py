"""Should loans count less than transfers? Validation season only.

Usage:
    python -m eval.loan_weight_check

Trains the tuned LightGCN on the history before VAL_START with loans drawn
as positives with weight 1.0 (as now) or 0.5, 3 seeds each, and compares
NDCG@10 and Recall@20 on season 2024/25. Writes eval/results/loan_weight.md.
"""

import json
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from eval.harness import build_setup, evaluate
from src.config import RESULTS_DIR, TEST_START, VAL_START
from src.features import _raw, load_or_build
from src.split import load_interactions
from src.train import Config, fit, ranking

WEIGHTS = [1.0, 0.5]
SEEDS = [0, 1, 2]


def main() -> int:
    with open(RESULTS_DIR / "best_config.json") as f:
        cfg = Config(**json.load(f))
    inter = load_interactions()
    feats = load_or_build(VAL_START)
    setup = build_setup(inter, feats, VAL_START, TEST_START, _raw("player_valuations"))
    rows = []
    for w in WEIGHTS:
        for s in SEEDS:
            fi = fit(setup, feats, replace(cfg, loan_weight=w, seed=s), monitor=False, verbose=False)
            m = evaluate(setup, None, ranked=ranking(fi, setup, "lightgcn"))
            rows.append({"loan_weight": w, "seed": s, "ndcg@10": m["ndcg@10"], "recall@20": m["recall@20"]})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby("loan_weight")[["ndcg@10", "recall@20"]].agg(["mean", "std"])
    lines = ["# Loan weight (validation season 2024/25, 3 seeds)", "",
             "| Loan weight | NDCG@10 | Recall@20 |", "| --- | --- | --- |"]
    for w in WEIGHTS:
        r = agg.loc[w]
        lines.append(f"| {w} | {r[('ndcg@10', 'mean')]:.4f} ± {r[('ndcg@10', 'std')]:.4f} | "
                     f"{r[('recall@20', 'mean')]:.4f} ± {r[('recall@20', 'std')]:.4f} |")
    (RESULTS_DIR / "loan_weight.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
