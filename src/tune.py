"""Hyperparameter search on the validation season (2024/25).

Usage:
    python -m src.tune                  # full search (long)
    python -m src.tune --rerank-only    # only step 3, with the current best_config.json

1. LightGCN grid (history window, layers, L2) with early stopping on the
   validation warm NDCG@10.
2. With the best LightGCN, choice of warm_mix and cold_slots on the
   validation NDCG@10 (all targets).
3. Choice of the re-ranking weights (resale, age, mobility) on the
   validation NDCG@10, with the budget filter on.
Nothing from the test season is used. Writes eval/results/tuning.csv,
eval/results/tuning_hybrid.csv, eval/results/tuning_rerank.csv,
eval/results/best_config.json, eval/results/best_rerank.json and
eval/results/val_training_log.csv.
"""

import itertools
import json
import sys
import time
from dataclasses import replace

import pandas as pd

from eval.harness import build_setup, evaluate
from src.config import RESULTS_DIR, TEST_START, VAL_START
from src.features import _raw, load_or_build
from src.split import load_interactions
from src.train import Config, config_dict, fit, ranking

RERANK_GRID = {
    "resale_weight": [0.0, 0.5, 1.0],
    "age_weight": [0.0, 0.25, 0.5, 1.0],
    "mobility_weight": [0.0, 0.5],
}

GRID = {
    "years": [5, 10, None],
    "layers": [1, 2, 3],
    "reg": [1e-4, 1e-3],
}


def tune_rerank(fitted, setup, feats, warm_mix: float) -> dict:
    """Grid over the re-ranking weights on the validation season (no retraining)."""
    from eval.run_experiments import reranked

    fitted.config = replace(fitted.config, warm_mix=warm_mix, cold_slots=0)
    rows = []
    for weights in itertools.product(*RERANK_GRID.values()):
        w = dict(zip(RERANK_GRID, weights))
        m = evaluate(setup, None, ranked=reranked(fitted, setup, feats, **w))
        rows.append({**w, **m})
        print(f"{w}: ndcg@10={m['ndcg@10']:.4f} age@10={m['age@10']:.1f} "
              f"value_change@10={m['value_change@10']:+.3f}", flush=True)
    rows = pd.DataFrame(rows)
    rows.to_csv(RESULTS_DIR / "tuning_rerank.csv", index=False)
    top = rows.sort_values("ndcg@10", ascending=False).iloc[0]
    best = {k: float(top[k]) for k in RERANK_GRID}
    with open(RESULTS_DIR / "best_rerank.json", "w") as f:
        json.dump(best, f, indent=2)
    print("best re-ranking:", best)
    return best


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inter = load_interactions()
    feats = load_or_build(VAL_START)
    setup = build_setup(inter, feats, VAL_START, TEST_START, _raw("player_valuations"))

    if "--rerank-only" in argv:
        # keep best_config.json, retrain it once on the validation split, tune the re-ranking only
        with open(RESULTS_DIR / "best_config.json") as f:
            cfg = Config(**json.load(f))
        fitted = fit(setup, feats, cfg, monitor=False, verbose=False)
        tune_rerank(fitted, setup, feats, cfg.warm_mix)
        return 0

    rows, best, best_fit = [], -1.0, None
    for years, layers, reg in itertools.product(*GRID.values()):
        cfg = Config(years=years, layers=layers, reg=reg, epochs=150)
        t0 = time.time()
        fitted = fit(setup, feats, cfg, monitor=True, verbose=False)
        m = evaluate(setup, None, ranked=ranking(fitted, setup, "lightgcn"))
        row = {**config_dict(cfg), "best_epoch": fitted.best_epoch, **m, "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"years={years} layers={layers} reg={reg}: ndcg@10={m['ndcg@10']:.4f} "
              f"recall@20={m['recall@20']:.4f} epoch={fitted.best_epoch}", flush=True)
        if m["ndcg@10"] > best:
            best, best_fit = m["ndcg@10"], fitted
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tuning.csv", index=False)

    # warm_mix and cold_slots on the best LightGCN (no retraining needed)
    second = []
    for warm_mix, cold_slots in itertools.product([0.0, 0.25, 0.5], [0, 1, 2, 3]):
        best_fit.config = replace(best_fit.config, warm_mix=warm_mix, cold_slots=cold_slots)
        m = evaluate(setup, None, ranked=ranking(best_fit, setup, "hybrid"))
        second.append({"warm_mix": warm_mix, "cold_slots": cold_slots, **m})
        print(f"warm_mix={warm_mix} cold_slots={cold_slots}: ndcg@10={m['ndcg@10']:.4f} "
              f"cold_ndcg@10={m['cold_ndcg@10']:.4f}", flush=True)
    second = pd.DataFrame(second)
    second.to_csv(RESULTS_DIR / "tuning_hybrid.csv", index=False)
    top = second.sort_values("ndcg@10", ascending=False).iloc[0]

    final = replace(best_fit.config, warm_mix=float(top["warm_mix"]), cold_slots=int(top["cold_slots"]),
                    epochs=int(best_fit.best_epoch))
    with open(RESULTS_DIR / "best_config.json", "w") as f:
        json.dump(config_dict(final), f, indent=2)
    pd.DataFrame(best_fit.log).to_csv(RESULTS_DIR / "val_training_log.csv", index=False)
    print("best config:", config_dict(final))

    tune_rerank(best_fit, setup, feats, final.warm_mix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
