"""Hyperparameter search on the validation season (2024/25).

Usage:
    python -m src.tune                  # full search (long)
    python -m src.tune --rerank-only    # only step 3, retrains best_config.json with 3 seeds

1. LightGCN grid (history window, layers, L2) with early stopping on the
   validation warm NDCG@10.
2. With the best LightGCN, choice of warm_mix and cold_slots on the
   validation NDCG@10 (all targets).
3. Choice of the re-ranking weights (resale, age, mobility) on the
   validation season, budget filter on, 3 seeds: the strongest age weight
   whose NDCG@10 stays within one seed std of the best (see select_rerank).
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
from src.rerank import select_rerank
from src.train import Config, config_dict, fit, ranking

RERANK_GRID = {
    "resale_weight": [0.0, 0.5, 1.0],
    "age_weight": [0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
    "mobility_weight": [0.0, 0.5],
}

GRID = {
    "years": [5, 10, None],
    "layers": [1, 2, 3],
    "reg": [1e-4, 1e-3],
}


RERANK_SEEDS = [0, 1, 2]


def tune_rerank(setup, feats, cfg: Config) -> dict:
    """Grid over the re-ranking weights on the validation season, mean and std over seeds."""
    from eval.run_experiments import reranked

    fits = []
    for seed in RERANK_SEEDS:
        fi = fit(setup, feats, replace(cfg, seed=seed), monitor=False, verbose=False)
        fi.config = replace(fi.config, cold_slots=0)
        fits.append(fi)
    rows = []
    for weights in itertools.product(*RERANK_GRID.values()):
        w = dict(zip(RERANK_GRID, weights))
        runs = pd.DataFrame([evaluate(setup, None, ranked=reranked(fi, setup, feats, **w)) for fi in fits])
        row = {**w}
        for m in ["ndcg@10", "recall@20", "age@10", "value_change@10", "within_budget@10", "coverage@10"]:
            row[m] = runs[m].mean()
            row[m + "_std"] = runs[m].std(ddof=0)
        rows.append(row)
        print(f"{w}: ndcg@10={row['ndcg@10']:.4f} ± {row['ndcg@10_std']:.4f} age@10={row['age@10']:.1f} "
              f"value_change@10={row['value_change@10']:+.3f}", flush=True)
    table = pd.DataFrame(rows)
    top = select_rerank(table)
    table["selected"] = (table[list(RERANK_GRID)] == top[list(RERANK_GRID)]).all(axis=1)
    table.to_csv(RESULTS_DIR / "tuning_rerank.csv", index=False)
    best = {k: float(top[k]) for k in RERANK_GRID}
    with open(RESULTS_DIR / "best_rerank.json", "w") as f:
        json.dump(best, f, indent=2)
    print(f"selected re-ranking: {best} (ndcg@10={top['ndcg@10']:.4f}, age@10={top['age@10']:.1f}, "
          f"value_change@10={top['value_change@10']:+.3f})")
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
        tune_rerank(setup, feats, cfg)
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

    tune_rerank(setup, feats, final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
