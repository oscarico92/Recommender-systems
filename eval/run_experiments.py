"""Final evaluation on the test season (2025/26), with ablation.

Usage:
    python -m eval.run_experiments            # needs eval/results/best_config.json (python -m src.tune)

The model is trained on the interactions before TEST_START with the
hyperparameters chosen on the validation season, so the test targets are
never used for training or tuning. Each model is trained with 3 seeds.

Writes eval/results/test_results.{csv,md}, eval/results/case_studies.md and
docs/figures/loss_curves.png.
"""

import json
import sys
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eval.baselines import (  # noqa: E402
    market_value_scores,
    most_popular_scores,
    random_scores,
    recent_popular_scores,
)
from eval.harness import EvalSetup, build_setup, evaluate, mask_scores, target_reference  # noqa: E402
from src.config import FIGURES_DIR, RESULTS_DIR, TEST_START  # noqa: E402
from src.features import _raw, load_or_build  # noqa: E402
from src.rerank import merge_with_cold_slots, rerank_group  # noqa: E402
from src.split import load_interactions  # noqa: E402
from src.train import Config, fit, item_embeddings, ranking  # noqa: E402

SEEDS = [0, 1, 2]
METRICS = ["ndcg@10", "recall@20", "cold_ndcg@10", "warm_ndcg@10", "coverage@10",
           "within_budget@10", "value_change@10", "age@10"]
CASE_CLUBS = {985: "Manchester United", 294: "Benfica", 371: "Celtic"}


def player_moves(setup: EvalSetup) -> np.ndarray:
    """Number of moves of each catalogue player between the selected clubs before the cutoff."""
    counts = setup.history.groupby("player_id").size()
    return counts.reindex(setup.item_ids).fillna(0).to_numpy(float)


def reranked(fitted, setup: EvalSetup, feats: pd.DataFrame, resale_weight: float, use_budget: bool = True,
             age_weight: float = 0.0, mobility_weight: float = 0.0):
    value = feats["value_eur"].to_numpy(float)
    trend = feats["value_trend"].to_numpy(float)
    extra = dict(age=feats["age"].to_numpy(float), age_weight=age_weight,
                 moves=player_moves(setup), mobility_weight=mobility_weight)
    s_warm = fitted.club_emb @ item_embeddings(fitted, "hybrid").T
    s_cold = fitted.club_emb @ fitted.item_emb_enc.T
    s_warm[:, ~fitted.in_graph] = -np.inf
    s_cold[:, fitted.in_graph] = -np.inf
    s_warm, s_cold = mask_scores(setup, s_warm), mask_scores(setup, s_cold)
    warm_rank, cold_rank = [], []
    for i in range(len(setup.club_ids)):
        budget = setup.club_budget[i] if use_budget else None
        warm_rank.append(rerank_group(s_warm[i], value, trend, budget, resale_weight, 20, **extra))
        cold_rank.append(rerank_group(s_cold[i], value, trend, budget, resale_weight, 20, **extra))
    return merge_with_cold_slots(np.array(warm_rank), np.array(cold_rank), fitted.config.cold_slots)


def _tuned_rerank() -> dict:
    """Re-ranking weights chosen on the validation season by src/tune.py (defaults if absent)."""
    path = RESULTS_DIR / "best_rerank.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"resale_weight": 0.5, "age_weight": 0.0, "mobility_weight": 0.0}


def _tuned_label() -> str:
    w = _tuned_rerank()
    return f"(resale {w['resale_weight']}, age {w['age_weight']}, mobility {w['mobility_weight']})"


def replace_slots(fi, n):
    fi.config = replace(fi.config, cold_slots=n)
    return fi


def new_talents(setup: EvalSetup, scores: np.ndarray) -> dict:
    """Metrics of a list made only of players unknown to the graph at the cutoff."""
    s = np.array(scores, dtype=float, copy=True)
    s[:, setup.is_warm] = -np.inf
    m = evaluate(setup, s)
    return {"ndcg@10": m["cold_ndcg@10"], "coverage@10": m["coverage@10"],
            "within_budget@10": m["within_budget@10"], "value_change@10": m["value_change@10"],
            "age@10": m["age@10"]}


def summarise(rows: dict[str, list[dict]], metrics: list[str]) -> pd.DataFrame:
    table = []
    for name, runs in rows.items():
        df = pd.DataFrame(runs)
        row = {"model": name}
        for m in metrics:
            row[m] = df[m].mean()
            row[m + "_std"] = df[m].std(ddof=0) if len(df) > 1 else 0.0
        table.append(row)
    return pd.DataFrame(table)


def markdown(table: pd.DataFrame, metrics: list[str]) -> list[str]:
    lines = ["| Model | " + " | ".join(metrics) + " |", "| --- | " + " | ".join("---" for _ in metrics) + " |"]
    for _, r in table.iterrows():
        lines.append(f"| {r['model']} | " + " | ".join(fmt(r[m], r[m + '_std']) for m in metrics) + " |")
    return lines


def fmt(mean: float, std: float | None = None) -> str:
    if np.isnan(mean):
        return "n/a"
    return f"{mean:.4f}" if std is None or std == 0 else f"{mean:.4f} ± {std:.4f}"


def main() -> int:
    with open(RESULTS_DIR / "best_config.json") as f:
        cfg = Config(**json.load(f))
    inter = load_interactions()
    feats = load_or_build(TEST_START)
    setup = build_setup(inter, feats, TEST_START, None, _raw("player_valuations"))
    print(f"test: {len(setup.club_ids)} clubs, {len(setup.item_ids)} players in the catalogue, "
          f"{setup.n_targets} target recruitments ({setup.target_coverage:.1%} in the catalogue)")

    rows: dict[str, list[dict]] = {}
    rows["Random"] = [evaluate(setup, random_scores(setup, s)) for s in SEEDS]
    rows["Most popular (all history)"] = [evaluate(setup, most_popular_scores(setup, s)) for s in SEEDS]
    rows["Most popular (last 2 seasons)"] = [evaluate(setup, recent_popular_scores(setup, 2, s)) for s in SEEDS]
    rows["Highest market value"] = [evaluate(setup, market_value_scores(setup))]

    variants = {
        "LightGCN alone": lambda fi: ranking(replace_slots(fi, 0), setup, "lightgcn"),
        "Feature encoder alone": lambda fi: ranking(fi, setup, "encoder"),
        "LightGCN + 2 places for new players": lambda fi: ranking(replace_slots(fi, 2), setup, "hybrid"),
        "LightGCN + budget re-ranking": lambda fi: reranked(replace_slots(fi, 0), setup, feats, 0.0),
        "LightGCN + budget + resale (w=0.5)": lambda fi: reranked(replace_slots(fi, 0), setup, feats, 0.5),
        "LightGCN + budget + resale (w=1.0)": lambda fi: reranked(replace_slots(fi, 0), setup, feats, 1.0),
        # week 4: age and mobility corrections (weights chosen on validation, see src/tune.py)
        "LightGCN + budget + resale 0.5 + age (w=0.5)":
            lambda fi: reranked(replace_slots(fi, 0), setup, feats, 0.5, age_weight=0.5),
        "LightGCN + budget + resale 0.5 + age (w=1.0)":
            lambda fi: reranked(replace_slots(fi, 0), setup, feats, 0.5, age_weight=1.0),
        "LightGCN + budget + resale 0.5 + mobility (w=0.5)":
            lambda fi: reranked(replace_slots(fi, 0), setup, feats, 0.5, mobility_weight=0.5),
        f"LightGCN + budget + tuned re-ranking {_tuned_label()}":
            lambda fi: reranked(replace_slots(fi, 0), setup, feats, **_tuned_rerank()),
    }
    # "New talents" list: only players unknown to the graph at the cutoff
    cold_rows: dict[str, list[dict]] = {
        "Random": [new_talents(setup, random_scores(setup, s)) for s in SEEDS],
        "Highest market value": [new_talents(setup, market_value_scores(setup))],
    }
    fitted_seed0 = None
    for seed in SEEDS:
        fi = fit(setup, feats, replace(cfg, seed=seed), monitor=False, verbose=False)
        if seed == 0:
            fitted_seed0 = fi
        for name, fn in variants.items():
            rows.setdefault(name, []).append(evaluate(setup, None, ranked=fn(fi)))
        enc_scores = fi.club_emb @ fi.item_emb_enc.T
        cold_rows.setdefault("Cold-start encoder (ours)", []).append(new_talents(setup, enc_scores))
        print(f"seed {seed} done", flush=True)

    table = summarise(rows, METRICS)
    ref = target_reference(setup)
    cold_metrics = ["ndcg@10", "coverage@10", "within_budget@10", "value_change@10", "age@10"]
    cold_table = summarise(cold_rows, cold_metrics)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "test_results.csv", index=False)
    cold_table.to_csv(RESULTS_DIR / "test_results_new_talents.csv", index=False)

    lines = [
        f"# Test results (season 2025/26, cutoff {TEST_START})",
        "",
        f"{len(setup.club_ids)} clubs, {len(setup.item_ids)} players in the catalogue, "
        f"{setup.n_targets} target recruitments ({setup.target_coverage:.1%} in the catalogue), "
        f"{(~setup.is_warm).sum()} catalogue players unknown to the graph at the cutoff. "
        f"Mean ± std over {len(SEEDS)} seeds. Config: `{json.dumps(cfg.__dict__)}`.",
        "",
        "## Main list (all players)",
        "",
        *markdown(table, METRICS),
        "",
        f"Reference, players actually signed after the cutoff ({ref['n']} in the catalogue): "
        f"mean age {ref['age']:.1f}, mean value change {ref['value_change']:+.3f}. "
        f"The whole catalogue: mean age {setup.item_age.mean():.1f}, "
        f"mean value change {setup.future_change.mean():+.3f}.",
        "",
        "## New talents list (only players unknown to the graph at the cutoff)",
        "",
        "These players had no recruitment by a club of the 10 leagues in the 10 years before the cutoff, so LightGCN cannot score them. NDCG@10 is computed on the recruitments of such players only.",
        "",
        *markdown(cold_table, cold_metrics),
    ]
    (RESULTS_DIR / "test_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    case_studies(fitted_seed0, setup, feats)
    loss_figure()
    return 0


def case_studies(fi, setup: EvalSetup, feats: pd.DataFrame) -> None:
    ranked = reranked(fi, setup, feats, 0.5)
    idx = setup.club_index
    out = [f"# Case studies (test cutoff {TEST_START}, full model, resale weight 0.5)", "",
           "A star marks a player the club actually signed after the cutoff. 'new' marks a player "
           "unknown to the graph at the cutoff (cold start).", ""]
    for club_id, name in CASE_CLUBS.items():
        if club_id not in idx:
            continue
        i = idx[club_id]
        signed = set(setup.item_ids[setup.targets[i]])
        out += [f"## {name} (budget {setup.club_budget[i] / 1e6:.1f} M EUR)", "",
                "| Rank | Player | Pos. | Age | Club at cutoff | Value (M EUR) | Trend 12m | Note |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r, j in enumerate(ranked[i][:10], start=1):
            p = feats.iloc[j]
            note = []
            if p["player_id"] in signed:
                note.append("**signed**")
            if not fi.in_graph[j]:
                note.append("new")
            out.append(f"| {r} | {p['name']} | {p['position']} | {p['age']:.0f} | {p['club_name']} | "
                       f"{p['value_eur'] / 1e6:.1f} | {p['value_trend']:+.2f} | {' '.join(note)} |")
        enc = fi.club_emb[i] @ fi.item_emb_enc.T
        enc[fi.in_graph] = -np.inf
        enc[setup.exclude[i]] = -np.inf
        new = np.argsort(-enc)[:5]
        out += ["", "New talents (unknown to the graph): " + ", ".join(
            f"{feats.iloc[j]['name']} ({feats.iloc[j]['club_name']}, {feats.iloc[j]['age']:.0f})"
            + (" **signed**" if feats.iloc[j]['player_id'] in signed else "") for j in new)]
        actual = feats[feats["player_id"].isin(signed)]["name"].head(12).tolist()
        out += ["", "Actual signings in the catalogue: " + (", ".join(actual) if actual else "none"), ""]
    (RESULTS_DIR / "case_studies.md").write_text("\n".join(out) + "\n")


def loss_figure() -> None:
    log = pd.read_csv(RESULTS_DIR / "val_training_log.csv")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(log["epoch"], log["train_loss"], label="train BPR loss")
    v = log.dropna(subset=["val_loss"])
    ax[0].plot(v["epoch"], v["val_loss"], marker="o", label="validation BPR loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_title("Loss (LightGCN, validation run)")
    ax[0].legend()
    ax[1].plot(v["epoch"], v["val_ndcg@10"], marker="o", color="tab:green")
    ax[1].set_xlabel("epoch")
    ax[1].set_title("Validation warm NDCG@10")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "loss_curves.png", dpi=150)


if __name__ == "__main__":
    sys.exit(main())
