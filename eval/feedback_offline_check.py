"""Does the serving-time feedback update help predict real recruitments?

Usage:
    python -m eval.feedback_offline_check

Validation season 2024/25 only (the test season is not touched). The model
is trained on the history before VAL_START. For each club with at least 3
recruitments in the season, its first 2 recruitments (by date) play the
role of shortlisted players, and the later ones are the targets. The club
vector is updated toward the 2 "shortlisted" players (Rocchio, as in
src/predict.py) and NDCG@10 / Recall@20 on the later recruitments are
compared with the list without update.

Control: the same update with 2 random catalogue players instead of the
club's real first signings. If the gain came from any change of the club
vector, the control would gain too.

Main list as in the API: warm players ranked by LightGCN. Shortlisted
players and the usual exclusions (past signings, squad) are removed.
3 seeds. Writes eval/results/feedback_offline_check.md.
"""

import json
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from eval.harness import _ndcg, build_setup
from src.config import RESULTS_DIR, TEST_START, VAL_START
from src.features import _raw, load_or_build
from src.split import load_interactions, temporal_split
from src.train import Config, fit, item_embeddings

STRENGTHS = [0.25, 0.5, 1.0]
SEEDS = [0, 1, 2]
N_SHORTLIST = 2


def unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def evaluate_club(scores: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    ranked = np.argsort(-scores)[:20]
    return _ndcg(ranked, targets, 10), float(np.isin(targets, ranked).mean())


def main() -> int:
    with open(RESULTS_DIR / "best_config.json") as f:
        cfg = Config(**json.load(f))
    inter = load_interactions()
    feats = load_or_build(VAL_START)
    setup = build_setup(inter, feats, VAL_START, TEST_START, _raw("player_valuations"))
    _, future = temporal_split(inter, VAL_START, TEST_START)
    item_pos = setup.item_index

    # per club: first N_SHORTLIST recruitments as shortlist, the rest as targets
    cases = []
    for ci, club_id in enumerate(setup.club_ids):
        f = future[future["club_id"] == club_id].sort_values("transfer_date")
        idx = [item_pos[p] for p in f["player_id"] if p in item_pos]
        idx = list(dict.fromkeys(i for i in idx if i not in set(setup.exclude[ci])))
        if len(idx) >= N_SHORTLIST + 1:
            cases.append((ci, np.array(idx[:N_SHORTLIST]), np.array(idx[N_SHORTLIST:])))

    rng = np.random.default_rng(0)
    rows = []
    for seed in SEEDS:
        fi = fit(setup, feats, replace(cfg, seed=seed), monitor=False, verbose=False)
        warm_emb = item_embeddings(fi, "hybrid")
        anchor_emb = unit(np.where(fi.in_graph[:, None], warm_emb, fi.item_emb_enc))
        for ci, short, targets in cases:
            c = fi.club_emb[ci]
            banned = np.zeros(len(setup.item_ids), bool)
            banned[setup.exclude[ci]] = True
            banned[short] = True
            banned |= ~fi.in_graph
            random_short = rng.choice(np.where(~banned)[0], N_SHORTLIST, replace=False)

            def score(vec):
                s = warm_emb @ vec
                s[banned] = -np.inf
                return s

            base = evaluate_club(score(c), targets)
            rows.append({"seed": seed, "variant": "no update", "strength": 0.0, "club": ci,
                         "ndcg": base[0], "recall": base[1]})
            for a in STRENGTHS:
                for variant, anchors in (("real first signings", short), ("random players (control)", random_short)):
                    q = c + a * np.linalg.norm(c) * anchor_emb[anchors].mean(0)
                    n, r = evaluate_club(score(q), targets)
                    rows.append({"seed": seed, "variant": variant, "strength": a, "club": ci,
                                 "ndcg": n, "recall": r})
        print(f"seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    per_seed = df.groupby(["variant", "strength", "seed"])[["ndcg", "recall"]].mean().reset_index()
    agg = per_seed.groupby(["variant", "strength"]).agg(
        ndcg=("ndcg", "mean"), ndcg_sd=("ndcg", "std"), recall=("recall", "mean"), recall_sd=("recall", "std"))

    # paired comparison with the list without update, club by club (seed 0)
    s0 = df[df["seed"] == 0]
    base0 = s0[s0["variant"] == "no update"].set_index("club")["ndcg"]

    lines = [
        "# Serving-time feedback: offline check on the validation season",
        "",
        f"Season 2024/25, model trained before {VAL_START}, {len(cases)} clubs with at least "
        f"{N_SHORTLIST + 1} recruitments. The first {N_SHORTLIST} recruitments of each club act as shortlisted "
        "players, NDCG@10 and Recall@20 are computed on its later recruitments. Mean ± std over 3 seeds. "
        "Control: the same update toward 2 random players.",
        "",
        "| Club vector | Strength | NDCG@10 | Recall@20 | Clubs better / worse than no update (seed 0) |",
        "| --- | --- | --- | --- | --- |",
    ]
    order = [("no update", 0.0)] + [(v, a) for a in STRENGTHS for v in ("real first signings", "random players (control)")]
    for v, a in order:
        r = agg.loc[(v, a)]
        if v == "no update":
            better = "-"
        else:
            d = s0[(s0["variant"] == v) & (s0["strength"] == a)].set_index("club")["ndcg"] - base0
            better = f"{(d > 0).sum()} / {(d < 0).sum()}"
        lines.append(f"| {v} | {a} | {r['ndcg']:.4f} ± {r['ndcg_sd']:.4f} | {r['recall']:.4f} ± {r['recall_sd']:.4f} "
                     f"| {better} |")
    (RESULTS_DIR / "feedback_offline_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
