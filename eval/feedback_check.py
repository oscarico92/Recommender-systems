"""Does retraining with app feedback generalise to similar players?

Usage:
    python -m eval.feedback_check

For each test club, a scout shortlists the players ranked 30, 40 and 50 and
dismisses the top 2. The shortlisted and dismissed players themselves are
then excluded from the club's list (the API hides them), so moving them up
or down would only show memorisation. What we measure is the rank change of
their neighbours: the 20 players closest to each of them in the embedding
space of the model without feedback.

The model is trained (same seed) without and with the feedback of that club.
As a reference for the run-to-run noise, the model is also trained without
feedback with another seed, and the same neighbour groups are measured.
Ranks are among the warm catalogue players (LightGCN list) of the club.
Writes eval/results/feedback_check.md.
"""

import json
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from eval.harness import build_setup
from src.config import RESULTS_DIR, SERVING_CUTOFF
from src.feedback import to_signals
from src.features import load_or_build
from src.predict import FEEDBACK_STRENGTH
from src.split import load_interactions
from src.train import Config, fit, item_embeddings

CLUBS = {985: "Manchester United", 294: "Benfica", 371: "Celtic", 826: "RC Lens", 1237: "Brighton"}
N_NEIGHBOURS = 20


def club_ranks(fitted, setup, ci: int, hidden: np.ndarray, club_vec: np.ndarray | None = None) -> np.ndarray:
    """Rank (1 = best) of every catalogue player for club ci; hidden players get NaN."""
    c = fitted.club_emb[ci] if club_vec is None else club_vec
    s = item_embeddings(fitted, "hybrid") @ c
    s[~fitted.in_graph] = -np.inf
    s[setup.exclude[ci]] = -np.inf   # past signings and current squad
    s[hidden] = -np.inf
    order = np.argsort(-s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    r[~np.isfinite(s)] = np.nan
    return r


def neighbours(emb: np.ndarray, anchors: np.ndarray, allowed: np.ndarray, k: int) -> np.ndarray:
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    out = set()
    for a in anchors:
        sim = e @ e[a]
        sim[~allowed] = -np.inf
        out.update(np.argsort(-sim)[:k].tolist())
    return np.array(sorted(out))


def main() -> int:
    with open(RESULTS_DIR / "best_config.json") as f:
        cfg = Config(**json.load(f))
    inter = load_interactions()
    feats = load_or_build(SERVING_CUTOFF)
    setup = build_setup(inter, feats, SERVING_CUTOFF, None)
    pid = setup.item_ids

    base = fit(setup, feats, cfg, monitor=False, verbose=False)
    other_seed = fit(setup, feats, replace(cfg, seed=cfg.seed + 1), monitor=False, verbose=False)
    base_emb = item_embeddings(base, "hybrid")

    rows, overlaps = [], []
    for club_id, name in CLUBS.items():
        ci = setup.club_index[club_id]
        r0 = club_ranks(base, setup, ci, np.array([], dtype=int))
        order = np.argsort(np.nan_to_num(r0, nan=np.inf))
        shortlisted, dismissed = order[[29, 39, 49]], order[[0, 1]]
        feedback_players = np.concatenate([shortlisted, dismissed])

        allowed = np.isfinite(r0)
        allowed[feedback_players] = False
        near_short = neighbours(base_emb, shortlisted, allowed, N_NEIGHBOURS)
        near_dism = neighbours(base_emb, dismissed, allowed, N_NEIGHBOURS)
        near_dism = np.setdiff1d(near_dism, near_short)
        others = np.setdiff1d(np.where(allowed)[0], np.concatenate([near_short, near_dism]))

        now = "2026-11-17T12:00:00Z"
        actions = pd.DataFrame(
            [(i, club_id, int(pid[j]), "shortlisted", now) for i, j in enumerate(shortlisted)]
            + [(10 + i, club_id, int(pid[j]), "dismissed", now) for i, j in enumerate(dismissed)],
            columns=["id", "club_id", "player_id", "action", "created_at"],
        )
        with_fb = fit(setup, feats, cfg, monitor=False, verbose=False, feedback=to_signals(actions))

        r_base = club_ranks(base, setup, ci, feedback_players)
        r_fb = club_ranks(with_fb, setup, ci, feedback_players)
        r_seed = club_ranks(other_seed, setup, ci, feedback_players)
        # serving-time update (what the API does): no retraining, same model
        unit = base_emb / (np.linalg.norm(base_emb, axis=1, keepdims=True) + 1e-9)
        c = base.club_emb[ci]
        q = c + FEEDBACK_STRENGTH * np.linalg.norm(c) * (unit[shortlisted].mean(0) - unit[dismissed].mean(0))
        r_q = club_ranks(base, setup, ci, feedback_players, club_vec=q)
        top = lambda r: set(np.argsort(np.nan_to_num(r, nan=np.inf))[:10])  # noqa: E731
        overlaps.append({"retrain": len(top(r_base) & top(r_fb)) / 10, "serving": len(top(r_base) & top(r_q)) / 10,
                         "seed": len(top(r_base) & top(r_seed)) / 10})
        for group, idx in [("neighbours of shortlisted", near_short), ("neighbours of dismissed", near_dism),
                           ("all other players", others)]:
            up_fb = r_base[idx] - r_fb[idx]        # positive = moved up
            up_seed = r_base[idx] - r_seed[idx]
            up_q = r_base[idx] - r_q[idx]
            rows.append({"club": name, "group": group, "n": len(idx),
                         "median_up_serving": float(np.nanmedian(up_q)),
                         "median_up_feedback": float(np.nanmedian(up_fb)),
                         "median_up_new_seed": float(np.nanmedian(up_seed))})
        print(f"{name} done", flush=True)

    df = pd.DataFrame(rows)
    agg = df.groupby("group", sort=False).agg(n=("n", "sum"), q=("median_up_serving", "mean"),
                                               fb=("median_up_feedback", "mean"),
                                               seed=("median_up_new_seed", "mean"))
    ov = pd.DataFrame(overlaps).mean()
    lines = [
        "# Retraining with app feedback: does it generalise?",
        "",
        f"Serving model (cutoff {SERVING_CUTOFF}). For each of {len(CLUBS)} clubs ({', '.join(CLUBS.values())}), "
        "the players ranked 30, 40 and 50 are shortlisted and the top 2 are dismissed. These 5 players are then "
        "hidden from the club's list (the API does the same), so the table measures their neighbours: the "
        f"{N_NEIGHBOURS} closest players to each of them in the embedding space of the model without feedback. "
        "Ranks are among about 6,000 warm candidates. Positive = moved up.",
        "",
        "Three columns:",
        "",
        f"- **Serving-time update** (what the API does): the club vector moves toward the shortlisted players and "
        f"away from the dismissed ones (Rocchio, strength {FEEDBACK_STRENGTH}), same model, no retraining.",
        "- **Retraining**: shortlisted players become positive edges (weight 0.5 in the BPR loss) and dismissed "
        "players hard negatives (`python -m src.export_model --feedback`), same seed.",
        "- **New seed**: retraining without any feedback with another seed, the run-to-run noise the retraining "
        "column must be compared with.",
        "",
        "| Group | Players (5 clubs) | Serving-time update | Retraining | New seed, no feedback |",
        "| --- | --- | --- | --- | --- |",
    ]
    for g, r in agg.iterrows():
        lines.append(f"| {g} | {int(r['n'])} | {r['q']:+.0f} | {r['fb']:+.0f} | {r['seed']:+.0f} |")
    lines += ["", f"Top 10 kept (mean over the clubs): serving-time update {ov['serving']:.0%}, retraining "
              f"{ov['retrain']:.0%}, new seed {ov['seed']:.0%}.",
              "", "Per club:", "", "| Club | Group | Players | Serving-time update | Retraining | New seed |",
              "| --- | --- | --- | --- | --- | --- |"]
    for _, r in df.iterrows():
        lines.append(f"| {r['club']} | {r['group']} | {r['n']} | {r['median_up_serving']:+.0f} | "
                     f"{r['median_up_feedback']:+.0f} | {r['median_up_new_seed']:+.0f} |")
    (RESULTS_DIR / "feedback_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
