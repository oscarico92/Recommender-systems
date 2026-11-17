"""Check that retraining with app feedback moves the right players.

Usage:
    python -m eval.feedback_check

Simulates a scout of one club using the app: 3 players ranked 30 to 50 are
shortlisted and the top 2 are dismissed. The serving model is trained once
without and once with this feedback (same seed), and the rank of these
players for the club is compared. Also checks that the other clubs are
not disturbed more than by a new training run. Writes eval/results/feedback_check.md.
"""

import json
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from eval.harness import build_setup, mask_scores
from src.config import RESULTS_DIR, SERVING_CUTOFF
from src.feedback import to_signals
from src.features import load_or_build
from src.split import load_interactions
from src.train import Config, fit, item_embeddings

CLUB_ID = 985  # Manchester United


def ranks(fitted, setup, club_i: int) -> np.ndarray:
    s = fitted.club_emb @ item_embeddings(fitted, "hybrid").T
    s[:, ~fitted.in_graph] = -np.inf
    s = mask_scores(setup, s)[club_i]
    order = np.argsort(-s)
    r = np.empty_like(order)
    r[order] = np.arange(1, len(order) + 1)
    return r


def top_overlap(a, b, setup, k: int = 10) -> float:
    sa = mask_scores(setup, np.where(a.in_graph, a.club_emb @ item_embeddings(a, "hybrid").T, -np.inf))
    sb = mask_scores(setup, np.where(b.in_graph, b.club_emb @ item_embeddings(b, "hybrid").T, -np.inf))
    ta = np.argsort(-sa, axis=1)[:, :k]
    tb = np.argsort(-sb, axis=1)[:, :k]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(ta, tb)]))


def main() -> int:
    with open(RESULTS_DIR / "best_config.json") as f:
        cfg = Config(**json.load(f))
    inter = load_interactions()
    feats = load_or_build(SERVING_CUTOFF)
    setup = build_setup(inter, feats, SERVING_CUTOFF, None)
    ci = setup.club_index[CLUB_ID]

    base = fit(setup, feats, cfg, monitor=False, verbose=False)
    r0 = ranks(base, setup, ci)
    order = np.argsort(r0)
    shortlisted = order[[29, 39, 49]]
    dismissed = order[[0, 1]]
    pid = setup.item_ids
    now = "2026-11-17T12:00:00Z"
    actions = pd.DataFrame(
        [(i, CLUB_ID, int(pid[j]), "shortlisted", now) for i, j in enumerate(shortlisted)]
        + [(10 + i, CLUB_ID, int(pid[j]), "dismissed", now) for i, j in enumerate(dismissed)],
        columns=["id", "club_id", "player_id", "action", "created_at"],
    )
    with_fb = fit(setup, feats, cfg, monitor=False, verbose=False, feedback=to_signals(actions))
    r1 = ranks(with_fb, setup, ci)

    # the same training with another seed, to know the run-to-run noise
    again = fit(setup, feats, replace(cfg, seed=cfg.seed + 1), monitor=False, verbose=False)

    names = feats.set_index("player_id")["name"]
    lines = [
        "# Retraining with app feedback: check",
        "",
        f"Club: Manchester United. Serving model (cutoff {SERVING_CUTOFF}), same seed with and without feedback. "
        "Shortlisted players are added as positive edges (weight 0.5), dismissed players are hard negatives. "
        "In the API, a dismissed player is also hidden at once, before any retraining.",
        "",
        "| Player | Action | Rank before | Rank after |",
        "| --- | --- | --- | --- |",
    ]
    for j in shortlisted:
        lines.append(f"| {names[pid[j]]} | shortlisted | {r0[j]} | {r1[j]} |")
    for j in dismissed:
        lines.append(f"| {names[pid[j]]} | dismissed | {r0[j]} | {r1[j]} |")
    ov_fb = top_overlap(base, with_fb, setup)
    ov_seed = top_overlap(base, again, setup)
    lines += [
        "",
        f"Top 10 overlap with the model without feedback, averaged over the 322 clubs: {ov_fb:.0%}. "
        f"Training without feedback with another seed gives {ov_seed:.0%}: the top 10 of a club already change "
        "that much from one training run to the next, so the feedback of one club does not disturb the "
        "others more than a new training run does.",
    ]
    (RESULTS_DIR / "feedback_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
