"""Train the served model on all the data and export it for the API.

Usage:
    python -m src.export_model

Uses the hyperparameters of eval/results/best_config.json, all the
interactions before SERVING_CUTOFF, and the player features at that date.
Writes models/serving/ (a few MB, committed so that the app runs without
training). The API only needs numpy and pandas to read it.
"""

import json
import sys

import numpy as np
import pandas as pd

from eval.harness import build_setup
from src.config import RESULTS_DIR, SERVING_CUTOFF, SERVING_DIR
from src.features import _raw, load_or_build
from src.split import load_interactions
from src.train import Config, fit, item_embeddings


def main() -> int:
    with open(RESULTS_DIR / "best_config.json") as f:
        cfg = Config(**json.load(f))
    inter = load_interactions()
    feats = load_or_build(SERVING_CUTOFF)
    setup = build_setup(inter, feats, SERVING_CUTOFF, None)
    fitted = fit(setup, feats, cfg, monitor=False, verbose=False)

    clubs = pd.read_parquet(RESULTS_DIR.parents[1] / "data" / "processed" / "clubs.parquet")
    clubs = clubs.set_index("club_id").reindex(setup.club_ids).reset_index()
    clubs["budget_eur"] = setup.club_budget

    names = _raw("players").set_index("player_id")["name"]
    # full history: used to exclude past signings and, for the last 5 years, to explain
    history = setup.history[["club_id", "player_id", "transfer_date"]].copy()
    history["player_name"] = names.reindex(history["player_id"]).fillna("?").to_numpy()

    # LightGCN embeddings of every player of the graph, for the explanations
    graph = fitted.graph
    graph_emb = fitted.graph_player_emb.astype(np.float32)

    meta_cols = ["player_id", "name", "position", "sub_position", "age", "country",
                 "club_id", "club_name", "league_id", "value_eur", "value_trend"]
    players = feats[meta_cols].copy()

    SERVING_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SERVING_DIR / "embeddings.npz",
        club_emb=fitted.club_emb.astype(np.float32),
        warm_emb=item_embeddings(fitted, "hybrid").astype(np.float32),
        cold_emb=fitted.item_emb_enc.astype(np.float32),
        in_graph=fitted.in_graph,
        graph_ids=graph.player_ids,
        graph_emb=graph_emb,
    )
    players.to_parquet(SERVING_DIR / "players.parquet", index=False)
    clubs.to_parquet(SERVING_DIR / "clubs.parquet", index=False)
    history.to_parquet(SERVING_DIR / "history.parquet", index=False)
    # re-ranking weights chosen on validation (python -m src.tune --rerank-only), used as API defaults
    rerank = {}
    if (RESULTS_DIR / "best_rerank.json").exists():
        with open(RESULTS_DIR / "best_rerank.json") as f:
            rerank = {k: v for k, v in json.load(f).items() if k in ("age_weight", "mobility_weight")}
    with open(SERVING_DIR / "config.json", "w") as f:
        json.dump({**cfg.__dict__, **rerank, "cutoff": SERVING_CUTOFF,
                   "model_version": f"lightgcn-{SERVING_CUTOFF}"}, f, indent=2)
    print(f"exported {len(clubs)} clubs, {len(players)} players to {SERVING_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
