"""Baselines scored with the same harness as the models.

- random: uniform random ranking
- most_popular: players recruited the most often before the cutoff (all history)
- recent_popular: players recruited the most often in the last 2 seasons
- market_value: players with the highest market value at the cutoff
  (a naive scouting rule, harder to beat than random)
"""

import numpy as np
import pandas as pd

from eval.harness import EvalSetup


def random_scores(setup: EvalSetup, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((len(setup.club_ids), len(setup.item_ids)))


def most_popular_scores(setup: EvalSetup, seed: int = 0) -> np.ndarray:
    counts = setup.history.groupby("player_id").size()
    pop = counts.reindex(setup.item_ids).fillna(0).to_numpy(float)
    # tiny noise to break ties between the many players never recruited
    pop = pop + np.random.default_rng(seed).random(len(pop)) * 1e-3
    return np.tile(pop, (len(setup.club_ids), 1))


def recent_popular_scores(setup: EvalSetup, seasons: int = 2, seed: int = 0) -> np.ndarray:
    start = pd.Timestamp(setup.cutoff) - pd.Timedelta(days=int(365.25 * seasons))
    recent = setup.history[setup.history["transfer_date"] >= start]
    pop = recent.groupby("player_id").size().reindex(setup.item_ids).fillna(0).to_numpy(float)
    pop = pop + np.random.default_rng(seed).random(len(pop)) * 1e-3
    return np.tile(pop, (len(setup.club_ids), 1))


def market_value_scores(setup: EvalSetup) -> np.ndarray:
    return np.tile(np.log1p(setup.item_value), (len(setup.club_ids), 1))
