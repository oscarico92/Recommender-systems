"""Evaluation harness shared by every model and baseline.

A model gives a score matrix of shape (n_clubs, n_catalogue). The harness
removes the players a club cannot be recommended (already recruited before
the cutoff, or currently in its squad), ranks the rest and compares the top
of the ranking with the recruitments made after the cutoff.

Metrics
- ndcg@10, recall@20: accuracy on all target recruitments
- cold_ndcg@10: only targets unknown to the training graph (cold start)
- warm_ndcg@10: only targets already seen in the training graph
- coverage@10: share of the catalogue recommended to at least one club
- within_budget@10: share of the top 10 whose market value is within the
  club's budget (90th percentile of the values it recruited in 3 years)
- value_change@10: mean log change of the market value of the top 10 in
  the year after the cutoff. Uses future valuations, for evaluation only.
- age@10: mean age of the top 10 at the cutoff
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.split import temporal_split

K_NDCG = 10
K_RECALL = 20


@dataclass
class EvalSetup:
    cutoff: str
    club_ids: np.ndarray                 # (C,)
    item_ids: np.ndarray                 # (N,) catalogue player ids
    history: pd.DataFrame                # interactions before the cutoff
    targets: list[np.ndarray]            # per club, catalogue indices
    exclude: list[np.ndarray]            # per club, catalogue indices
    is_warm: np.ndarray                  # (N,) player seen before the cutoff
    item_value: np.ndarray               # (N,) market value at the cutoff
    club_budget: np.ndarray              # (C,)
    future_change: np.ndarray            # (N,) log value change after the cutoff
    target_coverage: float               # share of targets that are in the catalogue
    n_targets: int = field(default=0)
    item_age: np.ndarray | None = None   # (N,) age at the cutoff

    @property
    def club_index(self) -> dict[int, int]:
        return {c: i for i, c in enumerate(self.club_ids)}

    @property
    def item_index(self) -> dict[int, int]:
        return {p: i for i, p in enumerate(self.item_ids)}


def club_value_quantile(history: pd.DataFrame, club_ids: np.ndarray, cutoff: str, q: float,
                        default: float) -> np.ndarray:
    """Quantile q of the market values a club recruited in the 3 years before the cutoff
    (all its history if it recruited nobody in 3 years, `default` if nothing is known)."""
    recent = history[history["transfer_date"] >= pd.Timestamp(cutoff) - pd.Timedelta(days=3 * 365)]
    q_recent = recent.groupby("club_id")["market_value_eur"].quantile(q)
    q_all = history.groupby("club_id")["market_value_eur"].quantile(q)
    out = q_recent.reindex(club_ids).fillna(q_all.reindex(club_ids)).fillna(default)
    return out.to_numpy(float)


def club_budgets(history: pd.DataFrame, club_ids: np.ndarray, cutoff: str) -> np.ndarray:
    """Ceiling of the price range: 90th percentile of the values recruited in 3 years."""
    return club_value_quantile(history, club_ids, cutoff, 0.9, 1e6)


def club_value_floors(history: pd.DataFrame, club_ids: np.ndarray, cutoff: str) -> np.ndarray:
    """Floor of the price range: 25th percentile of the values recruited in 3 years."""
    return np.minimum(club_value_quantile(history, club_ids, cutoff, 0.25, 0.0),
                      club_budgets(history, club_ids, cutoff))


def build_setup(
    interactions: pd.DataFrame,
    features: pd.DataFrame,
    cutoff: str,
    end: str | None,
    valuations: pd.DataFrame | None = None,
) -> EvalSetup:
    history, future = temporal_split(interactions, cutoff, end)
    club_ids = np.sort(history["club_id"].unique())
    item_ids = features["player_id"].to_numpy()
    item_idx = {p: i for i, p in enumerate(item_ids)}

    exclude, targets = [], []
    hist_by_club = history.groupby("club_id")["player_id"].apply(set)
    squad_by_club = features.groupby("club_id")["player_id"].apply(set)
    fut_by_club = future.groupby("club_id")["player_id"].apply(set)
    n_targets = n_in_cat = 0
    for c in club_ids:
        ex = hist_by_club.get(c, set()) | squad_by_club.get(c, set())
        ex_idx = np.array(sorted(item_idx[p] for p in ex if p in item_idx), dtype=np.int64)
        fut = fut_by_club.get(c, set())
        n_targets += len(fut)
        tg = [item_idx[p] for p in fut if p in item_idx]
        n_in_cat += len(tg)
        tg_idx = np.setdiff1d(np.array(tg, dtype=np.int64), ex_idx)
        exclude.append(ex_idx)
        targets.append(tg_idx)

    seen = set(history["player_id"])
    is_warm = np.array([p in seen for p in item_ids])

    future_change = np.zeros(len(item_ids))
    if valuations is not None:
        t = pd.Timestamp(cutoff)
        later = valuations[(valuations["date"] >= t) & (valuations["date"] < t + pd.Timedelta(days=365))]
        last_future = later.sort_values("date").groupby("player_id")["market_value_in_eur"].last()
        fv = last_future.reindex(item_ids).to_numpy(float)
        now = features["value_eur"].to_numpy(float)
        future_change = np.where(np.isnan(fv), 0.0, np.log1p(fv) - np.log1p(now))

    return EvalSetup(
        cutoff=cutoff,
        club_ids=club_ids,
        item_ids=item_ids,
        history=history,
        targets=targets,
        exclude=exclude,
        is_warm=is_warm,
        item_value=features["value_eur"].to_numpy(float),
        club_budget=club_budgets(history, club_ids, cutoff),
        future_change=future_change,
        target_coverage=n_in_cat / max(n_targets, 1),
        n_targets=n_targets,
        item_age=features["age"].to_numpy(float),
    )


def mask_scores(setup: EvalSetup, scores: np.ndarray) -> np.ndarray:
    scores = np.array(scores, dtype=np.float64, copy=True)
    for i, ex in enumerate(setup.exclude):
        scores[i, ex] = -np.inf
    return scores


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def _ndcg(ranked: np.ndarray, target: np.ndarray, k: int) -> float:
    hits = np.isin(ranked[:k], target)
    dcg = (hits / np.log2(np.arange(2, len(hits) + 2))).sum()
    idcg = (1 / np.log2(np.arange(2, min(len(target), k) + 2))).sum()
    return float(dcg / idcg)


def evaluate(setup: EvalSetup, scores: np.ndarray, ranked: np.ndarray | None = None) -> dict:
    """Metrics of a score matrix. `ranked` (C, >=20) overrides the ranking
    when a re-ranking step changes the order."""
    if ranked is None:
        ranked = top_k(mask_scores(setup, scores), K_RECALL)
    ndcg, recall, cold, warm = [], [], [], []
    for i, tg in enumerate(setup.targets):
        if len(tg) == 0:
            continue
        r = ranked[i]
        ndcg.append(_ndcg(r, tg, K_NDCG))
        recall.append(np.isin(tg, r[:K_RECALL]).mean())
        tg_cold = tg[~setup.is_warm[tg]]
        tg_warm = tg[setup.is_warm[tg]]
        if len(tg_cold):
            cold.append(_ndcg(r, tg_cold, K_NDCG))
        if len(tg_warm):
            warm.append(_ndcg(r, tg_warm, K_NDCG))
    top10 = ranked[:, :K_NDCG]
    within = setup.item_value[top10] <= setup.club_budget[:, None]
    return {
        "ndcg@10": float(np.mean(ndcg)),
        "recall@20": float(np.mean(recall)),
        "cold_ndcg@10": float(np.mean(cold)) if cold else float("nan"),
        "warm_ndcg@10": float(np.mean(warm)) if warm else float("nan"),
        "coverage@10": float(len(np.unique(top10)) / len(setup.item_ids)),
        "within_budget@10": float(within.mean()),
        "value_change@10": float(setup.future_change[top10].mean()),
        "age@10": float(setup.item_age[top10].mean()) if setup.item_age is not None else float("nan"),
        "n_clubs_evaluated": len(ndcg),
    }


def target_reference(setup: EvalSetup) -> dict:
    """Mean age and value change of the players actually signed (in the catalogue)."""
    idx = np.concatenate([t for t in setup.targets if len(t)])
    return {
        "value_change": float(setup.future_change[idx].mean()),
        "age": float(setup.item_age[idx].mean()) if setup.item_age is not None else float("nan"),
        "n": int(len(idx)),
    }
