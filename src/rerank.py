"""Budget and resale re-ranking, shared by the evaluation and the API.

For one club and one group of players (warm or cold):
1. take the `n_candidates` best players by model score, among the players
   whose market value is at least `min_value` (the club's price floor) when given
2. drop the players whose market value is above the club's budget
3. re-score and sort:
       z(model score)
     + resale_weight   * z(value trend over the last 12 months, known at the cutoff)
     - age_weight      * z(age at the cutoff)
     - mobility_weight * z(log(1 + number of past moves between the selected clubs))

The age term corrects the age bias of the model (its top 10 is about 29
years old on the test season, the players actually signed are 24.5). The
mobility term pushes down the players that change club very often, who are
recommended to many clubs because they are central in the graph.
All z-scores are computed on the candidates left after the budget filter.

The price floor exists because the budget alone is only a ceiling: for a
club like Paris Saint-Germain (budget 56 M EUR) the model's best candidates
included players valued 0.2 M EUR, far below anything the club signs for its
first team. Without `min_value` the behaviour is unchanged, so the evaluation
results in eval/results/ are reproduced exactly.
"""

import numpy as np
import pandas as pd

N_CANDIDATES = 50


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def rerank_group(
    scores: np.ndarray,
    value: np.ndarray,
    trend: np.ndarray,
    budget: float | None,
    resale_weight: float,
    length: int,
    n_candidates: int = N_CANDIDATES,
    age: np.ndarray | None = None,
    age_weight: float = 0.0,
    moves: np.ndarray | None = None,
    mobility_weight: float = 0.0,
    min_value: float | None = None,
) -> np.ndarray:
    """Indices of the best `length` players after re-ranking.

    `scores` must already be -inf for players that cannot be recommended.
    If the budget leaves fewer than `length` players, the list is completed
    with over-budget players in model order, so that it always has `length` items.
    With a price floor, the list is completed first with players below the
    floor but within the budget (the budget is the hard limit, the floor a
    preference), then with over-budget players.
    """
    finite = np.isfinite(scores)
    order = np.argsort(-np.where(finite, scores, -np.inf))
    order = order[finite[order]]
    extra = dict(scores=scores, value=value, trend=trend, budget=budget, resale_weight=resale_weight,
                 size=max(n_candidates, length), age=age, age_weight=age_weight,
                 moves=moves, mobility_weight=mobility_weight)
    if min_value:
        ok, rest = _pick(order[value[order] >= min_value], **extra)
        if len(ok) < length:
            low, _ = _pick(order[value[order] < min_value], **extra)
            ok = np.concatenate([ok, low])
    else:
        ok, rest = _pick(order, **extra)
    out = np.concatenate([ok, rest])[:length]
    if len(out) < length:  # very small candidate set: pad with the next players
        extra = order[~np.isin(order, out)][: length - len(out)]
        out = np.concatenate([out, extra])
    return out


def _pick(order, scores, value, trend, budget, resale_weight, size, age, age_weight, moves, mobility_weight):
    """Among the first `size` players of `order`: the ones within budget, re-ranked,
    and the over-budget ones in model order."""
    cand = order[:size]
    if budget is not None:
        ok = cand[value[cand] <= budget]
        rest = cand[value[cand] > budget]
    else:
        ok, rest = cand, cand[:0]
    if len(ok):
        new = _z(scores[ok]) + resale_weight * _z(trend[ok])
        if age is not None and age_weight:
            new -= age_weight * _z(age[ok])
        if moves is not None and mobility_weight:
            new -= mobility_weight * _z(np.log1p(moves[ok]))
        ok = ok[np.argsort(-new, kind="stable")]
    return ok, rest


def merge_with_cold_slots(warm_rank: np.ndarray, cold_rank: np.ndarray, cold_slots: int, length: int = 20) -> np.ndarray:
    """Interleave two rankings: `cold_slots` places out of every 10 go to cold players.

    With 2 slots, ranks 5 and 10 (then 15 and 20) are the best cold players,
    the other ranks are the best warm players.
    """
    if cold_slots <= 0:
        return warm_rank[:, :length]
    step = 10 // cold_slots
    is_cold = np.array([(pos + 1) % step == 0 and (pos % 10) // step < cold_slots for pos in range(length)])
    out = np.empty((warm_rank.shape[0], length), dtype=np.int64)
    out[:, is_cold] = cold_rank[:, : is_cold.sum()]
    out[:, ~is_cold] = warm_rank[:, : (~is_cold).sum()]
    return out


def select_rerank(table: pd.DataFrame) -> pd.Series:
    """Selection rule for the re-ranking weights.

    Keeping the best NDCG@10 alone would probably keep age_weight = 0: a
    younger list does not have to be more accurate. The rule is therefore:
    1. tolerance = the seed standard deviation of the best NDCG@10
    2. keep the settings whose mean NDCG@10 >= best - tolerance
    3. among them, take the strongest age_weight, then the best NDCG@10
    """
    best = table.loc[table["ndcg@10"].idxmax()]
    ok = table[table["ndcg@10"] >= best["ndcg@10"] - best["ndcg@10_std"]]
    return ok.sort_values(["age_weight", "ndcg@10"], ascending=[False, False]).iloc[0]
