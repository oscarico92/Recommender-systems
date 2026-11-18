import numpy as np
import pandas as pd

from eval.harness import EvalSetup, _ndcg, evaluate, top_k
from src.rerank import merge_with_cold_slots, rerank_group
from src.split import temporal_split


def test_ndcg_values():
    assert _ndcg(np.array([5, 1, 2]), np.array([5]), 10) == 1.0
    assert abs(_ndcg(np.array([1, 5, 2]), np.array([5]), 10) - 1 / np.log2(3)) < 1e-12
    assert _ndcg(np.array([1, 2, 3]), np.array([9]), 10) == 0.0


def test_temporal_split_has_no_overlap():
    df = pd.DataFrame({"transfer_date": pd.to_datetime(["2024-06-30", "2024-07-01", "2025-07-01"]),
                       "club_id": 1, "player_id": [1, 2, 3]})
    hist, fut = temporal_split(df, "2024-07-01", "2025-07-01")
    assert list(hist.player_id) == [1] and list(fut.player_id) == [2]


def _setup():
    n = 30
    return EvalSetup(
        cutoff="2024-07-01", club_ids=np.array([1, 2]), item_ids=np.arange(n),
        history=pd.DataFrame(), targets=[np.array([3]), np.array([25, 26])],
        exclude=[np.array([0]), np.array([], dtype=np.int64)],
        is_warm=np.arange(n) < 20, item_value=np.full(n, 1e6), club_budget=np.array([2e6, 5e5]),
        future_change=np.zeros(n), target_coverage=1.0)


def test_evaluate_perfect_and_excluded():
    st = _setup()
    scores = np.zeros((2, 30))
    scores[0, 0] = 10      # excluded, must not count
    scores[0, 3] = 5
    scores[1, [25, 26]] = 5
    m = evaluate(st, scores)
    assert m["ndcg@10"] == 1.0 and m["recall@20"] == 1.0
    assert m["cold_ndcg@10"] == 1.0 and m["warm_ndcg@10"] == 1.0
    assert m["within_budget@10"] == 0.5   # club 2 budget is below every value


def test_top_k_order():
    s = np.array([[0.1, 0.9, 0.5, 0.7]])
    assert top_k(s, 3).tolist() == [[1, 3, 2]]


def test_merge_with_cold_slots():
    w = np.arange(100, 120)[None, :]
    c = np.arange(20)[None, :]
    out = merge_with_cold_slots(w, c, 2)[0]
    assert out[4] == 0 and out[9] == 1 and out[0] == 100 and len(set(out)) == 20
    assert (merge_with_cold_slots(w, c, 0)[0] == w[0]).all()


def test_rerank_budget_and_resale():
    scores = np.array([3.0, 2.0, 1.0, -np.inf])
    value = np.array([50e6, 1e6, 1e6, 1e6])
    trend = np.array([0.0, -0.5, 1.0, 0.0])
    out = rerank_group(scores, value, trend, budget=10e6, resale_weight=0.0, length=3)
    assert out.tolist() == [1, 2, 0]          # over budget goes last, excluded never appears
    out = rerank_group(scores, value, trend, budget=10e6, resale_weight=5.0, length=2)
    assert out.tolist() == [2, 1]             # strong resale focus favours the rising player


def test_rerank_age_and_mobility():
    scores = np.array([3.0, 2.0, 1.0])
    value = np.full(3, 1e6)
    trend = np.zeros(3)
    age = np.array([31.0, 29.0, 21.0])
    moves = np.array([0.0, 6.0, 1.0])
    base = rerank_group(scores, value, trend, None, 0.0, 3)
    assert base.tolist() == [0, 1, 2]
    # default weights (0) keep the model order even when age and moves are given
    same = rerank_group(scores, value, trend, None, 0.0, 3, age=age, moves=moves)
    assert same.tolist() == base.tolist()
    young = rerank_group(scores, value, trend, None, 0.0, 3, age=age, age_weight=5.0)
    assert young[0] == 2                      # strong age penalty puts the 21-year-old first
    calm = rerank_group(scores, value, trend, None, 0.0, 3, moves=moves, mobility_weight=5.0)
    assert calm[-1] == 1                      # the player with 6 moves goes last


def test_select_rerank_prefers_younger_within_tolerance():
    from src.rerank import select_rerank
    table = pd.DataFrame({
        "resale_weight": [0.5, 0.5, 0.5, 0.5],
        "age_weight": [0.0, 0.5, 1.0, 1.0],
        "mobility_weight": [0.0, 0.0, 0.0, 0.5],
        "ndcg@10": [0.0120, 0.0115, 0.0100, 0.0112],
        "ndcg@10_std": [0.0010, 0.0010, 0.0010, 0.0010],
    })
    top = select_rerank(table)
    # 0.0100 is outside the tolerance (0.0120 - 0.0010), so the strongest
    # admissible age weight is 1.0 with mobility 0.5 (ndcg 0.0112)
    assert (top["age_weight"], top["mobility_weight"]) == (1.0, 0.5)
    # with a zero tolerance the most accurate setting wins
    assert select_rerank(table.assign(**{"ndcg@10_std": 0.0}))["age_weight"] == 0.0
