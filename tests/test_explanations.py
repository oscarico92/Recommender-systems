"""Explanations of the served model (needs models/serving/, committed in the repo)."""

import pytest

from src.config import SERVING_DIR

pytestmark = pytest.mark.skipif(not (SERVING_DIR / "embeddings.npz").exists(), reason="no exported model")


@pytest.fixture(scope="module")
def rec():
    from src.predict import Recommender
    return Recommender.load()


def test_new_talent_explanations_vary(rec):
    """Week 4 bug: every new talent was explained by the same two signings."""
    club_id = int(rec.clubs.sort_values("n_transfers").iloc[-1]["club_id"])
    out = rec.recommend(club_id, k=10, only_new=True)
    pairs = {tuple(sorted(s["player_id"] for s in o["similar_to"])) for o in out}
    assert len(pairs) >= 4
    assert all(s["basis"] == "profile" for o in out for s in o["similar_to"])


def test_new_talent_explained_by_same_position(rec):
    pos = dict(zip(rec.players["player_id"], rec.players["position"]))
    same = total = 0
    for club_id in rec.clubs["club_id"].head(20):
        for o in rec.recommend(int(club_id), k=10, only_new=True):
            for s in o["similar_to"]:
                total += 1
                same += pos.get(s["player_id"]) == o["position"]
    assert same / total > 0.8


def test_age_weight_lowers_mean_age(rec):
    clubs = rec.clubs["club_id"].head(30)
    def mean_age(w):
        ages = [o["age"] for c in clubs for o in rec.recommend(int(c), k=10, age_weight=w)]
        return sum(ages) / len(ages)
    assert mean_age(1.0) < mean_age(0.0) - 1.0


def test_config_endpoint_matches_recommend_defaults(rec):
    from fastapi.testclient import TestClient

    from api.main import app
    client = TestClient(app)
    cfg = client.get("/config").json()
    out = client.get("/recommend", params={"club_id": int(rec.clubs["club_id"].iloc[0])}).json()
    assert out["age_weight"] == cfg["age_weight"] and out["mobility_weight"] == cfg["mobility_weight"]
