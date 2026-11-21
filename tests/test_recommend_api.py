"""API checks with the committed serving model, without retraining."""

import pytest
from fastapi.testclient import TestClient

from api.interactions import get_db
from api.main import app
from src.config import SERVING_DIR
from src.db import connect

pytestmark = pytest.mark.skipif(
    not (SERVING_DIR / "embeddings.npz").exists(), reason="no exported model"
)


@pytest.fixture
def client(tmp_path):
    def db():
        conn = connect(tmp_path / "actions.db")
        try:
            yield conn
        finally:
            conn.close()

    app.dependency_overrides[get_db] = db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.parametrize("position", ["Goalkeeper", "Defender", "Midfield", "Attack"])
def test_position_filter_and_budget_flag(client, position):
    result = client.get("/recommend", params={
        "club_id": 985, "position": position, "budget_eur": 500000, "k": 5,
    })
    assert result.status_code == 200
    items = result.json()["recommendations"]
    assert len(items) == 5
    assert len({p["player_id"] for p in items}) == len(items)
    assert all(p["position"] == position for p in items)
    assert all(p["within_budget"] == (p["market_value_eur"] <= 500000) for p in items)


def test_new_talents_are_unknown_to_graph(client):
    result = client.get("/recommend", params={"club_id": 331, "new_talents": True, "k": 5})
    assert result.status_code == 200
    items = result.json()["recommendations"]
    assert len(items) == 5 and all(p["never_transferred"] for p in items)


def test_feedback_is_kept_between_requests_and_separate_for_each_club(client):
    other_before = client.get("/recommend", params={"club_id": 331}).json()["recommendations"]
    player = client.get("/recommend", params={"club_id": 985}).json()["recommendations"][0]
    saved = client.post("/log_interaction", json={
        "club_id": 985, "player_id": player["player_id"], "action": "shortlisted",
    })
    assert saved.status_code == 201
    updated = client.get("/recommend", params={"club_id": 985}).json()
    assert updated["feedback"] == {"liked": 1, "dismissed": 0}
    assert player["player_id"] not in {p["player_id"] for p in updated["recommendations"]}
    assert client.get("/interactions", params={"club_id": 985}).json()[0]["id"] == saved.json()["id"]
    other_after = client.get("/recommend", params={"club_id": 331}).json()["recommendations"]
    assert other_after == other_before


@pytest.mark.parametrize("params,status", [
    ({"club_id": 999999999}, 404),
    ({"club_id": 985, "k": 0}, 422),
    ({"club_id": 985, "budget_eur": -1}, 422),
    ({"club_id": 985, "position": "Striker"}, 422),
])
def test_invalid_requests(client, params, status):
    assert client.get("/recommend", params=params).status_code == status
