import pytest
from fastapi.testclient import TestClient

from api.interactions import get_db, get_known_ids
from api.main import app
from src.db import connect


@pytest.fixture
def client(tmp_path):
    path = tmp_path / "test.db"

    def db():
        conn = connect(path)
        try:
            yield conn
        finally:
            conn.close()

    app.dependency_overrides[get_db] = db
    app.dependency_overrides[get_known_ids] = lambda: (frozenset({583, 1}), frozenset({42, 2}))
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_log_and_read_back(client):
    r = client.post("/log_interaction", json={"club_id": 583, "player_id": 42, "action": "shortlisted", "rank": 3})
    assert r.status_code == 201
    body = r.json()
    assert body["id"] == 1 and body["action"] == "shortlisted" and body["created_at"]

    rows = client.get("/interactions", params={"club_id": 583}).json()
    assert len(rows) == 1 and rows[0]["player_id"] == 42


@pytest.mark.parametrize("payload", [
    {"club_id": 1, "player_id": 2, "action": "liked"},
    {"club_id": -1, "player_id": 2, "action": "viewed"},
    {"club_id": 1, "action": "viewed"},
])
def test_invalid_payload_rejected(client, payload):
    assert client.post("/log_interaction", json=payload).status_code == 422


def test_unknown_ids_rejected(client):
    assert client.post("/log_interaction", json={"club_id": 999, "player_id": 42, "action": "viewed"}).status_code == 404
    assert client.post("/log_interaction", json={"club_id": 583, "player_id": 999, "action": "viewed"}).status_code == 404


@pytest.mark.parametrize("limit", [-5, 0, 5000])
def test_interactions_limit_validated(client, limit):
    assert client.get("/interactions", params={"limit": limit}).status_code == 422
