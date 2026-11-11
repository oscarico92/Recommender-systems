import pytest
from fastapi.testclient import TestClient

from api.interactions import get_db
from api.main import app
from src.db import connect


@pytest.fixture
def client(tmp_path):
    conn = connect(tmp_path / "test.db")
    app.dependency_overrides[get_db] = lambda: conn
    yield TestClient(app)
    app.dependency_overrides.clear()
    conn.close()


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
