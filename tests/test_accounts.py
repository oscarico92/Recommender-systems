"""Accounts API: sign up with a club, sign in, refused cases. Temporary database."""

import pytest
from fastapi.testclient import TestClient

from api.interactions import get_db
from api.main import app
from src.db import connect


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "accounts.db"

    def db():
        conn = connect(db_path)
        try:
            yield conn
        finally:
            conn.close()

    app.dependency_overrides[get_db] = db
    with TestClient(app) as c:
        c.db_path = db_path
        yield c
    app.dependency_overrides.pop(get_db, None)


def test_sign_up_then_sign_in(client):
    r = client.post("/accounts", json={"username": "scout_mufc", "password": "secret123", "club_id": 985})
    assert r.status_code == 201 and r.json() == {"username": "scout_mufc", "club_id": 985}
    r = client.post("/login", json={"username": "scout_mufc", "password": "secret123"})
    assert r.status_code == 200 and r.json()["club_id"] == 985


def test_password_is_not_stored_in_clear(client):
    client.post("/accounts", json={"username": "scout_mufc", "password": "secret123", "club_id": 985})
    row = connect(client.db_path).execute("SELECT * FROM accounts").fetchone()
    assert "secret123" not in row["password_hash"] and len(row["salt"]) == 32


def test_unknown_club_is_refused(client):
    from src.config import SERVING_DIR
    if not (SERVING_DIR / "clubs.parquet").exists():
        pytest.skip("no exported model")
    r = client.post("/accounts", json={"username": "scout_x", "password": "secret123", "club_id": 99999999})
    assert r.status_code == 404


def test_refused_cases(client):
    client.post("/accounts", json={"username": "scout_mufc", "password": "secret123", "club_id": 985})
    # same username, other case
    assert client.post("/accounts", json={"username": "SCOUT_MUFC", "password": "secret123",
                                          "club_id": 985}).status_code == 409
    # wrong password and unknown user give the same answer
    wrong = client.post("/login", json={"username": "scout_mufc", "password": "nope-nope"})
    unknown = client.post("/login", json={"username": "nobody", "password": "secret123"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()
    # invalid input
    assert client.post("/accounts", json={"username": "ab", "password": "secret123",
                                          "club_id": 985}).status_code == 422
    assert client.post("/accounts", json={"username": "scout_2", "password": "123",
                                          "club_id": 985}).status_code == 422
