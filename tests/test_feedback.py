import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.feedback import to_signals


def _actions(rows):
    return pd.DataFrame(rows, columns=["id", "club_id", "player_id", "action", "created_at"])


def test_last_action_wins_and_weights():
    s = to_signals(_actions([
        (1, 10, 1, "shortlisted", "2026-11-01T10:00:00Z"),
        (2, 10, 1, "dismissed", "2026-11-01T11:00:00Z"),   # shortlisted then dismissed
        (3, 10, 2, "shortlisted", "2026-11-01T10:00:00Z"),
        (4, 10, 3, "signed", "2026-11-02T10:00:00Z"),
        (5, 10, 4, "viewed", "2026-11-02T10:00:00Z"),      # ignored
        (6, 11, 1, "shortlisted", "2026-11-02T10:00:00Z"),
    ]))
    pos = s.positives.set_index(["club_id", "player_id"])["weight"].to_dict()
    assert pos == {(10, 2): 0.5, (10, 3): 1.0, (11, 1): 0.5}
    assert list(map(tuple, s.negatives.to_numpy())) == [(10, 1)]
    assert s.n_actions == 6


def test_graph_weights_and_extra_edges():
    pytest.importorskip("torch")
    from src.graph import build_graph

    hist = pd.DataFrame({"club_id": [1, 1, 2], "player_id": [100, 101, 100],
                         "transfer_date": pd.to_datetime(["2020-01-01"] * 3), "is_loan": [False, True, False]})
    extra = pd.DataFrame({"club_id": [2, 9], "player_id": [200, 201], "weight": [0.5, 1.0]})
    g = build_graph(hist, np.array([1, 2]), np.array([100, 101, 200]), "2021-01-01",
                    loan_weight=0.5, extra_edges=extra)
    edges = {(int(c), int(g.player_ids[p])): float(w) for c, p, w in zip(g.pos_club, g.pos_player, g.pos_weight)}
    # loan halved, feedback edge added for a known club, unknown club 9 ignored
    assert edges == {(0, 100): 1.0, (0, 101): 0.5, (1, 100): 1.0, (1, 200): 0.5}


def test_dismissed_player_hidden_by_api(tmp_path):
    from api.interactions import get_db, get_known_ids
    from api.main import app
    from src.config import SERVING_DIR
    from src.db import connect

    if not (SERVING_DIR / "embeddings.npz").exists():
        pytest.skip("no exported model")
    path = tmp_path / "t.db"

    def db():
        conn = connect(path)
        try:
            yield conn
        finally:
            conn.close()

    app.dependency_overrides[get_db] = db
    app.dependency_overrides[get_known_ids] = lambda: None
    try:
        client = TestClient(app)
        first = client.get("/recommend", params={"club_id": 985, "k": 5}).json()["recommendations"][0]
        r = client.post("/log_interaction", json={"club_id": 985, "player_id": first["player_id"], "action": "dismissed"})
        assert r.status_code == 201
        ids = [x["player_id"] for x in client.get("/recommend", params={"club_id": 985, "k": 5}).json()["recommendations"]]
        assert first["player_id"] not in ids
    finally:
        app.dependency_overrides.clear()


def test_club_query_moves_neighbours_of_liked_players_up():
    from src.config import SERVING_DIR
    from src.predict import Recommender

    if not (SERVING_DIR / "embeddings.npz").exists():
        pytest.skip("no exported model")
    rec = Recommender.load()
    ci = rec._club_pos[985]
    warm = np.where(rec.in_graph)[0]
    base = rec.warm_emb[warm] @ rec.club_emb[ci]
    liked = warm[np.argsort(-base)[40]]            # a player ranked about 40
    unit = rec.warm_emb[warm] / np.linalg.norm(rec.warm_emb[warm], axis=1, keepdims=True)
    near = np.argsort(-(unit @ unit[np.searchsorted(warm, liked)]))[1:21]
    q = rec.club_query(ci, liked_ids={int(rec._player_ids[liked])})
    after = rec.warm_emb[warm] @ q
    rank = lambda s: np.argsort(np.argsort(-s))  # noqa: E731
    assert np.median(rank(base)[near] - rank(after)[near]) > 0   # neighbours move up
    assert np.allclose(rec.club_query(ci), rec.club_emb[ci])     # no feedback, no change


def test_shortlisted_player_hidden_by_api(tmp_path):
    from api.interactions import get_db, get_known_ids
    from api.main import app
    from src.config import SERVING_DIR
    from src.db import connect

    if not (SERVING_DIR / "embeddings.npz").exists():
        pytest.skip("no exported model")
    path = tmp_path / "t.db"

    def db():
        conn = connect(path)
        try:
            yield conn
        finally:
            conn.close()

    app.dependency_overrides[get_db] = db
    app.dependency_overrides[get_known_ids] = lambda: None
    try:
        client = TestClient(app)
        first = client.get("/recommend", params={"club_id": 985, "k": 5}).json()["recommendations"][0]
        client.post("/log_interaction", json={"club_id": 985, "player_id": first["player_id"], "action": "shortlisted"})
        body = client.get("/recommend", params={"club_id": 985, "k": 5}).json()
        assert first["player_id"] not in [x["player_id"] for x in body["recommendations"]]
        assert body["feedback"] == {"liked": 1, "dismissed": 0}
    finally:
        app.dependency_overrides.clear()


def test_restored_player_has_no_training_signal():
    signals = to_signals(_actions([
        (1, 10, 1, "dismissed", "2026-11-01T10:00:00Z"),
        (2, 10, 1, "restored", "2026-11-01T11:00:00Z"),
        (3, 11, 1, "dismissed", "2026-11-01T10:00:00Z"),
    ]))
    assert signals.positives.empty
    assert list(map(tuple, signals.negatives.to_numpy())) == [(11, 1)]
