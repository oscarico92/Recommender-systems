import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import SERVING_DIR
from src.export_profiles import build_profiles, build_value_history, crest_club_ids


def test_crest_uses_the_id_usually_paired_with_the_club_name():
    vals = pd.DataFrame({"current_club_name": ["AS Roma"] * 3 + ["Bayern Munich"] * 2,
                         "current_club_id": [12, 12, 27, 27, 27]})
    feats = pd.DataFrame({"player_id": [1, 2], "club_name": ["AS Roma", "Unknown club"], "club_id": [27, 99]})
    assert crest_club_ids(feats, vals).tolist() == [12, 99]


def test_percentiles_are_within_position_and_history_before_cutoff():
    feats = pd.DataFrame({
        "player_id": [1, 2, 3], "position": ["Attack", "Attack", "Goalkeeper"], "club_name": ["A", "A", "B"],
        "club_id": [5, 5, 6], "height_cm": 180, "games": 10, "minutes": [100, 900, 50], "goals_per90": [0.1, 0.5, 0],
        "assists_per90": 0, "cards_per90": 0, "euro_games": 0, "top_league_minutes_share": 0, "value_trend": 0,
        "peak_value_eur": 1e6, "n_past_transfers": 1,
    })
    players = pd.DataFrame({"player_id": [1, 2, 3], "image_url": ["u1", "u2", "u3"], "foot": ["left", None, "right"]})
    out = build_profiles(feats, players).set_index("player_id")
    assert out.loc[2, "pct_minutes"] == 100 and out.loc[1, "pct_minutes"] == 50
    assert out.loc[3, "pct_minutes"] == 100          # alone in its position
    assert out.loc[2, "foot"] == "unknown"
    vals = pd.DataFrame({"player_id": [1, 1, 1], "market_value_in_eur": [1, 2, 3],
                         "date": pd.to_datetime(["2019-01-01", "2025-01-01", "2026-08-01"])})
    hist = build_value_history(vals, [1], "2026-07-01")
    assert hist["value_eur"].tolist() == [2]


needs_model = pytest.mark.skipif(not (SERVING_DIR / "player_profiles.parquet").exists(),
                                 reason="no exported profiles")


@needs_model
def test_players_endpoint_and_media_in_recommend():
    from api.main import app

    client = TestClient(app)
    rec = client.get("/recommend", params={"club_id": 985, "k": 3}).json()
    assert rec["club_logo_url"].endswith("/985.png")
    first = rec["recommendations"][0]
    assert first["photo_url"].startswith("https://") and first["club_logo_url"].startswith("https://")
    players = client.get("/players", params={"ids": f"{first['player_id']},123456789"}).json()
    assert len(players) == 1                      # unknown ids are skipped
    p = players[0]
    assert set(p["percentiles"]) >= {"minutes", "goals_per90", "value_trend"}
    assert all(0 <= v <= 100 for v in p["percentiles"].values())
    assert p["value_history"] and all(h["date"] < "2026-07-01" for h in p["value_history"])
    assert client.get("/players", params={"ids": "abc"}).status_code == 422
