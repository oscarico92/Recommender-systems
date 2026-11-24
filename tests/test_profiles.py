import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import SERVING_DIR
from src.export_profiles import (
    build_profiles,
    build_value_history,
    club_leagues,
    crest_club_ids,
    same_club_name,
)


def test_crest_uses_the_id_usually_paired_with_an_unambiguous_club_name():
    vals = pd.DataFrame({"current_club_name": ["AS Roma"] * 3 + ["Unknown"] * 3 + ["Alianza FC"] * 5,
                         "current_club_id": [12, 12, 27, 5, 5, 5, 1, 1, 2, 3, 4]})
    feats = pd.DataFrame({"player_id": [1, 2, 3, 4], "club_name": ["AS Roma", "Unknown", "Alianza FC", "Other"],
                          "club_id": [27, 99, 98, 97]})
    # Roma is mapped; "Unknown" is a pseudo club; Alianza FC (same id in 2 of 5) is ambiguous
    assert crest_club_ids(feats, vals, min_count=1).tolist() == [12, 99, 98, 97]


def test_same_club_name_ignores_generic_words_and_accents():
    assert same_club_name("AS Roma", "Associazione Sportiva Roma")
    assert same_club_name("Besiktas JK", "Beşiktaş Jimnastik Kulübü")
    assert not same_club_name("Olbia Calcio 1905", "Cagliari Calcio")


def test_league_follows_the_corrected_club_but_not_for_youth_teams():
    clubs = pd.DataFrame({"club_id": [12, 1269], "domestic_competition_id": ["IT1", "NL1"]})
    leagues = club_leagues(pd.Series([12, 1269, 5]), pd.Series([27, 999, 5]), pd.Series(["L1", "Other", "FR1"]),
                           clubs, names=pd.Series(["AS Roma", "PEC Zwolle U21", "Lens"]))
    assert leagues.tolist() == ["IT1", "Other", "FR1"]


def test_percentiles_are_within_position_and_history_before_cutoff():
    feats = pd.DataFrame({
        "player_id": [1, 2, 3], "position": ["Attack", "Attack", "Goalkeeper"], "club_name": ["A", "A", "B"],
        "league_id": ["FR1", "FR1", "GB1"],
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
    # a player on loan shows the crest and league of the club he plays for
    zaragoza = client.get("/players", params={"ids": "713644"}).json()
    if zaragoza and zaragoza[0]["current_club"] == "AS Roma":
        assert zaragoza[0]["league_id"] == "IT1" and zaragoza[0]["club_logo_url"].endswith("/12.png")
