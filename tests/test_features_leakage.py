import pandas as pd

import src.features as feats


def _tables(with_future: bool):
    players = pd.DataFrame({"player_id": [1], "name": ["A"], "position": ["Attack"], "sub_position": ["CF"],
                            "country_of_citizenship": ["France"], "date_of_birth": pd.to_datetime(["2000-01-01"]),
                            "height_in_cm": [180]})
    vals = [("2024-01-01", 1e6), ("2025-01-01", 2e6)] + ([("2025-08-01", 50e6)] if with_future else [])
    valuations = pd.DataFrame({"player_id": 1, "date": pd.to_datetime([d for d, _ in vals]),
                               "market_value_in_eur": [v for _, v in vals], "current_club_id": 10,
                               "current_club_name": "X", "player_club_domestic_competition_id": "FR1"})
    apps = [("2025-03-01", 2)] + ([("2025-09-01", 5)] if with_future else [])
    appearances = pd.DataFrame({"player_id": 1, "date": pd.to_datetime([d for d, _ in apps]),
                                "minutes_played": 90, "goals": [g for _, g in apps], "assists": 0,
                                "yellow_cards": 0, "red_cards": 0, "competition_id": "FR1"})
    transfers = pd.DataFrame({"player_id": [1], "transfer_date": pd.to_datetime(["2025-08-15" if with_future else "2010-01-01"])})
    return {"players": players, "player_valuations": valuations, "appearances": appearances, "transfers": transfers}


def test_future_rows_do_not_change_features(monkeypatch):
    out = {}
    for with_future in (False, True):
        tables = _tables(with_future)
        monkeypatch.setattr(feats, "_raw", lambda t, tables=tables: tables[t])
        out[with_future] = feats.build_player_features("2025-07-01")
    cols = ["value_eur", "value_trend", "games", "goals_per90", "age"]
    pd.testing.assert_frame_equal(out[False][cols], out[True][cols])
    assert out[True]["value_eur"].iloc[0] == 2e6
