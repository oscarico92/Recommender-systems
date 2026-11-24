"""Export the player profiles shown in the app (photos, stats, value history).

Usage:
    python -m src.export_profiles

No training: it reads the raw Transfermarkt tables and the player features
at SERVING_CUTOFF (the same catalogue as the served model) and writes:

- models/serving/player_profiles.parquet: one row per catalogue player with
  the photo URL, the stats of the year before the cutoff and their
  percentile among players of the same position (0 to 100), for the radar
- models/serving/value_history.parquet: market valuations of these players
  over the 5 years before the cutoff, for the value chart

The model and its embeddings are not touched.
"""

import sys

import pandas as pd

from src.config import SERVING_CUTOFF, SERVING_DIR
from src.features import _raw, load_or_build

# Stats shown on the radar, with the label used in the app.
RADAR = {
    "minutes": "Minutes played",
    "goals_per90": "Goals per 90",
    "assists_per90": "Assists per 90",
    "euro_games": "European games",
    "top_league_minutes_share": "Top-league minutes",
    "value_trend": "Value trend",
}
PROFILE_COLUMNS = [
    "player_id", "image_url", "foot", "height_cm", "games", "minutes", "goals_per90", "assists_per90",
    "cards_per90", "euro_games", "top_league_minutes_share", "value_trend", "peak_value_eur",
    "n_past_transfers", "club_id", "crest_club_id",
]


def crest_club_ids(features: pd.DataFrame, valuations: pd.DataFrame) -> pd.Series:
    """Club id matching the club name shown for each player, for the crest.

    In some valuations the club name and the club id disagree (a player on
    loan at AS Roma carries Bayern's id). Each club name gets the id it is
    most often paired with over all valuations, which is its own id.
    """
    pairs = valuations.dropna(subset=["current_club_name", "current_club_id"])
    best = (pairs.groupby(["current_club_name", "current_club_id"]).size()
            .reset_index(name="n").sort_values("n").groupby("current_club_name")["current_club_id"].last())
    return features["club_name"].map(best).fillna(features["club_id"])


def build_profiles(features: pd.DataFrame, players: pd.DataFrame,
                   valuations: pd.DataFrame | None = None) -> pd.DataFrame:
    f = features.copy()
    f["crest_club_id"] = crest_club_ids(f, valuations) if valuations is not None else f["club_id"]
    info = players.set_index("player_id")[["image_url", "foot"]]
    f = f.join(info, on="player_id")
    out = f[PROFILE_COLUMNS].copy()
    out["foot"] = out["foot"].fillna("unknown")
    # percentile within the same position, so a defender is compared with defenders
    for col in RADAR:
        out[f"pct_{col}"] = (f.groupby("position")[col].rank(pct=True) * 100).round(0).fillna(0)
    return out


def build_value_history(valuations: pd.DataFrame, player_ids, cutoff: str, years: int = 5) -> pd.DataFrame:
    t = pd.Timestamp(cutoff)
    v = valuations[(valuations["date"] < t) & (valuations["date"] >= t - pd.Timedelta(days=int(365.25 * years)))]
    v = v[v["player_id"].isin(set(player_ids))]
    return (v[["player_id", "date", "market_value_in_eur"]]
            .rename(columns={"market_value_in_eur": "value_eur"})
            .sort_values(["player_id", "date"])
            .reset_index(drop=True))


def main() -> int:
    feats = load_or_build(SERVING_CUTOFF)
    served = pd.read_parquet(SERVING_DIR / "players.parquet", columns=["player_id"])
    feats = feats[feats["player_id"].isin(served["player_id"])]
    profiles = build_profiles(feats, _raw("players"), _raw("player_valuations"))
    history = build_value_history(_raw("player_valuations"), profiles["player_id"], SERVING_CUTOFF)
    profiles.to_parquet(SERVING_DIR / "player_profiles.parquet", index=False)
    history.to_parquet(SERVING_DIR / "value_history.parquet", index=False)
    print(f"{len(profiles)} profiles, {len(history)} valuations -> {SERVING_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
