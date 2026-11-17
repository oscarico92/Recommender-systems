"""Player features computed at a cutoff date, without leakage.

Usage:
    python -m src.features --cutoff 2025-07-01

Every feature only uses rows dated strictly before the cutoff:
market valuations, appearances and transfers. The static profile fields
(birth date, position, foot, height, citizenship) do not change over time.
Snapshot fields of the players table that describe the present (current
club, current market value, contract end) are never used.

Output: data/processed/player_features_<cutoff>.parquet, one row per player
of the catalogue (players with a market valuation in the year before the
cutoff and at most CATALOG_MAX_AGE years old).
"""

import argparse
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from src.clean_transfers import load_table
from src.config import CATALOG_MAX_AGE, LEAGUES, PROCESSED_DIR

POSITIONS = ["Goalkeeper", "Defender", "Midfield", "Attack"]
FEET = ["right", "left", "both"]
EURO_CUPS = {"CL", "EL", "UCOL"}  # Champions League, Europa League, Conference League
TOP_COUNTRIES = 20

# Columns that are features (the rest is metadata for display).
META_COLUMNS = [
    "player_id", "name", "position", "sub_position", "age", "country",
    "club_id", "club_name", "league_id", "value_eur", "value_trend",
]


@lru_cache(maxsize=None)
def _raw(table: str) -> pd.DataFrame:
    df = load_table(table)
    for col in ("date", "date_of_birth", "transfer_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _last_before(df: pd.DataFrame, date: pd.Timestamp, by: str = "player_id") -> pd.DataFrame:
    """Last row of each player strictly before `date`."""
    sub = df[df["date"] < date]
    return sub.sort_values("date").groupby(by).tail(1).set_index(by)


def build_player_features(cutoff: str) -> pd.DataFrame:
    t = pd.Timestamp(cutoff)
    year_before = t - pd.Timedelta(days=365)
    players = _raw("players").set_index("player_id")
    vals = _raw("player_valuations")
    apps = _raw("appearances")
    transfers = _raw("transfers")

    # catalogue: valued in the last year, not too old
    recent = vals[(vals["date"] < t) & (vals["date"] >= year_before)]
    ids = pd.Index(recent["player_id"].unique(), name="player_id")
    ids = ids.intersection(players.index)
    age = (t - players.loc[ids, "date_of_birth"]).dt.days / 365.25
    ids = ids[(age.fillna(25) <= CATALOG_MAX_AGE).to_numpy()]

    f = pd.DataFrame(index=ids)
    p = players.loc[ids]

    # profile
    f["name"] = p["name"]
    f["position"] = p["position"].fillna("Missing")
    f["sub_position"] = p["sub_position"].fillna("Missing")
    f["country"] = p["country_of_citizenship"].fillna("Missing")
    f["age"] = ((t - p["date_of_birth"]).dt.days / 365.25).fillna(25.0)
    f["height_cm"] = p["height_in_cm"].where(p["height_in_cm"].between(150, 210))
    f["height_cm"] = f["height_cm"].fillna(f["height_cm"].median())

    # market value
    last = _last_before(vals, t)
    old = _last_before(vals, year_before)
    peak = vals[vals["date"] < t].groupby("player_id")["market_value_in_eur"].max()
    f["value_eur"] = last["market_value_in_eur"].reindex(ids).fillna(0)
    old_value = old["market_value_in_eur"].reindex(ids)
    f["value_trend"] = np.log1p(f["value_eur"]) - np.log1p(old_value.fillna(f["value_eur"]))
    f["peak_value_eur"] = peak.reindex(ids).fillna(f["value_eur"])

    # club at the cutoff, from the last valuation (dated, so no leakage)
    f["club_id"] = last["current_club_id"].reindex(ids)
    f["club_name"] = last["current_club_name"].reindex(ids).fillna("Unknown")
    f["league_id"] = last["player_club_domestic_competition_id"].reindex(ids).fillna("Other")

    # appearances in the last year
    a = apps[(apps["date"] < t) & (apps["date"] >= year_before) & apps["player_id"].isin(ids)]
    g = a.groupby("player_id")
    f["games"] = g.size().reindex(ids).fillna(0)
    f["minutes"] = g["minutes_played"].sum().reindex(ids).fillna(0)
    goals = g["goals"].sum().reindex(ids).fillna(0)
    assists = g["assists"].sum().reindex(ids).fillna(0)
    per90 = f["minutes"].clip(lower=90) / 90
    f["goals_per90"] = goals / per90
    f["assists_per90"] = assists / per90
    f["cards_per90"] = (g["yellow_cards"].sum() + 2 * g["red_cards"].sum()).reindex(ids).fillna(0) / per90
    league_min = a[a["competition_id"].isin(LEAGUES)].groupby("player_id")["minutes_played"].sum()
    f["top_league_minutes_share"] = (league_min.reindex(ids).fillna(0) / f["minutes"].clip(lower=1)).clip(0, 1)
    f["euro_games"] = a[a["competition_id"].isin(EURO_CUPS)].groupby("player_id").size().reindex(ids).fillna(0)

    # transfer history
    tr = transfers[transfers["transfer_date"] < t]
    f["n_past_transfers"] = tr.groupby("player_id").size().reindex(ids).fillna(0)

    f = f.reset_index()
    return f


def feature_matrix(f: pd.DataFrame, countries: list[str] | None = None) -> tuple[np.ndarray, list[str], list[str]]:
    """Numeric matrix for the models (not standardised).

    `countries` fixes the one-hot columns, so that the train and the test
    matrices have the same columns. Returns (X, column names, countries).
    """
    if countries is None:
        countries = f["country"].value_counts().head(TOP_COUNTRIES).index.tolist()
    cols: dict[str, np.ndarray] = {}
    cols["age"] = f["age"].to_numpy()
    cols["age_sq"] = (f["age"].to_numpy() - 25) ** 2
    cols["height_cm"] = f["height_cm"].to_numpy()
    cols["log_value"] = np.log1p(f["value_eur"].to_numpy())
    cols["log_peak_value"] = np.log1p(f["peak_value_eur"].to_numpy())
    cols["value_trend"] = f["value_trend"].to_numpy()
    for c in ["games", "minutes", "euro_games", "n_past_transfers"]:
        cols[f"log_{c}"] = np.log1p(f[c].to_numpy())
    for c in ["goals_per90", "assists_per90", "cards_per90", "top_league_minutes_share"]:
        cols[c] = f[c].to_numpy()
    for pos in POSITIONS:
        cols[f"pos_{pos}"] = (f["position"] == pos).to_numpy(float)
    for lg in LEAGUES:
        cols[f"league_{lg}"] = (f["league_id"] == lg).to_numpy(float)
    for ctry in countries:
        cols[f"country_{ctry}"] = (f["country"] == ctry).to_numpy(float)
    names = list(cols)
    X = np.column_stack([cols[n] for n in names]).astype(np.float32)
    return np.nan_to_num(X), names, countries


def features_path(cutoff: str):
    return PROCESSED_DIR / f"player_features_{cutoff}.parquet"


def load_or_build(cutoff: str) -> pd.DataFrame:
    path = features_path(cutoff)
    if path.exists():
        return pd.read_parquet(path)
    f = build_player_features(cutoff)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    f.to_parquet(path, index=False)
    return f


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cutoff", required=True, help="YYYY-MM-DD, features use data strictly before it")
    args = parser.parse_args(argv)
    f = build_player_features(args.cutoff)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    f.to_parquet(features_path(args.cutoff), index=False)
    print(f"{len(f)} players, {f.shape[1]} columns -> {features_path(args.cutoff)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
