"""Add the clubs' price floor to an exported model, without retraining.

    python -m src.export_price_floor

`python -m src.export_model` computes the floor (`min_value_eur` in
clubs.parquet) from the exact market values at signing. This script does the
same for a model already in models/serving/, using only the files there: the
value of each signing is the player's last valuation before the transfer date
(value_history.parquet, written by `python -m src.export_profiles`).

Check: computed the same way, the 90th percentile gives back the stored
budget for the median club (ratio 1.00), so the approximation is close.
Signings with no known valuation (about 16%) are left out.
"""

import sys

import numpy as np
import pandas as pd

from src.config import SERVING_DIR

FLOOR_QUANTILE = 0.25
YEARS = 3


def signing_values(history: pd.DataFrame, valuations: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Signings of the last YEARS years with the player's value at the transfer date."""
    recent = history[history["transfer_date"] >= cutoff - pd.Timedelta(days=YEARS * 365)]
    recent = recent.sort_values("transfer_date")
    vals = valuations.rename(columns={"date": "transfer_date"}).sort_values("transfer_date")
    merged = pd.merge_asof(recent, vals[["player_id", "transfer_date", "value_eur"]],
                           on="transfer_date", by="player_id", direction="backward")
    return merged.dropna(subset=["value_eur"])


def club_floors(clubs: pd.DataFrame, history: pd.DataFrame, valuations: pd.DataFrame,
                cutoff: pd.Timestamp) -> np.ndarray:
    signed = signing_values(history, valuations, cutoff)
    floor = signed.groupby("club_id")["value_eur"].quantile(FLOOR_QUANTILE)
    floor = floor.reindex(clubs["club_id"]).fillna(0.0).to_numpy(float)
    return np.minimum(floor, clubs["budget_eur"].to_numpy(float))


def main() -> int:
    clubs = pd.read_parquet(SERVING_DIR / "clubs.parquet")
    history = pd.read_parquet(SERVING_DIR / "history.parquet")
    history["transfer_date"] = pd.to_datetime(history["transfer_date"])
    valuations = pd.read_parquet(SERVING_DIR / "value_history.parquet")
    valuations["date"] = pd.to_datetime(valuations["date"])
    cutoff = pd.Timestamp(pd.read_json(SERVING_DIR / "config.json", typ="series")["cutoff"])
    clubs["min_value_eur"] = club_floors(clubs, history, valuations, cutoff)
    clubs.to_parquet(SERVING_DIR / "clubs.parquet", index=False)
    print(f"price floor added for {len(clubs)} clubs, median {clubs['min_value_eur'].median() / 1e6:.2f} M EUR")
    return 0


if __name__ == "__main__":
    sys.exit(main())
