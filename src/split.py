"""Temporal split of the interactions.

The model only sees interactions dated strictly before the cutoff. The
targets are the recruitments made between the cutoff and the end date.
Validation: cutoff VAL_START, end TEST_START (season 2024/25).
Test: cutoff TEST_START, no end (season 2025/26 and the summer 2026 window).
"""

import pandas as pd

from src.config import PROCESSED_DIR


def load_interactions() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
    df["transfer_date"] = pd.to_datetime(df["transfer_date"])
    return df


def temporal_split(
    interactions: pd.DataFrame, cutoff: str, end: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (history before the cutoff, targets in [cutoff, end))."""
    t = pd.Timestamp(cutoff)
    history = interactions[interactions["transfer_date"] < t]
    targets = interactions[interactions["transfer_date"] >= t]
    if end is not None:
        targets = targets[targets["transfer_date"] < pd.Timestamp(end)]
    return history.copy(), targets.copy()
