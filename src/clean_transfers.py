"""Turn raw Transfermarkt transfers into club -> player interactions.

Usage:
    python -m src.clean_transfers

Input:  data/raw/transfers.csv(.gz) and data/raw/clubs.csv(.gz)
Output: data/processed/interactions.parquet   one row per recruitment
        data/processed/clubs.parquet          selected clubs (the users)
        data/processed/cleaning_report.json   rows removed at each step

Cleaning rules (see data/README.md for the reasons):
 1. drop rows with a missing player, club or date, exact duplicates and
    transfers dated after the run date (announced but not done yet)
 2. detect loans: A -> B then B -> A within MAX_LOAN_DAYS, both with a fee
    of 0 or unknown. The first move is flagged is_loan, the return is dropped
 3. drop moves to a pseudo club (Retired, Without Club, Unknown, ...)
 4. drop moves to a youth or reserve team, and promotions from a club's own
    youth or reserve team to its first team
 5. keep only recruiting clubs from the selected leagues
 6. keep only clubs with at least MIN_CLUB_TRANSFERS recruitments

The loans are detected on the full history, before any other filter, so
that a loan to a club outside the selected leagues is still seen.
"""

import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    LEAGUES,
    MAX_LOAN_DAYS,
    MIN_CLUB_TRANSFERS,
    NON_CLUB_NAMES,
    PROCESSED_DIR,
    RAW_DIR,
)
from src.download_data import find_local_file

# Words that mark a youth or reserve team in a Transfermarkt club name,
# e.g. "Manchester United U21", "Real Madrid Castilla", "FC Porto B".
_TEAM_SUFFIX = re.compile(
    r"\b(ii|iii|b|c|u\d{2}|sub-?\d{2}|youth|yth\.?|reserves?|academy|"
    r"jgd\.?|juniors?|primavera|castilla)\b",
    flags=re.IGNORECASE,
)


def normalise_name(name: object) -> str:
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s+", " ", name).strip().lower()


def is_youth_or_reserve(name: object) -> bool:
    return bool(_TEAM_SUFFIX.search(normalise_name(name)))


def parent_name(name: object) -> str:
    """Club name without its youth or reserve suffix."""
    base = _TEAM_SUFFIX.sub(" ", normalise_name(name))
    return re.sub(r"\s+", " ", base).strip()


def load_table(table: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = find_local_file(table, raw_dir)
    if path is None:
        raise FileNotFoundError(
            f"{table} not found in {raw_dir}. Run `python -m src.download_data` first."
        )
    return pd.read_csv(path)


def detect_loans(df: pd.DataFrame, max_loan_days: int = MAX_LOAN_DAYS) -> pd.DataFrame:
    """Add is_loan (outgoing loan) and is_loan_return (end of loan) flags.

    The public table stores loans, ends of loan and free transfers all with a
    fee of 0, so a loan is recognised by its shape: the player goes from A to
    B and comes back from B to A shortly after, without a fee.
    """
    df = df.sort_values(["player_id", "transfer_date"]).reset_index(drop=True)
    nxt = df.groupby("player_id")[["from_club_id", "to_club_id", "transfer_date", "transfer_fee"]].shift(-1)

    no_fee = df["transfer_fee"].fillna(0).eq(0)
    next_no_fee = nxt["transfer_fee"].fillna(0).eq(0) & nxt["transfer_date"].notna()
    goes_back = (nxt["from_club_id"] == df["to_club_id"]) & (nxt["to_club_id"] == df["from_club_id"])
    short = (nxt["transfer_date"] - df["transfer_date"]).dt.days.le(max_loan_days)

    is_loan = (no_fee & next_no_fee & goes_back & short).to_numpy()
    df["is_loan"] = is_loan
    # The row right after a detected loan is its return.
    df["is_loan_return"] = np.concatenate([[False], is_loan[:-1]])
    return df


def clean_transfers(
    transfers: pd.DataFrame,
    clubs: pd.DataFrame,
    leagues: dict[str, str] = LEAGUES,
    min_club_transfers: int = MIN_CLUB_TRANSFERS,
    max_loan_days: int = MAX_LOAN_DAYS,
    today: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (interactions, selected clubs, report)."""
    report: dict[str, int] = {}

    def log(step: str, frame: pd.DataFrame) -> None:
        report[step] = int(len(frame))

    df = transfers.copy()
    log("0_raw", df)

    # 1. basic cleaning
    df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
    df = df.dropna(subset=["player_id", "from_club_id", "to_club_id", "transfer_date"])
    for col in ("player_id", "from_club_id", "to_club_id"):
        df[col] = df[col].astype("int64")
    df = df.drop_duplicates(subset=["player_id", "transfer_date", "from_club_id", "to_club_id"])
    cutoff = pd.Timestamp(today or date.today())
    df = df[df["transfer_date"] <= cutoff]
    log("1_basic_cleaning", df)

    # 2. loans
    df = detect_loans(df, max_loan_days)
    df = df[~df["is_loan_return"]].drop(columns="is_loan_return")
    log("2_without_loan_returns", df)

    # 3. pseudo clubs
    df = df[~df["to_club_name"].map(normalise_name).isin(NON_CLUB_NAMES)]
    log("3_without_pseudo_clubs", df)

    # 4. youth and reserve teams, internal promotions
    to_youth = df["to_club_name"].map(is_youth_or_reserve)
    from_youth = df["from_club_name"].map(is_youth_or_reserve)
    same_parent = df["from_club_name"].map(parent_name) == df["to_club_name"].map(parent_name)
    df = df[~to_youth & ~(from_youth & same_parent)]
    log("4_without_youth_and_internal_moves", df)

    # 5. selected leagues
    selected = clubs[clubs["domestic_competition_id"].isin(leagues)][
        ["club_id", "name", "domestic_competition_id"]
    ].rename(columns={"name": "club_name", "domestic_competition_id": "league_id"})
    df = df.merge(selected[["club_id", "league_id"]], left_on="to_club_id", right_on="club_id", how="inner")
    log("5_selected_leagues", df)

    # 6. active enough clubs
    counts = df.groupby("club_id").size()
    keep = counts[counts >= min_club_transfers].index
    df = df[df["club_id"].isin(keep)]
    log("6_min_club_transfers", df)

    interactions = (
        df.assign(
            fee_eur=df["transfer_fee"],
            is_free=df["transfer_fee"].eq(0) & ~df["is_loan"],
        )[
            [
                "club_id",
                "player_id",
                "from_club_id",
                "transfer_date",
                "transfer_season",
                "fee_eur",
                "market_value_in_eur",
                "is_loan",
                "is_free",
                "league_id",
            ]
        ]
        .rename(columns={"market_value_in_eur": "market_value_eur"})
        .sort_values(["transfer_date", "club_id", "player_id"])
        .reset_index(drop=True)
    )

    club_table = (
        selected[selected["club_id"].isin(keep)]
        .merge(counts.rename("n_transfers").reset_index(), on="club_id")
        .sort_values("n_transfers", ascending=False)
        .reset_index(drop=True)
    )

    report["n_clubs"] = int(interactions["club_id"].nunique())
    report["n_players"] = int(interactions["player_id"].nunique())
    report["n_loans"] = int(interactions["is_loan"].sum())
    report["first_date"] = str(interactions["transfer_date"].min().date()) if len(interactions) else None
    report["last_date"] = str(interactions["transfer_date"].max().date()) if len(interactions) else None
    return interactions, club_table, report


def main() -> int:
    transfers = load_table("transfers")
    clubs = load_table("clubs")
    interactions, club_table, report = clean_transfers(transfers, clubs)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(PROCESSED_DIR / "interactions.parquet", index=False)
    club_table.to_parquet(PROCESSED_DIR / "clubs.parquet", index=False)
    with open(PROCESSED_DIR / "cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved to {PROCESSED_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
