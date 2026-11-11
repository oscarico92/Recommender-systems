"""Shared paths and data pipeline settings."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# SQLite database that stores new interactions logged by the app.
DB_PATH = Path(os.environ.get("SCOUTAI_DB_PATH", DATA_DIR / "scoutai.db"))

# Public mirror of the transfermarkt-datasets project (CC0, snapshot of July 2026).
DATA_BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"

# Tables needed by the project. "appearances" is large (about 1.9M rows) and is
# only needed in week 3 for the player features.
TABLES = [
    "transfers",
    "players",
    "clubs",
    "competitions",
    "player_valuations",
    "appearances",
]

# Ten well covered first tier leagues (Transfermarkt competition ids).
LEAGUES = {
    "GB1": "Premier League",
    "ES1": "LaLiga",
    "IT1": "Serie A",
    "L1": "Bundesliga",
    "FR1": "Ligue 1",
    "PO1": "Liga Portugal",
    "NL1": "Eredivisie",
    "BE1": "Jupiler Pro League",
    "TR1": "Super Lig",
    "SC1": "Scottish Premiership",
}

# A club needs at least this many transfers to be kept as a user.
MIN_CLUB_TRANSFERS = 10

# A move A -> B followed by B -> A within this delay is treated as a loan
# and its return. Loans rarely last more than two seasons.
MAX_LOAN_DAYS = 760

# Transfermarkt pseudo clubs that are not a real destination.
NON_CLUB_NAMES = {
    "retired",
    "without club",
    "unknown",
    "career break",
    "ban",
    "pause",
}
