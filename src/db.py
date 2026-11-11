"""SQLite storage for the interactions logged by the app.

Each row is one action of a club user on a player shown by ScoutAI. These
rows are added to the training data of the next run (week 4).
"""

import sqlite3
from pathlib import Path

from src.config import DB_PATH

ACTIONS = ("viewed", "shortlisted", "dismissed", "signed")

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS interactions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    club_id       INTEGER NOT NULL,
    player_id     INTEGER NOT NULL,
    action        TEXT    NOT NULL CHECK (action IN ({", ".join(repr(a) for a in ACTIONS)})),
    rank          INTEGER,           -- position of the player in the list shown, if any
    model_version TEXT,              -- model that produced the list, if any
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_interactions_club ON interactions (club_id, created_at);
CREATE INDEX IF NOT EXISTS idx_interactions_player ON interactions (player_id);
"""


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def insert_interaction(
    conn: sqlite3.Connection,
    club_id: int,
    player_id: int,
    action: str,
    rank: int | None = None,
    model_version: str | None = None,
) -> dict:
    cur = conn.execute(
        "INSERT INTO interactions (club_id, player_id, action, rank, model_version) VALUES (?, ?, ?, ?, ?)",
        (club_id, player_id, action, rank, model_version),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM interactions WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


def list_interactions(conn: sqlite3.Connection, club_id: int | None = None, limit: int = 100) -> list[dict]:
    if club_id is None:
        rows = conn.execute("SELECT * FROM interactions ORDER BY id DESC LIMIT ?", (limit,))
    else:
        rows = conn.execute(
            "SELECT * FROM interactions WHERE club_id = ? ORDER BY id DESC LIMIT ?", (club_id, limit)
        )
    return [dict(r) for r in rows.fetchall()]
