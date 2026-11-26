"""SQLite storage for the app: interactions logged by the app, and accounts.

Each interaction row is one action of a club user on a player shown by
ScoutAI. These rows are added to the training data of the next run (week 4).

Each account is a scout of one club: the club is chosen when the account is
created. Passwords are stored as salted PBKDF2-SHA256 hashes, never in clear.
This is a demo-grade login (no session tokens on the API side).
"""

import hashlib
import hmac
import os
import sqlite3
from pathlib import Path

from src.config import DB_PATH

ACTIONS = ("viewed", "shortlisted", "dismissed", "signed", "restored")

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
CREATE TABLE IF NOT EXISTS accounts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT    NOT NULL,      -- hex PBKDF2-SHA256
    salt          TEXT    NOT NULL,      -- hex, 16 random bytes
    club_id       INTEGER NOT NULL,
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""

PBKDF2_ITERATIONS = 200_000


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    # Upgrade legacy CHECK constraints without dropping accounts or action history.
    definition = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'interactions'").fetchone()[0]
    if "'restored'" not in definition:
        try:
            conn.execute("BEGIN IMMEDIATE")
            definition = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'interactions'").fetchone()[0]
            if "'restored'" not in definition:
                conn.execute("ALTER TABLE interactions RENAME TO interactions_legacy")
                conn.execute("DROP INDEX IF EXISTS idx_interactions_club")
                conn.execute("DROP INDEX IF EXISTS idx_interactions_player")
                for statement in SCHEMA.split(";"):
                    if statement.strip():
                        conn.execute(statement)
                conn.execute("INSERT INTO interactions SELECT * FROM interactions_legacy")
                conn.execute("DROP TABLE interactions_legacy")
            conn.commit()
        except Exception:
            conn.rollback()
            conn.close()
            raise
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


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS).hex()


def create_account(conn: sqlite3.Connection, username: str, password: str, club_id: int) -> dict | None:
    """Create an account. Returns None if the username is already taken."""
    salt = os.urandom(16)
    try:
        conn.execute(
            "INSERT INTO accounts (username, password_hash, salt, club_id) VALUES (?, ?, ?, ?)",
            (username, _hash_password(password, salt), salt.hex(), club_id),
        )
    except sqlite3.IntegrityError:
        return None
    conn.commit()
    return {"username": username, "club_id": club_id}


def check_account(conn: sqlite3.Connection, username: str, password: str) -> dict | None:
    """The account if the password is right, None otherwise."""
    row = conn.execute("SELECT * FROM accounts WHERE username = ?", (username,)).fetchone()
    if row is None:
        return None
    if not hmac.compare_digest(row["password_hash"], _hash_password(password, bytes.fromhex(row["salt"]))):
        return None
    return {"username": row["username"], "club_id": row["club_id"]}


def update_account_club(conn: sqlite3.Connection, username: str, password: str, club_id: int) -> dict | None:
    """Change a scout's club after checking the current password."""
    account = check_account(conn, username, password)
    if account is None:
        return None
    conn.execute("UPDATE accounts SET club_id = ? WHERE username = ?", (club_id, account["username"]))
    conn.commit()
    return {"username": account["username"], "club_id": club_id}
