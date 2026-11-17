"""/log_interaction endpoint (data capture).

Kept as a separate router so that the main API (api/main.py) can grow
without touching this file.
"""

import sqlite3
from collections.abc import Iterator
from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.config import DB_PATH, SERVING_DIR
from src.db import connect, insert_interaction, list_interactions

router = APIRouter(tags=["interactions"])


def get_db() -> Iterator[sqlite3.Connection]:
    """One SQLite connection per request, so concurrent requests do not share one."""
    conn = connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


@lru_cache(maxsize=1)
def _known_ids() -> tuple[frozenset, frozenset] | None:
    if not (SERVING_DIR / "clubs.parquet").exists():
        return None
    clubs = pd.read_parquet(SERVING_DIR / "clubs.parquet", columns=["club_id"])["club_id"]
    players = pd.read_parquet(SERVING_DIR / "players.parquet", columns=["player_id"])["player_id"]
    past = pd.read_parquet(SERVING_DIR / "history.parquet", columns=["player_id"])["player_id"]
    return frozenset(clubs.tolist()), frozenset(players.tolist()) | frozenset(past.tolist())


def get_known_ids() -> tuple[frozenset, frozenset] | None:
    """Club and player ids of the served model (None if no model is exported)."""
    return _known_ids()


class InteractionIn(BaseModel):
    club_id: int = Field(gt=0, examples=[583])
    player_id: int = Field(gt=0, examples=[342229])
    action: Literal["viewed", "shortlisted", "dismissed", "signed"]
    rank: int | None = Field(default=None, ge=1, description="Position in the list shown")
    model_version: str | None = Field(default=None, max_length=64)


class InteractionOut(InteractionIn):
    id: int
    created_at: str


@router.post("/log_interaction", response_model=InteractionOut, status_code=201)
def log_interaction(
    body: InteractionIn,
    conn: sqlite3.Connection = Depends(get_db),
    known: tuple[frozenset, frozenset] | None = Depends(get_known_ids),
) -> dict:
    if known is not None:
        clubs, players = known
        if body.club_id not in clubs:
            raise HTTPException(404, f"Unknown club_id {body.club_id}")
        if body.player_id not in players:
            raise HTTPException(404, f"Unknown player_id {body.player_id}")
    return insert_interaction(conn, **body.model_dump())


@router.get("/interactions", response_model=list[InteractionOut])
def get_interactions(
    club_id: int | None = None,
    limit: int = Query(100, ge=1, le=1000),
    conn: sqlite3.Connection = Depends(get_db),
) -> list[dict]:
    return list_interactions(conn, club_id=club_id, limit=limit)
