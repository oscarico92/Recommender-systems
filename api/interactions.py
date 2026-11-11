"""/log_interaction endpoint (data capture).

Kept as a separate router so that the main API (api/main.py) can grow
without touching this file.
"""

import sqlite3
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from src.config import DB_PATH
from src.db import connect, insert_interaction, list_interactions

router = APIRouter(tags=["interactions"])

_conn: sqlite3.Connection | None = None


def get_db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = connect(DB_PATH)
    return _conn


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
def log_interaction(body: InteractionIn, conn: sqlite3.Connection = Depends(get_db)) -> dict:
    return insert_interaction(conn, **body.model_dump())


@router.get("/interactions", response_model=list[InteractionOut])
def get_interactions(
    club_id: int | None = None, limit: int = 100, conn: sqlite3.Connection = Depends(get_db)
) -> list[dict]:
    return list_interactions(conn, club_id=club_id, limit=min(limit, 1000))
