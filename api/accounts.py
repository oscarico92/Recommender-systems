"""Accounts: a scout signs up with a username, a password and his club, then signs in.

The app sends the user to the login page first; the club of the account is
the club the recommendations are made for.
"""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.interactions import get_db, get_known_ids
from src.db import check_account, create_account, update_account_club

router = APIRouter(tags=["accounts"])

USERNAME = r"^[A-Za-z0-9_.-]+$"


class AccountIn(BaseModel):
    username: str = Field(min_length=3, max_length=30, pattern=USERNAME, examples=["scout_mufc"])
    password: str = Field(min_length=6, max_length=128)
    club_id: int = Field(gt=0, examples=[985])


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=30)
    password: str = Field(min_length=1, max_length=128)


class AccountClubIn(BaseModel):
    password: str = Field(min_length=1, max_length=128)
    club_id: int = Field(gt=0, examples=[371])


class AccountOut(BaseModel):
    username: str
    club_id: int


@router.post("/accounts", response_model=AccountOut, status_code=201)
def sign_up(
    body: AccountIn,
    conn: sqlite3.Connection = Depends(get_db),
    known: tuple[frozenset, frozenset] | None = Depends(get_known_ids),
) -> dict:
    if known is not None and body.club_id not in known[0]:
        raise HTTPException(404, f"Unknown club_id {body.club_id}")
    account = create_account(conn, body.username, body.password, body.club_id)
    if account is None:
        raise HTTPException(409, "This username is already taken")
    return account


@router.post("/login", response_model=AccountOut)
def sign_in(body: LoginIn, conn: sqlite3.Connection = Depends(get_db)) -> dict:
    account = check_account(conn, body.username, body.password)
    if account is None:
        # same message for an unknown user and a wrong password
        raise HTTPException(401, "Wrong username or password")
    return account


@router.patch("/accounts/{username}", response_model=AccountOut)
def change_club(
    username: str,
    body: AccountClubIn,
    conn: sqlite3.Connection = Depends(get_db),
    known: tuple[frozenset, frozenset] | None = Depends(get_known_ids),
) -> dict:
    if known is not None and body.club_id not in known[0]:
        raise HTTPException(404, f"Unknown club_id {body.club_id}")
    account = update_account_club(conn, username, body.password, body.club_id)
    if account is None:
        raise HTTPException(401, "Wrong username or password")
    return account
