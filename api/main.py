"""ScoutAI API. Run with: uvicorn api.main:app --reload

Endpoints
- GET  /health
- GET  /config                   model version and default re-ranking weights
- GET  /clubs                    clubs that can ask for recommendations
- GET  /recommend?club_id=985    top players for a club (real model)
- GET  /players?ids=1,2          player profiles: photo, stats, percentiles, value history
- POST /log_interaction          a user action on a recommended player
- GET  /interactions             logged actions (to check the database)
- POST /accounts                 create a scout account (username, password, club)
- POST /login                    sign in, returns the account's club
"""

import sqlite3
import time
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.accounts import router as accounts_router
from api.interactions import get_db
from api.interactions import router as interactions_router
from src.config import SERVING_DIR

app = FastAPI(title="ScoutAI API", version="0.3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(interactions_router)
app.include_router(accounts_router)


@lru_cache(maxsize=1)
def get_recommender():
    from src.predict import Recommender

    if not (SERVING_DIR / "embeddings.npz").exists():
        raise HTTPException(503, "No exported model. Run `python -m src.export_model` (see README).")
    return Recommender.load(SERVING_DIR)


CLUB_LOGO_URL = "https://tmssl.akamaized.net/images/wappen/head/{club_id}.png"


def club_logo_url(club_id) -> str | None:
    """Transfermarkt crest of a club (loaded by the viewer's browser)."""
    try:
        return CLUB_LOGO_URL.format(club_id=int(club_id))
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def get_profiles():
    """Player profiles and value history (python -m src.export_profiles). None if not exported."""
    import pandas as pd

    path = SERVING_DIR / "player_profiles.parquet"
    if not path.exists():
        return None
    profiles = pd.read_parquet(path).set_index("player_id")
    history = pd.read_parquet(SERVING_DIR / "value_history.parquet")
    history = {pid: g for pid, g in history.groupby("player_id")}
    return profiles, history


def _clean(value):
    """JSON-safe scalar (no numpy types, no NaN)."""
    try:
        if value != value:  # NaN
            return None
    except TypeError:
        return value
    return value.item() if hasattr(value, "item") else value


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": (SERVING_DIR / "embeddings.npz").exists()}


DEFAULT_RESALE_WEIGHT = 0.5


@app.get("/config")
def config() -> dict:
    """Defaults used by /recommend when a parameter is not given (tuned on validation)."""
    rec = get_recommender()
    return {
        "model_version": rec.config.get("model_version"),
        "resale_weight": float(rec.config.get("resale_weight", DEFAULT_RESALE_WEIGHT)),
        "age_weight": float(rec.config.get("age_weight", 0.0)),
        "mobility_weight": float(rec.config.get("mobility_weight", 0.0)),
        "cold_slots": int(rec.config.get("cold_slots", 0)),
    }


@app.get("/clubs")
def clubs() -> list[dict]:
    rec = get_recommender()
    cols = ["club_id", "club_name", "league_id", "budget_eur", "n_transfers"]
    clubs_df = rec.clubs.copy()
    clubs_df["min_value_eur"] = [rec.club_min_value(c) for c in clubs_df["club_id"]]
    out = clubs_df[cols + ["min_value_eur"]].sort_values(["league_id", "club_name"]).to_dict(orient="records")
    for c in out:
        c["logo_url"] = club_logo_url(c["club_id"])
    return out


@app.get("/players")
def players(ids: str = Query(..., description="Comma separated player ids, at most 20")) -> list[dict]:
    try:
        wanted = [int(x) for x in ids.split(",") if x.strip()][:20]
    except ValueError:
        raise HTTPException(422, "ids must be integers separated by commas")
    rec = get_recommender()
    data = get_profiles()
    meta = rec.players.set_index("player_id")
    out = []
    for pid in wanted:
        if pid not in meta.index:
            continue
        m = meta.loc[pid]
        item = {"player_id": pid, "name": m["name"], "position": m["position"],
                "sub_position": m["sub_position"], "age": round(float(m["age"]), 1),
                "country": m["country"], "current_club": m["club_name"], "league_id": m["league_id"],
                "market_value_eur": float(m["value_eur"]), "club_logo_url": club_logo_url(m["club_id"]),
                "photo_url": None, "stats": {}, "percentiles": {}, "value_history": []}
        if data is not None and pid in data[0].index:
            p = data[0].loc[pid]
            item["photo_url"] = _clean(p["image_url"])
            item["club_logo_url"] = club_logo_url(p["crest_club_id"])
            item["league_id"] = _clean(p["club_league_id"]) or item["league_id"]
            item["stats"] = {k: _clean(p[k]) for k in ["foot", "height_cm", "games", "minutes", "goals_per90",
                                                        "assists_per90", "cards_per90", "euro_games",
                                                        "top_league_minutes_share", "value_trend",
                                                        "peak_value_eur", "n_past_transfers"]}
            item["percentiles"] = {c[4:]: _clean(p[c]) for c in p.index if c.startswith("pct_")}
            h = data[1].get(pid)
            if h is not None:
                item["value_history"] = [{"date": d.strftime("%Y-%m-%d"), "value_eur": float(v)}
                                         for d, v in zip(h["date"], h["value_eur"])]
        out.append(item)
    return out


@app.get("/recommend")
def recommend(
    club_id: int,
    k: int = Query(10, ge=1, le=50),
    budget_eur: float | None = Query(None, gt=0, description="Price ceiling. Defaults to the club's usual budget (90th percentile of the values it recruited in 3 years)"),
    min_value_eur: float | None = Query(None, ge=0, description="Price floor. Defaults to the club's usual floor (25th percentile of the values it recruited in 3 years). 0 removes it"),
    resale_weight: float | None = Query(None, ge=0, le=3, description="Weight of the 12-month value trend. Defaults to the value tuned on validation"),
    age_weight: float | None = Query(None, ge=0, le=3, description="Penalty on age (corrects the age bias). Defaults to the value tuned on validation"),
    mobility_weight: float | None = Query(None, ge=0, le=3, description="Penalty on players who changed club often. Defaults to the value tuned on validation"),
    cold_slots: int | None = Query(None, ge=0, le=5, description="Places of the top 10 for players unknown to the graph"),
    position: Literal["Goalkeeper", "Defender", "Midfield", "Attack"] | None = None,
    new_talents: bool = Query(False, description="Only players unknown to the graph (no recruitment by a selected club in 10 years), ranked by the cold-start encoder"),
    conn: sqlite3.Connection = Depends(get_db),
) -> dict:
    rec = get_recommender()
    start = time.perf_counter()
    # feedback of the club in the app, used at once (no retraining needed):
    # the last action on each player counts. Shortlisted, signed and dismissed
    # players are hidden, and the club vector moves toward the liked ones and
    # away from the dismissed ones (Recommender.club_query).
    last = {}
    for player_id, action in conn.execute(
        "SELECT player_id, action FROM interactions WHERE club_id = ? AND action != 'viewed' ORDER BY id",
        (club_id,),
    ):
        last[player_id] = action
    liked = {p for p, a in last.items() if a in ("shortlisted", "signed")}
    disliked = {p for p, a in last.items() if a == "dismissed"}
    try:
        if resale_weight is None:
            resale_weight = float(rec.config.get("resale_weight", DEFAULT_RESALE_WEIGHT))
        if age_weight is None:
            age_weight = float(rec.config.get("age_weight", 0.0))
        if mobility_weight is None:
            mobility_weight = float(rec.config.get("mobility_weight", 0.0))
        items = rec.recommend(club_id, k=k, budget=budget_eur, min_value=min_value_eur, resale_weight=resale_weight,
                              age_weight=age_weight, mobility_weight=mobility_weight,
                              cold_slots=cold_slots, position=position, only_new=new_talents,
                              exclude_ids=liked | disliked, liked_ids=liked, disliked_ids=disliked)
    except KeyError:
        raise HTTPException(404, f"Unknown club_id {club_id}. See /clubs.")
    club = rec.clubs[rec.clubs["club_id"] == club_id].iloc[0]
    budget = budget_eur if budget_eur is not None else float(club["budget_eur"])
    floor = min(min_value_eur if min_value_eur is not None else rec.club_min_value(club_id), budget)
    data = get_profiles()
    meta = rec.players.set_index("player_id")["club_id"]
    for item in items:
        pid = item["player_id"]
        crest = data[0].at[pid, "crest_club_id"] if data is not None and pid in data[0].index else meta.get(pid)
        item["club_logo_url"] = club_logo_url(crest)
        if data is not None and pid in data[0].index:  # league of the club shown (fixed for loans)
            item["league_id"] = _clean(data[0].at[pid, "club_league_id"]) or item["league_id"]
        item["photo_url"] = (_clean(data[0].at[pid, "image_url"])
                             if data is not None and pid in data[0].index else None)
    return {
        "club_id": club_id,
        "club_name": club["club_name"],
        "club_logo_url": club_logo_url(club_id),
        "budget_eur": budget,
        "min_value_eur": floor,
        "model_version": rec.config.get("model_version"),
        "resale_weight": resale_weight,
        "age_weight": age_weight,
        "mobility_weight": mobility_weight,
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "feedback": {"liked": len(liked), "dismissed": len(disliked)},
        "recommendations": items,
    }
