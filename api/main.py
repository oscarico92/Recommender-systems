"""ScoutAI API. Run with: uvicorn api.main:app --reload

Endpoints
- GET  /health
- GET  /clubs                    clubs that can ask for recommendations
- GET  /recommend?club_id=985    top players for a club (real model)
- POST /log_interaction          a user action on a recommended player
- GET  /interactions             logged actions (to check the database)
"""

import time
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.interactions import router as interactions_router
from src.config import SERVING_DIR

app = FastAPI(title="ScoutAI API", version="0.3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(interactions_router)


@lru_cache(maxsize=1)
def get_recommender():
    from src.predict import Recommender

    if not (SERVING_DIR / "embeddings.npz").exists():
        raise HTTPException(503, "No exported model. Run `python -m src.export_model` (see README).")
    return Recommender.load(SERVING_DIR)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": (SERVING_DIR / "embeddings.npz").exists()}


@app.get("/clubs")
def clubs() -> list[dict]:
    rec = get_recommender()
    cols = ["club_id", "club_name", "league_id", "budget_eur", "n_transfers"]
    return rec.clubs[cols].sort_values(["league_id", "club_name"]).to_dict(orient="records")


@app.get("/recommend")
def recommend(
    club_id: int,
    k: int = Query(10, ge=1, le=50),
    budget_eur: float | None = Query(None, gt=0, description="Defaults to the club's usual budget"),
    resale_weight: float = Query(0.5, ge=0, le=3, description="Weight of the 12-month value trend"),
    cold_slots: int | None = Query(None, ge=0, le=5, description="Places of the top 10 for players unknown to the graph"),
    position: Literal["Goalkeeper", "Defender", "Midfield", "Attack"] | None = None,
    new_talents: bool = Query(False, description="Only players unknown to the graph (no recruitment by a selected club in 10 years), ranked by the cold-start encoder"),
) -> dict:
    rec = get_recommender()
    start = time.perf_counter()
    try:
        items = rec.recommend(club_id, k=k, budget=budget_eur, resale_weight=resale_weight,
                              cold_slots=cold_slots, position=position, only_new=new_talents)
    except KeyError:
        raise HTTPException(404, f"Unknown club_id {club_id}. See /clubs.")
    club = rec.clubs[rec.clubs["club_id"] == club_id].iloc[0]
    return {
        "club_id": club_id,
        "club_name": club["club_name"],
        "budget_eur": budget_eur if budget_eur is not None else float(club["budget_eur"]),
        "model_version": rec.config.get("model_version"),
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "recommendations": items,
    }
