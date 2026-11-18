"""Recommendations from the exported model (numpy only, no PyTorch).

    from src.predict import Recommender
    rec = Recommender.load()
    rec.recommend(club_id=985, k=10)

The artifacts are created by `python -m src.export_model` in models/serving/.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SERVING_DIR
from src.rerank import merge_with_cold_slots, rerank_group

# Strength of the serving-time feedback update (see club_query). Checked in
# eval/feedback_check.py and eval/feedback_offline_check.py.
FEEDBACK_STRENGTH = 0.25


@dataclass
class Recommender:
    clubs: pd.DataFrame          # club_id, club_name, league_id, budget_eur, n_transfers
    players: pd.DataFrame        # catalogue metadata, aligned with the embeddings
    history: pd.DataFrame        # club_id, player_id, player_name, transfer_date (explanations)
    club_emb: np.ndarray
    warm_emb: np.ndarray
    cold_emb: np.ndarray
    in_graph: np.ndarray
    graph_ids: np.ndarray        # players of the graph (past signings)
    graph_emb: np.ndarray
    config: dict

    @classmethod
    def load(cls, path: Path | str = SERVING_DIR) -> "Recommender":
        path = Path(path)
        arr = np.load(path / "embeddings.npz")
        with open(path / "config.json") as f:
            config = json.load(f)
        history = pd.read_parquet(path / "history.parquet")
        history["transfer_date"] = pd.to_datetime(history["transfer_date"])
        return cls(
            clubs=pd.read_parquet(path / "clubs.parquet"),
            players=pd.read_parquet(path / "players.parquet"),
            history=history,
            club_emb=arr["club_emb"], warm_emb=arr["warm_emb"], cold_emb=arr["cold_emb"],
            in_graph=arr["in_graph"].astype(bool), graph_ids=arr["graph_ids"], graph_emb=arr["graph_emb"],
            config=config,
        )

    def __post_init__(self):
        self._club_pos = {c: i for i, c in enumerate(self.clubs["club_id"])}
        self._graph_pos = {p: i for i, p in enumerate(self.graph_ids)}
        self._player_club = self.players["club_id"].to_numpy()
        self._player_ids = self.players["player_id"].to_numpy()
        self._value = self.players["value_eur"].to_numpy(float)
        self._trend = self.players["value_trend"].to_numpy(float)
        self._position = self.players["position"].to_numpy()

    def club_query(self, ci: int, liked_ids=None, disliked_ids=None, strength: float = FEEDBACK_STRENGTH) -> np.ndarray:
        """Club vector moved toward the players it liked and away from the dismissed ones.

        Rocchio update, applied at serving time: c + strength * |c| * (mean of
        the unit embeddings of liked players - mean of the dismissed ones).
        Players similar to a shortlisted one move up at once, without retraining.
        """
        c = self.club_emb[ci]
        if strength <= 0 or not (liked_ids or disliked_ids):
            return c
        direction = np.zeros_like(c)
        for ids, sign in ((liked_ids, 1.0), (disliked_ids, -1.0)):
            if ids:
                vecs = self._unit_embeddings(ids)
                if len(vecs):
                    direction += sign * vecs.mean(0)
        return c + strength * np.linalg.norm(c) * direction

    def _unit_embeddings(self, ids) -> np.ndarray:
        mask = np.isin(self._player_ids, list(ids))
        if not mask.any():
            return np.zeros((0, self.club_emb.shape[1]))
        emb = np.where(self.in_graph[mask, None], self.warm_emb[mask], self.cold_emb[mask])
        return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    def club_list(self) -> list[dict]:
        return self.clubs.to_dict(orient="records")

    def recommend(
        self,
        club_id: int,
        k: int = 10,
        budget: float | None = None,
        resale_weight: float = 0.0,
        cold_slots: int | None = None,
        position: str | None = None,
        only_new: bool = False,
        exclude_ids: set[int] | None = None,
        liked_ids: set[int] | None = None,
        disliked_ids: set[int] | None = None,
        feedback_strength: float = FEEDBACK_STRENGTH,
    ) -> list[dict]:
        """Top k players for a club.

        only_new=True gives the "new talents" list: only players never
        transferred before the model's cutoff, ranked by the cold-start encoder.
        """
        if club_id not in self._club_pos:
            raise KeyError(f"unknown club_id {club_id}")
        ci = self._club_pos[club_id]
        club = self.clubs.iloc[ci]
        if budget is None:
            budget = float(club["budget_eur"])
        cold_slots = self.config.get("cold_slots", 0) if cold_slots is None else cold_slots
        length = max(k, 10)

        c = self.club_query(ci, liked_ids, disliked_ids, feedback_strength)
        past = self.history[self.history["club_id"] == club_id]
        banned = np.isin(self._player_ids, past["player_id"].to_numpy()) | (self._player_club == club_id)
        if position:
            banned |= self._position != position
        if exclude_ids:
            banned |= np.isin(self._player_ids, list(exclude_ids))

        s_warm = np.where(self.in_graph & ~banned, self.warm_emb @ c, -np.inf)
        s_cold = np.where(~self.in_graph & ~banned, self.cold_emb @ c, -np.inf)
        warm = rerank_group(s_warm, self._value, self._trend, budget, resale_weight, length)
        cold = rerank_group(s_cold, self._value, self._trend, budget, resale_weight, length)
        if only_new:
            ranked = cold[:k]
        else:
            ranked = merge_with_cold_slots(warm[None, :], cold[None, :], cold_slots, length)[0][:k]

        five_years = past["transfer_date"].max() - pd.Timedelta(days=5 * 365) if len(past) else None
        recent = past[past["transfer_date"] >= five_years] if len(past) else past
        recent = recent.sort_values("transfer_date", ascending=False).head(40)
        recent = recent[recent["player_id"].isin(self._graph_pos.keys())]
        out = []
        for rank, j in enumerate(ranked, start=1):
            p = self.players.iloc[j]
            is_new = not bool(self.in_graph[j])
            emb = self.cold_emb[j] if is_new else self.warm_emb[j]
            score = float(emb @ c)
            out.append({
                "rank": rank,
                "player_id": int(p["player_id"]),
                "name": p["name"],
                "position": p["position"],
                "sub_position": p["sub_position"],
                "age": round(float(p["age"]), 1),
                "country": p["country"],
                "current_club": p["club_name"],
                "league_id": p["league_id"],
                "market_value_eur": float(p["value_eur"]),
                "value_trend_12m": round(float(p["value_trend"]), 3),
                "within_budget": bool(p["value_eur"] <= budget),
                "never_transferred": is_new,
                "score": round(score, 4),
                "similar_to": self._explain(emb, recent),
            })
        return out

    def _explain(self, emb: np.ndarray, recent: pd.DataFrame, n: int = 2) -> list[dict]:
        """The club's recent signings closest to the player in embedding space."""
        if recent.empty:
            return []
        idx = [self._graph_pos[p] for p in recent["player_id"]]
        g = self.graph_emb[idx]
        sim = g @ emb / (np.linalg.norm(g, axis=1) * np.linalg.norm(emb) + 1e-9)
        best = np.argsort(-sim)[:n]
        rows = recent.iloc[best]
        return [
            {"player_id": int(r.player_id), "name": r.player_name, "year": int(r.transfer_date.year),
             "similarity": round(float(sim[b]), 3)}
            for b, r in zip(best, rows.itertuples())
        ]
