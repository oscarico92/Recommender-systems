"""Turn the actions logged in the app into training signals.

Usage (retrain and export the served model with the feedback):
    python -m src.export_model --feedback

The SQLite table `interactions` (see src/db.py) stores what club users do
with the recommended players. For each (club, player) pair only the last
action counts, so a player shortlisted then dismissed is a dismissal.

| Last action  | Signal                                   | Weight |
| ------------ | ---------------------------------------- | ------ |
| signed       | positive edge in the graph               | 1.0    |
| shortlisted  | positive edge in the graph               | 0.5    |
| dismissed    | hard negative for that club in BPR       | -      |
| viewed       | ignored (seeing a player says nothing)   | -      |

The weight is the probability, relative to a real transfer (1.0), that the
edge is drawn as a positive in a training step.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import DB_PATH

POSITIVE_WEIGHTS = {"signed": 1.0, "shortlisted": 0.5}


@dataclass
class FeedbackSignals:
    positives: pd.DataFrame   # club_id, player_id, weight
    negatives: pd.DataFrame   # club_id, player_id
    n_actions: int

    def summary(self) -> str:
        return (f"{self.n_actions} logged actions: {len(self.positives)} positive pairs, "
                f"{len(self.negatives)} dismissed pairs")


def load_actions(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    if not Path(db_path).exists():
        return pd.DataFrame(columns=["club_id", "player_id", "action", "created_at", "id"])
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT id, club_id, player_id, action, created_at FROM interactions", conn)


def to_signals(actions: pd.DataFrame) -> FeedbackSignals:
    acts = actions[actions["action"] != "viewed"].sort_values(["created_at", "id"])
    last = acts.groupby(["club_id", "player_id"], as_index=False).tail(1)
    pos = last[last["action"].isin(POSITIVE_WEIGHTS)].copy()
    pos["weight"] = pos["action"].map(POSITIVE_WEIGHTS)
    neg = last[last["action"] == "dismissed"]
    return FeedbackSignals(
        positives=pos[["club_id", "player_id", "weight"]].reset_index(drop=True),
        negatives=neg[["club_id", "player_id"]].reset_index(drop=True),
        n_actions=len(actions),
    )


def load_feedback(db_path: Path | str = DB_PATH) -> FeedbackSignals:
    return to_signals(load_actions(db_path))
