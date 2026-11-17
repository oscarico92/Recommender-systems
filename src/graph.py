"""Bipartite club / player graph built from the training interactions only.

Node ids: clubs are 0..C-1, players are C..C+P-1. The player nodes are the
players seen in the history plus the catalogue players (the cold ones have
no edge, the cold-start encoder gives them an embedding).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass
class Graph:
    club_ids: np.ndarray        # (C,)
    player_ids: np.ndarray      # (P,)
    edge_index: torch.Tensor    # (2, 2E) undirected, global node ids
    pos_club: torch.Tensor      # (E,) club index
    pos_player: torch.Tensor    # (E,) player index (0..P-1)

    @property
    def n_clubs(self) -> int:
        return len(self.club_ids)

    @property
    def n_players(self) -> int:
        return len(self.player_ids)

    @property
    def n_nodes(self) -> int:
        return self.n_clubs + self.n_players


def build_graph(
    history: pd.DataFrame,
    club_ids: np.ndarray,
    catalogue_ids: np.ndarray,
    cutoff: str,
    years: int | None = None,
) -> Graph:
    """Graph of the recruitments before `cutoff`, optionally only the last `years`."""
    h = history
    if years is not None:
        h = h[h["transfer_date"] >= pd.Timestamp(cutoff) - pd.Timedelta(days=int(365.25 * years))]
    h = h[h["club_id"].isin(club_ids)].drop_duplicates(["club_id", "player_id"])

    player_ids = np.union1d(h["player_id"].unique(), catalogue_ids)
    club_idx = pd.Series(np.arange(len(club_ids)), index=club_ids)
    player_idx = pd.Series(np.arange(len(player_ids)), index=player_ids)

    c = torch.tensor(club_idx.loc[h["club_id"]].to_numpy(), dtype=torch.long)
    p = torch.tensor(player_idx.loc[h["player_id"]].to_numpy(), dtype=torch.long)
    p_node = p + len(club_ids)
    edge_index = torch.cat([torch.stack([c, p_node]), torch.stack([p_node, c])], dim=1)
    return Graph(club_ids=np.asarray(club_ids), player_ids=player_ids, edge_index=edge_index,
                 pos_club=c, pos_player=p)
