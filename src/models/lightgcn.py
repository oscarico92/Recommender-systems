"""LightGCN (He et al., SIGIR 2020) on the club / player graph.

Thin wrapper around the PyTorch Geometric implementation: embeddings are
propagated with LGConv over K layers and averaged, the score of a pair is
the dot product, and the model is trained with the BPR loss.
"""

import torch
from torch_geometric.nn.models import LightGCN

from src.graph import Graph


class ClubPlayerLightGCN(torch.nn.Module):
    def __init__(self, graph: Graph, dim: int = 64, layers: int = 3):
        super().__init__()
        self.n_clubs = graph.n_clubs
        self.model = LightGCN(num_nodes=graph.n_nodes, embedding_dim=dim, num_layers=layers)

    def embeddings(self, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Final (club, player) embeddings after propagation."""
        out = self.model.get_embedding(edge_index)
        return out[: self.n_clubs], out[self.n_clubs:]

    def bpr_loss(
        self,
        edge_index: torch.Tensor,
        club: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        reg: float = 1e-4,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """BPR loss. `weight` multiplies the loss of each positive edge
        (1 = a real transfer); with no weight it is the usual mean."""
        clubs, players = self.embeddings(edge_index)
        pos_score = (clubs[club] * players[pos]).sum(-1)
        neg_score = (clubs[club] * players[neg]).sum(-1)
        per_edge = -torch.nn.functional.logsigmoid(pos_score - neg_score)
        loss = per_edge.mean() if weight is None else (weight * per_edge).mean()
        # L2 on the layer-0 embeddings of the batch, as in the paper
        e0 = self.model.embedding.weight
        l2 = (e0[club].pow(2).sum() + e0[pos + self.n_clubs].pow(2).sum()
              + e0[neg + self.n_clubs].pow(2).sum()) / len(club)
        return loss + reg * l2
