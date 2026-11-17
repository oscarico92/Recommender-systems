"""Cold-start encoder: player features -> LightGCN embedding space.

LightGCN cannot score a player without any transfer. The encoder is an MLP
trained to reproduce the LightGCN embedding of the players that do have
transfers (distillation, in the spirit of DropoutNet, Volkovs et al.,
NeurIPS 2017). At prediction time it gives an embedding to every player
from his features only, so cold players can be ranked with the same club
embeddings.
"""

import numpy as np
import torch
from torch import nn


class ColdStartEncoder(nn.Module):
    def __init__(self, n_features: int, dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(n_features), requires_grad=False)
        self.std = nn.Parameter(torch.ones(n_features), requires_grad=False)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


def fit_encoder(
    X: np.ndarray,
    target_emb: np.ndarray,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 0,
) -> tuple[ColdStartEncoder, list[float]]:
    """Fit the encoder on warm players (features X, LightGCN embeddings)."""
    torch.manual_seed(seed)
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(target_emb, dtype=torch.float32)
    enc = ColdStartEncoder(x.shape[1], y.shape[1])
    enc.mean.data = x.mean(0)
    enc.std.data = x.std(0).clamp_min(1e-6)
    opt = torch.optim.Adam(enc.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    for _ in range(epochs):
        enc.train()
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), 1024):
            b = perm[i: i + 1024]
            loss = nn.functional.mse_loss(enc(x[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        losses.append(total / len(x))
    enc.eval()
    return enc, losses


@torch.no_grad()
def encode(enc: ColdStartEncoder, X: np.ndarray) -> np.ndarray:
    return enc(torch.tensor(X, dtype=torch.float32)).numpy()
