"""Training of LightGCN and of the cold-start encoder, and the score matrices.

`fit()` trains on the interactions before the setup's cutoff only. When
`monitor=True` (validation runs), it also computes at each check the
validation BPR loss and NDCG@10 on the setup's targets, and keeps the best
epoch. For the test run, `monitor` is False and the number of epochs comes
from the validation run, so the test targets are never looked at.
"""

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from eval.harness import EvalSetup, evaluate
from src.features import feature_matrix
from src.graph import Graph, build_graph
from src.models.coldstart import ColdStartEncoder, encode, fit_encoder
from src.models.lightgcn import ClubPlayerLightGCN
from src.rerank import merge_with_cold_slots  # noqa: F401  (re-exported)

if TYPE_CHECKING:
    from src.feedback import FeedbackSignals


@dataclass
class Config:
    dim: int = 64
    layers: int = 3
    lr: float = 5e-3
    reg: float = 1e-4
    years: int | None = 10       # only the last `years` of history in the graph
    epochs: int = 150
    batch: int = 2048
    eval_every: int = 5
    patience: int = 6            # checks without improvement before stopping
    warm_mix: float = 0.0        # share of the encoder embedding for warm players
    cold_slots: int = 2          # places of the top 10 kept for never transferred players
    loan_weight: float = 1.0     # weight of loan recruitments in the BPR loss
    hard_negative_rate: float = 0.05  # per dismissed player: share of the club's negatives drawn from them
    hard_negative_max: float = 0.3    # cap of that share
    seed: int = 0


@dataclass
class Fitted:
    config: Config
    graph: Graph
    club_emb: np.ndarray             # (C, d) aligned with setup.club_ids
    item_emb_lgcn: np.ndarray        # (N, d) catalogue, LightGCN (zeros for cold)
    item_emb_enc: np.ndarray         # (N, d) catalogue, encoder
    in_graph: np.ndarray             # (N,) catalogue player has at least one edge
    graph_player_emb: np.ndarray     # (P,) LightGCN embedding of every player node
    encoder: ColdStartEncoder
    countries: list[str]
    log: list[dict] = field(default_factory=list)
    best_epoch: int = 0


def _val_edges(setup: EvalSetup, graph: Graph, in_graph: np.ndarray, cat_to_node: np.ndarray):
    clubs, players = [], []
    for i, tg in enumerate(setup.targets):
        for j in tg:
            if in_graph[j]:
                clubs.append(i)
                players.append(cat_to_node[j])
    return torch.tensor(clubs, dtype=torch.long), torch.tensor(players, dtype=torch.long)


def _hard_negatives(feedback, graph: Graph, node_of: pd.Series) -> dict[int, np.ndarray]:
    """Club index -> player nodes the club dismissed in the app."""
    if feedback is None or feedback.negatives.empty:
        return {}
    club_pos = {c: i for i, c in enumerate(graph.club_ids)}
    out: dict[int, list[int]] = {}
    for c, p in zip(feedback.negatives["club_id"], feedback.negatives["player_id"]):
        if c in club_pos and p in node_of.index:
            out.setdefault(club_pos[c], []).append(int(node_of[p]))
    return {c: np.array(v) for c, v in out.items()}


def _swap_hard_negatives(neg, clubs, hard_neg, rate, cap, rng):
    """Replace some random negatives by players the club dismissed.

    The share grows with the number of dismissals (rate per player, capped),
    so 2 dismissals do not become half of the club's negatives.
    """
    neg = neg.clone()
    for k, c in enumerate(clubs.tolist()):
        if c in hard_neg and rng.random() < min(cap, rate * len(hard_neg[c])):
            neg[k] = int(rng.choice(hard_neg[c]))
    return neg


def fit(
    setup: EvalSetup,
    features: pd.DataFrame,
    cfg: Config,
    monitor: bool = False,
    verbose: bool = True,
    feedback: "FeedbackSignals | None" = None,
) -> Fitted:
    """Train LightGCN and the encoder.

    `feedback` (from src.feedback) adds the players shortlisted or signed in
    the app as extra positive edges, and uses the dismissed ones as hard
    negatives for their club.
    """
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    graph = build_graph(setup.history, setup.club_ids, setup.item_ids, setup.cutoff, cfg.years,
                        loan_weight=cfg.loan_weight,
                        extra_edges=feedback.positives if feedback is not None else None)
    node_of = pd.Series(np.arange(graph.n_players), index=graph.player_ids)
    hard_neg = _hard_negatives(feedback, graph, node_of)
    cat_to_node = node_of.loc[setup.item_ids].to_numpy()
    degree = np.bincount(graph.pos_player.numpy(), minlength=graph.n_players)
    in_graph = degree[cat_to_node] > 0

    model = ClubPlayerLightGCN(graph, cfg.dim, cfg.layers)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    n_edges = len(graph.pos_club)
    weights = graph.pos_weight
    weighted = weights is not None and bool((weights != 1).any())
    if monitor:
        v_club, v_pos = _val_edges(setup, graph, in_graph, cat_to_node)

    log, best, best_state, best_epoch, bad = [], -1.0, None, cfg.epochs, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        # every edge once per epoch; weighted edges (loans, feedback) count
        # in proportion to their weight in the loss, not in the sampling
        perm = torch.randperm(n_edges)
        total = 0.0
        for i in range(0, n_edges, cfg.batch):
            b = perm[i: i + cfg.batch]
            neg = torch.from_numpy(rng.integers(0, graph.n_players, len(b)))
            if hard_neg:
                neg = _swap_hard_negatives(neg, graph.pos_club[b], hard_neg, cfg.hard_negative_rate,
                                           cfg.hard_negative_max, rng)
            loss = model.bpr_loss(graph.edge_index, graph.pos_club[b], graph.pos_player[b], neg, cfg.reg,
                                  weight=weights[b] if weighted else None)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        row = {"epoch": epoch, "train_loss": total / n_edges}

        if monitor and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs):
            model.eval()
            with torch.no_grad():
                neg = torch.from_numpy(rng.integers(0, graph.n_players, len(v_club)))
                row["val_loss"] = model.bpr_loss(graph.edge_index, v_club, v_pos, neg, 0.0).item()
                ce, pe = model.embeddings(graph.edge_index)
            scores = ce.numpy() @ pe.numpy()[cat_to_node].T
            scores[:, ~in_graph] = -np.inf
            row["val_ndcg@10"] = evaluate(setup, scores)["warm_ndcg@10"]
            if verbose:
                print(f"  epoch {epoch:3d}  train {row['train_loss']:.4f}  val {row['val_loss']:.4f}  "
                      f"val warm ndcg@10 {row['val_ndcg@10']:.4f}")
            if row["val_ndcg@10"] > best:
                best, best_epoch, bad = row["val_ndcg@10"], epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    log.append(row)
                    break
        log.append(row)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ce, pe = model.embeddings(graph.edge_index)
    club_emb = ce.numpy()
    item_lgcn = pe.numpy()[cat_to_node]
    item_lgcn[~in_graph] = 0.0

    # cold-start encoder, trained on the catalogue players that are in the graph
    X, _, countries = feature_matrix(features)
    enc, _ = fit_encoder(X[in_graph], item_lgcn[in_graph], seed=cfg.seed)
    item_enc = encode(enc, X)

    return Fitted(config=cfg, graph=graph, club_emb=club_emb, item_emb_lgcn=item_lgcn,
                  item_emb_enc=item_enc, in_graph=in_graph, graph_player_emb=pe.numpy(), encoder=enc, countries=countries,
                  log=log, best_epoch=best_epoch if monitor else cfg.epochs)


def item_embeddings(fitted: Fitted, mode: str = "hybrid") -> np.ndarray:
    """Catalogue embeddings for a scoring mode.

    lightgcn: LightGCN only, cold players cannot be scored
    encoder:  feature encoder only, for every player (ablation)
    hybrid:   LightGCN for players in the graph, encoder for the cold ones
    """
    if mode == "lightgcn":
        return fitted.item_emb_lgcn
    if mode == "encoder":
        return fitted.item_emb_enc
    a = fitted.config.warm_mix
    warm = (1 - a) * fitted.item_emb_lgcn + a * fitted.item_emb_enc
    return np.where(fitted.in_graph[:, None], warm, fitted.item_emb_enc)


def scores(fitted: Fitted, mode: str = "hybrid") -> np.ndarray:
    s = fitted.club_emb @ item_embeddings(fitted, mode).T
    if mode == "lightgcn":
        s[:, ~fitted.in_graph] = -np.inf
    return s


def ranking(fitted: Fitted, setup: EvalSetup, mode: str = "hybrid", length: int = 20) -> np.ndarray:
    """Top `length` catalogue indices per club, after removing excluded players.

    In hybrid mode, warm players are ranked by LightGCN (mixed with the
    encoder if warm_mix > 0) and cold players by the encoder, then the two
    lists are merged with `cold_slots` places for cold players.
    """
    from eval.harness import mask_scores, top_k

    if mode != "hybrid":
        return top_k(mask_scores(setup, scores(fitted, mode)), length)
    s_warm = fitted.club_emb @ item_embeddings(fitted, "hybrid").T
    s_cold = fitted.club_emb @ fitted.item_emb_enc.T
    s_warm[:, ~fitted.in_graph] = -np.inf
    s_cold[:, fitted.in_graph] = -np.inf
    warm_rank = top_k(mask_scores(setup, s_warm), length)
    cold_rank = top_k(mask_scores(setup, s_cold), length)
    return merge_with_cold_slots(warm_rank, cold_rank, fitted.config.cold_slots, length)


def config_dict(cfg: Config) -> dict:
    return asdict(cfg)
