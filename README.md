# ScoutAI

Recommender Systems project, ING5 (ECE Paris), Group 6.

ScoutAI recommends football players to clubs. The club is the user, the player is the item, and a transfer is an interaction. A graph model (LightGCN) learns how each club recruits from more than 175,000 real transfers, and we extend it so that it can also recommend young players who have never been transferred, while respecting the club's budget.

## Team

| Member | Role | Technical ownership |
| --- | --- | --- |
| Oscar Schwartz | Project Lead | Repository, schedule, evaluation harness, graph construction and temporal split |
| Simon Gallais | Data Engineer | Data ingestion and cleaning, storage, `/log_interaction` endpoint |
| Gautier Deplanque | Lead ML Engineer | LightGCN in PyTorch Geometric, cold-start encoder |
| Mathis Leitao | ML Engineer (Ops) | Training loop, tuning, saved models, `predict()` and re-ranking |
| Hugo Bassaget | Systems Engineer | FastAPI backend, Streamlit frontend, deployment, README |

## Problem

Most clubs outside the elite have no data department. Their scouts follow the same well known names, and many good players in smaller leagues are never looked at. Existing tools filter players by position, age or market value, but they do not learn what a given club actually buys.

ScoutAI is different on three points:

1. **Collaborative**: it learns a club's recruitment habits from its own transfers and from clubs that recruit in a similar way.
2. **Cold-start players**: it can recommend young players with no transfer yet.
3. **Budget and resale aware**: recommendations stay within the club's budget and favour players whose market value is going up.

## Data

V1 dataset: the open [Transfermarkt dataset](https://github.com/dcaribou/transfermarkt-datasets) (CC0 licence, snapshot from July 2026).

| Table | Rows (approx.) | Use |
| --- | --- | --- |
| `transfers` | 175,000 | Interactions club (user) to player (item), with date and fee |
| `players` | 50,000 | Item features: position, age, nationality |
| `player_valuations` | 650,000 | Market value over time (budget and resale signals) |
| `appearances` | 1,890,000 | Performance features per season |
| `clubs` | 790 | User features: league, club value |

New interactions (player viewed, shortlisted, dismissed) are logged through `/log_interaction` into SQLite and added to the next training run.

## Approach

- **Base model**: LightGCN (He et al., SIGIR 2020, [arXiv:2002.02126](https://arxiv.org/abs/2002.02126)) on the bipartite club / player graph, trained with the BPR loss.
- **Cold-start encoder**: an MLP that maps player features to the LightGCN embedding space, so that never-transferred players can be scored.
- **Re-ranking**: budget filter and resale bonus (12-month market value trend) on the top 50 candidates.
- **Explanations**: the club's past signings closest to each recommended player.

## Evaluation

- Temporal split by season (train up to season N-1, validate on the end of N-1, test on season N). No test data in the graph, the features or the tuning.
- Metrics: NDCG@10 and Recall@20, against a random ranking and a most-popular ranking.
- Innovation metrics: cold-start NDCG@10, share of the top 10 within budget, market value change of the top 10, catalogue coverage.
- Ablation: LightGCN alone, feature MLP alone, LightGCN + cold-start encoder, full model.

## Tech stack

pandas, DuckDB, PyTorch, PyTorch Geometric, FastAPI, SQLite, Streamlit.

## Planned repository structure

```
data/        raw and processed data (not versioned, see data/README.md)
src/         preprocessing, graph construction, models, training
eval/        evaluation harness, baselines, results
api/         FastAPI backend
app/         Streamlit frontend
docs/        reports (project plan, status reports)
```

## Run locally

To be completed during week 3. The goal is: clone, create a virtual environment, `pip install -r requirements.txt`, download the data with one script, start the API and the app, all in under 10 minutes.

## Project status

See [PLAN_D_ACTION.md](PLAN_D_ACTION.md) for the action plan and the task list per member.
