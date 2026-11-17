# ScoutAI

Recommender Systems project, ING5 (ECE Paris), Group 6.

ScoutAI recommends football players to clubs. The club is the user, the player is the item, and a transfer is an interaction. A graph model (LightGCN) learns how each club recruits from 26,000 real recruitments, and we extend it so that it can also recommend players who have never been transferred, while respecting the club's budget and favouring players whose value is going up.

## Team

| Member | Role | Technical ownership |
| --- | --- | --- |
| Oscar Schwartz | Project Lead | Repository, schedule, evaluation harness, graph construction and temporal split |
| Simon Gallais | Data Engineer | Data ingestion and cleaning, player features, storage, `/log_interaction` endpoint |
| Gautier Deplanque | Lead ML Engineer | LightGCN in PyTorch Geometric, cold-start encoder |
| Mathis Leitao | ML Engineer (Ops) | Training loop, tuning, saved models, `predict()` and re-ranking |
| Hugo Bassaget | Systems Engineer | FastAPI backend, Streamlit frontend, deployment, README |

## Run locally (about 5 minutes)

The trained model is versioned in `models/serving/`, so running the app needs neither the data nor PyTorch.

1. Clone the repository and enter it.

   ```bash
   git clone https://github.com/oscarico92/Recommender-systems.git
   ```

   ```bash
   cd Recommender-systems
   ```

2. Create a virtual environment and install the dependencies (Python 3.10 or later).

   ```bash
   python -m venv .venv
   ```

   ```bash
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

   ```bash
   pip install -r requirements.txt
   ```

3. Start the API (leave this terminal open).

   ```bash
   uvicorn api.main:app --port 8000
   ```

   Check it at http://localhost:8000/docs, or:

   ```bash
   curl "http://localhost:8000/recommend?club_id=985&k=5"
   ```

4. In a second terminal (same virtual environment), start the interface.

   ```bash
   streamlit run app/streamlit_app.py
   ```

   It opens http://localhost:8501. Pick a club, adjust the budget, the resale focus and the number of places for new talents. "Shortlist" and "Dismiss" are saved through `/log_interaction` into `data/scoutai.db` (SQLite) and listed at the bottom of the page.

To reach the app from another machine on the same network, start the API with `--host 0.0.0.0`, then run `SCOUTAI_API_URL=http://<api-ip>:8000 streamlit run app/streamlit_app.py`.

Run the tests with `python -m pytest -q`.

## Re-train the model (optional, about 30 minutes on a laptop CPU)

```bash
pip install -r requirements-train.txt
```

PyTorch from PyPI is large. On a machine without a GPU, `pip install torch --index-url https://download.pytorch.org/whl/cpu` first is faster.

| Step | Command | Output |
| --- | --- | --- |
| 1. Download the Transfermarkt tables | `python -m src.download_data` | `data/raw/` |
| 2. Clean the transfers | `python -m src.clean_transfers` | `data/processed/interactions.parquet`, cleaning report |
| 3. Tune on the validation season | `python -m src.tune` | `eval/results/tuning*.csv`, `best_config.json` |
| 4. Evaluate on the test season | `python -m eval.run_experiments` | `eval/results/test_results.md`, `case_studies.md`, `docs/figures/loss_curves.png` |
| 5. Export the served model | `python -m src.export_model` | `models/serving/` |

The player features (`src/features.py`) are built automatically at each cutoff and cached in `data/processed/`. Data rules and limits are described in [data/README.md](data/README.md).

## Approach

- **Data**: the open [Transfermarkt dataset](https://github.com/dcaribou/transfermarkt-datasets) (CC0, snapshot of 6 July 2026). 175,165 transfers are cleaned into 26,372 recruitments by 322 clubs of 10 leagues (loans flagged, returns from loan, retirements, youth and reserve moves removed).
- **Base model**: LightGCN (He et al., SIGIR 2020, [arXiv:2002.02126](https://arxiv.org/abs/2002.02126)) on the bipartite club / player graph, PyTorch Geometric implementation, BPR loss with uniform negative sampling.
- **Cold-start encoder**: an MLP maps 48 player features (age, position, height, market value and its trend, minutes, goals and assists per 90, European games, league, nationality) to the LightGCN embedding space, trained by distillation on the players that are in the graph. It scores players who have never been transferred.
- **Two lists**: the main list is ranked by LightGCN. The "new talents" list ranks, with the encoder, the players unknown to the graph (no recruitment by a club of the 10 leagues in the last 10 years). The user can also give a few places of the main list to new talents (`cold_slots`, 0 by default because it lowered NDCG@10 on the validation season).
- **Re-ranking**: the 50 best candidates are filtered by the club's budget (90th percentile of the values it recruited in the last 3 years, or the user's value), then re-ordered with a bonus for the 12-month market value trend.
- **Explanations**: the club's recent signings closest to the recommended player in the embedding space.

## Evaluation

- **Temporal split**: train on everything before 1 July 2024, tune on season 2024/25, test on season 2025/26 (cutoff 1 July 2025). The test model is retrained on everything before the test cutoff with the tuned settings. Features are computed with data strictly before each cutoff (tested in `tests/test_features_leakage.py`), the graph only contains interactions before it, and the test season is never used for tuning.
- **Candidates**: every active player at the cutoff (a market valuation in the previous year, 36 or younger), about 18,000 players. Players the club already recruited or already has in its squad are excluded.
- **Metrics**: NDCG@10 and Recall@20 against random, most popular and highest market value rankings; cold-start NDCG@10 (targets unknown to the graph), coverage, share of the top 10 within budget, market value change of the top 10 in the following year.
- **Ablation**: LightGCN alone, feature encoder alone, LightGCN + cold-start encoder, full model with re-ranking.

## Results (test season 2025/26)

Full tables: [eval/results/test_results.md](eval/results/test_results.md). Mean over 3 seeds, 322 clubs, 17,628 candidate players, 2,993 target recruitments.

| Model | NDCG@10 | Recall@20 | Within budget | Value change of top 10 |
| --- | --- | --- | --- | --- |
| Random | 0.0004 | 0.0016 | 78% | +0.07 |
| Most popular | 0.0018 | 0.0026 | 80% | -0.46 |
| Highest market value | 0.0003 | 0.0009 | 0% | -0.17 |
| LightGCN alone | 0.0098 | 0.0173 | 76% | -0.21 |
| Feature encoder alone | 0.0097 | 0.0197 | 81% | -0.23 |
| LightGCN + budget re-ranking | 0.0108 | 0.0238 | 98% | -0.20 |
| **LightGCN + budget + resale (w=1)** | **0.0125** | **0.0239** | **98%** | **-0.11** |

New talents list (players unknown to the graph, 60% of the catalogue):

| Model | NDCG@10 on their recruitments |
| --- | --- |
| Random | 0.0006 |
| Highest market value | 0.0066 |
| **Cold-start encoder** | **0.0319** |

What it shows:

- LightGCN is 5 times better than the most popular ranking, and recommends 70 times more distinct players (coverage 5.7% against 0.08%), so it does not just push the same famous names.
- The budget re-ranking keeps 98% of the recommendations affordable and also improves accuracy, because clubs mostly sign players they can afford. The resale bonus raises NDCG@10 again and halves the value loss of the recommended players.
- The cold-start encoder finds new talents 5 times better than "highest market value" and 50 times better than random.
- Absolute values are low because each club signs about 10 players among 17,628 candidates: a random ranking scores 0.0004.

Known limits, to work on in week 4: a popularity bias toward players who often change clubs (the same experienced forwards appear for several clubs, see [case studies](eval/results/case_studies.md)), and new talents lists that include established stars who simply have not moved in 10 years. Training curves: [docs/figures/loss_curves.png](docs/figures/loss_curves.png).

## Repository structure

```
api/         FastAPI backend (/recommend, /clubs, /log_interaction)
app/         Streamlit frontend
data/        raw and processed data (not versioned, see data/README.md)
docs/        figures and reports
eval/        evaluation harness, baselines, experiments and results
models/      exported model served by the API (versioned)
src/         data pipeline, features, graph, models, training, tuning, prediction
tests/       unit tests
```

## Project status

See [PLAN_D_ACTION.md](PLAN_D_ACTION.md) for the action plan and the task list per member.
