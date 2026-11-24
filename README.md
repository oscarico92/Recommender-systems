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

   It opens http://localhost:8501. Pick a club, adjust the budget, the position and the strategy (resale, youth, frequent movers). The page has three tabs:

   - **Recommendations**: player cards with photo, club crest, market value and its 12-month trend, and a "Scouting report" (radar of the player's statistics compared with players of the same position, value curve over 5 years, key stats). A table view is also available.
   - **New talents**: players unknown to the graph, ranked by the cold-start encoder.
   - **Shortlist and compare**: the shortlisted players, and a radar comparison of 2 or 3 players.

   "Shortlist" and "Dismiss" are saved through `/log_interaction` into `data/scoutai.db` (SQLite). After a successful save, both recommendation lists refresh and the player disappears from the suggestions.

   Photos and club crests are loaded by the browser from Transfermarkt. Without internet, the cards show the player's initials and the rest of the app works the same.

   ![ScoutAI interface](docs/figures/app_screenshot.png)

   The budget is compared with estimated market values, not transfer fees. If too few of the model's candidates fit the budget, more expensive players complete the list; the page marks them "Above budget". The explanation panel distinguishes recruitment similarity from profile comparisons for new talents.

   If a request fails, the page shows a retry message. A failed history request does not mean that the database is empty. If saving an action times out, check the history before clicking again: the server may have saved it before the connection was lost.

To reach the app from another machine on the same network, start the API with `--host 0.0.0.0`, then run `SCOUTAI_API_URL=http://<api-ip>:8000 streamlit run app/streamlit_app.py`.

Run the tests with `python -m pytest -q`.

For a rehearsal with a separate database, set `SCOUTAI_DB_PATH` when starting the API (for example `SCOUTAI_DB_PATH=data/rehearsal.db uvicorn api.main:app --port 8000` on macOS/Linux). Use a new filename for a fresh rehearsal. The model stays the same, but previous test clicks do not affect the lists.

The system walkthrough and demo steps are in [docs/systems_engineer_notes.md](docs/systems_engineer_notes.md). The API and page tests use temporary databases and do not add actions to your demo database.

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
| 5b. Export the player profiles shown in the app (no training) | `python -m src.export_profiles` | `models/serving/player_profiles.parquet`, `value_history.parquet` |
| 6. Retrain with the actions logged in the app (optional) | `python -m src.export_model --feedback` | `models/serving/`, see [data/README.md](data/README.md#5-app-feedback) |

The player features (`src/features.py`) are built automatically at each cutoff and cached in `data/processed/`. Data rules and limits are described in [data/README.md](data/README.md).

## Approach

- **Data**: the open [Transfermarkt dataset](https://github.com/dcaribou/transfermarkt-datasets) (CC0, snapshot of 6 July 2026). 175,165 transfers are cleaned into 26,369 recruitments by 322 clubs of 10 leagues (loans flagged, returns from loan, retirements, youth and reserve moves removed).
- **Base model**: LightGCN (He et al., SIGIR 2020, [arXiv:2002.02126](https://arxiv.org/abs/2002.02126)) on the bipartite club / player graph, PyTorch Geometric implementation, BPR loss with uniform negative sampling.
- **Cold-start encoder**: an MLP maps 48 player features (age, position, height, market value and its trend, minutes, goals and assists per 90, European games, league, nationality) to the LightGCN embedding space, trained by distillation on the players that are in the graph. It scores players who have never been transferred.
- **Two lists**: the main list is ranked by LightGCN. The "new talents" list ranks, with the encoder, the players unknown to the graph (no recruitment by a club of the 10 leagues in the last 10 years). The user can also give a few places of the main list to new talents (`cold_slots`, 0 by default because it lowered NDCG@10 on the validation season).
- **Re-ranking**: the 50 best candidates are filtered by the club's budget (90th percentile of the values it recruited in the last 3 years, or the user's value), then re-ordered using the 12-month market value trend and penalties for age and frequent past moves. Weights are chosen on validation over 3 seeds and stored in `eval/results/best_rerank.json`.
- **Explanations**: for the main list, past signings in the same position come first and are ordered by embedding similarity; other positions fill any remaining places. For new talents, the comparison uses position and age at signing, with a small embedding-similarity term. These are reference players to help read the list, not a causal explanation of the model's ranking.
- **Feedback loop**: "Shortlist" and "Dismiss" are logged by `/log_interaction`. At once, `/recommend` hides these players and moves the club vector toward the shortlisted ones and away from the dismissed ones (Rocchio update), so similar players move up without retraining. On real recruitments of the validation season this does not improve NDCG@10 (0.0094 with and without), so it adapts the list to the scout rather than making it more accurate. Retraining with the feedback is also possible (`python -m src.export_model --feedback`) but its effect is within the run-to-run noise (see [data/README.md](data/README.md#5-app-feedback)).

## Evaluation

- **Temporal split**: train on everything before 1 July 2024, tune on season 2024/25, test on season 2025/26 (cutoff 1 July 2025). The test model is retrained on everything before the test cutoff with the tuned settings. Features are computed with data strictly before each cutoff (tested in `tests/test_features_leakage.py`), the graph only contains interactions before it, and the test season is never used for tuning.
- **Candidates**: every active player at the cutoff (a market valuation in the previous year, 36 or younger), about 18,000 players. Players the club already recruited or already has in its squad are excluded.
- **Metrics**: NDCG@10 and Recall@20 against random, most popular and highest market value rankings; cold-start NDCG@10 (targets unknown to the graph), coverage, share of the top 10 within budget, market value change of the top 10 in the following year.
- **Ablation**: LightGCN alone, feature encoder alone, LightGCN + cold-start encoder, full model with re-ranking.

## Results (test season 2025/26)

Full tables: [eval/results/test_results.md](eval/results/test_results.md). Mean ± standard deviation over 3 seeds, 322 clubs, 17,628 candidate players, 2,990 target recruitments.

| Model | NDCG@10 | Recall@20 | Within budget | Value change of top 10 | Mean age of top 10 |
| --- | --- | --- | --- | --- | --- |
| Random | 0.0004 | 0.0016 | 78% | +0.07 | 25.7 |
| Most popular (all history) | 0.0018 | 0.0026 | 80% | -0.46 | 33.1 |
| Most popular (last 2 seasons) | 0.0017 | 0.0038 | 41% | -0.06 | 24.6 |
| Highest market value | 0.0003 | 0.0009 | 0% | -0.17 | 23.4 |
| LightGCN alone | 0.0098 ± 0.0014 | 0.0173 ± 0.0036 | 76% | -0.21 | 29.1 |
| Feature encoder alone | 0.0094 ± 0.0001 | 0.0161 ± 0.0036 | 81% | -0.23 | 30.0 |
| LightGCN + budget re-ranking | 0.0108 ± 0.0017 | 0.0238 ± 0.0037 | **98%** | -0.20 | 29.5 |
| LightGCN + budget + resale (w=1) | 0.0125 ± 0.0027 | 0.0239 ± 0.0008 | **98%** | -0.11 | 28.4 |
| LightGCN + tuned re-ranking (resale 1, age 2, mobility 0) | 0.0162 ± 0.0024 | 0.0249 ± 0.0017 | **98%** | +0.04 | 25.6 |
| *Players actually signed* | | | | *+0.19* | *24.5* |

New talents list (players unknown to the graph, 60% of the catalogue):

| Model | NDCG@10 on their recruitments |
| --- | --- |
| Random | 0.0006 |
| Highest market value | 0.0066 |
| **Cold-start encoder** | **0.0325 ± 0.0022** |

What it shows:

- **Beats the sanity floor.** LightGCN reaches 5 times the NDCG@10 of both most popular rankings (all history or last 2 seasons), with clearly separated standard deviations, and recommends 70 times more distinct players (coverage 5.7% against 0.08%).
- **Budget compliance.** The budget re-ranking raises the share of affordable players from 76% to 98%.
- **Cold start.** The encoder finds new talents about 4.9 times better than "highest market value" and 54 times better than random.
- **Accuracy on validation.** NDCG@10 is similar within one seed standard deviation: 0.0138 ± 0.0012 with resale alone (w=1), versus 0.0144 ± 0.0005 with the selected weights. The higher test NDCG@10 is an observation on three seeds, not a claimed accuracy gain.
- **Age and value change.** The selected top 10 gains +0.04 in the following year, versus -0.11 with resale alone and +0.19 for players actually signed. Mean age falls from 28.4 to 25.6, still above the 24.5 of actual signings. The mobility penalty was tested but not selected by validation.
- Absolute values are low because each club signs about 10 players among 17,628 candidates: a random ranking scores 0.0004.

- **Run-to-run instability.** Across 322 clubs at the test cutoff, two seeds share 35.0% of the players in their top 10 on average. The lists shown in the demo and the case studies therefore depend on the training run. The committed model in `models/serving/` is the one used for the showcase. Tracked in issue oscarico92/Recommender-systems#5.

Known limits: new talents lists can include established stars who simply have not moved in 10 years. See the [seed-stability diagnostic](eval/results/seed_stability.md), [case studies](eval/results/case_studies.md) and [training curves](docs/figures/loss_curves.png).

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
