# Data

The data is not versioned. Everything in `data/raw/`, `data/processed/` and the SQLite file is created by the scripts below.

## 1. Download

```bash
python -m src.download_data
```

Downloads the Transfermarkt tables (`transfers`, `players`, `clubs`, `competitions`, `player_valuations`, `appearances`) from the public mirror of [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) (CC0 licence, snapshot of 6 July 2026) into `data/raw/`. Files already present are skipped (`--force` to download again, `--tables transfers clubs` for a subset).

If the mirror is down, download the same CSV files from [Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores) and put them in `data/raw/` (`.csv` or `.csv.gz`).

## 2. Clean the transfers

```bash
python -m src.clean_transfers
```

| Output | Content |
| --- | --- |
| `data/processed/interactions.parquet` | One row per recruitment: `club_id` (user), `player_id` (item), `from_club_id`, `transfer_date`, `transfer_season`, `fee_eur`, `market_value_eur`, `is_loan`, `possible_loan`, `is_free`, `league_id` |
| `data/processed/clubs.parquet` | The selected clubs: `club_id`, `club_name`, `league_id`, `n_transfers` |
| `data/processed/cleaning_report.json` | Number of rows left after each step (to quote in the status reports) |

### Cleaning rules

A transfer is kept as an interaction only if it is a recruitment choice made by a first team.

| Step | Rule | Why |
| --- | --- | --- |
| 1 | Drop rows with a missing player, club or date, exact duplicates, and transfers dated after the snapshot (`SNAPSHOT_DATE = 2026-07-06` in `src/config.py`) | Unusable or announced but not done; the output does not depend on the day the script is run |
| 2 | Detect loans: A to B then B to A within 760 days, both without a fee. The first move is kept with `is_loan = True`, the return is dropped. A move without fee that is the player's last known move and less than 760 days before the snapshot gets `possible_loan = True` | A return from loan is not a recruitment. Recent loans cannot be confirmed yet because their return has not happened |
| 3 | Drop moves to a pseudo club (`Retired`, `Without Club`, `Unknown`, `Career break`, ...) | End of career, not a recruitment. Signing a player who was without club is kept |
| 4 | Drop moves to a youth or reserve team (`U19`, `U21`, `B`, `II`, `Castilla`, ...) and promotions from a club's own academy to its first team | Internal decisions, not market choices. A player bought from another club's academy is kept |
| 5 | Keep only recruiting clubs from the 10 selected leagues (`src/config.py`) | Good data coverage |
| 6 | Keep only clubs with at least 10 recruitments | Enough history to learn a club's habits |

The loans are detected on the full history, before the other filters, so that a loan to a club outside the selected leagues is still recognised.

### Result on the snapshot of 6 July 2026

| Step | Rows left |
| --- | --- |
| Raw transfers | 175,165 |
| 1. Basic cleaning (up to the snapshot date 2026-07-06) | 174,645 |
| 2. Without returns from loan | 141,371 |
| 3. Without pseudo clubs | 137,384 |
| 4. Without youth, reserve and internal moves | 94,991 |
| 5. Recruiting club in the 10 leagues | 26,395 |
| 6. Clubs with at least 10 recruitments | **26,369** |

Final dataset: 26,369 interactions, 322 clubs (median 85 recruitments per club), 9,877 players, from 1999-07-01 to 2026-07-02. 31% are detected loans, 5% possible loans (see below), 16.5% free transfers, and the fee is unknown for 16%. From 2019 on there are about 2,000 to 3,000 recruitments per year.

Manual check: the 2023 signings of Manchester United come out right (Mount, Onana and Højlund bought, Evans free, Sabitzer, Weghorst, Butland, Reguilón and Amrabat flagged as loans).

### Known limitations

- **Loans are inferred.** In the public table, a loan, an end of loan and a free transfer all have `transfer_fee = 0`. A loan made permanent (option to buy) or a loan longer than 760 days is seen as a normal transfer. A player who leaves for free and comes back within two years is seen as a loan.
- **The league of a club is its current one** (`clubs.domestic_competition_id`). A club relegated before the snapshot is not in the selection, even for its older seasons.
- **Recent loans are not confirmed.** A loan is only recognised once the player comes back, so for the last two seasons `is_loan` is underestimated: 28% of the 2025/26 recruitments are detected loans against about 31% before. `possible_loan` flags the recent one-way moves without fee (29% of 2025/26). It also catches real free transfers, so the true loan share of the test season is between 28% and 57%. The flag is descriptive only, see the next point.
- **Loans count as full recruitments.** The project plan said loans would be a weaker interaction. We tested it in week 4 (`python -m eval.loan_weight_check`, [results](../eval/results/loan_weight.md)), with the weight applied to the BPR loss of each loan: weight 0.5 gives NDCG@10 0.0116 ± 0.0020 on the validation season against 0.0106 ± 0.0025 with weight 1. The difference is within one standard deviation, so there is no evidence of a gain, and loans stay full recruitments (`loan_weight = 1.0` in `src/train.py`): a loan is a recruitment decision of the club.
- **`transfer_fee` is null when unknown**, so `fee_eur` must not be read as 0 in that case.

## 3. Player features

```bash
python -m src.features --cutoff 2025-07-01
```

`data/processed/player_features_<cutoff>.parquet`, built automatically by the training scripts. One row per player of the catalogue: players with a market valuation in the year before the cutoff, 36 or younger (17,628 players at 1 July 2025).

| Group | Features | Source |
| --- | --- | --- |
| Profile | age, height, position, nationality (top 20) | `players` (fields that do not change) |
| Market value | last value, peak value, log change over 12 months | `player_valuations` before the cutoff |
| Club | club and league at the cutoff | last valuation before the cutoff |
| Form, last 12 months | games, minutes, goals, assists and cards per 90, share of minutes in the 10 leagues, European games | `appearances` before the cutoff |
| History | number of past transfers | `transfers` before the cutoff |

**No leakage**: every feature only uses rows dated strictly before the cutoff. The snapshot fields of `players` that describe today (current club, current value, contract end) are never used. `tests/test_features_leakage.py` checks that adding future rows does not change the features.

## 4. New interactions (SQLite)

The app logs what a club user does with the recommended players. The database is `data/scoutai.db` (change it with the `SCOUTAI_DB_PATH` environment variable) and is created automatically.

```sql
CREATE TABLE interactions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    club_id       INTEGER NOT NULL,
    player_id     INTEGER NOT NULL,
    action        TEXT    NOT NULL,  -- viewed | shortlisted | dismissed | signed
    rank          INTEGER,           -- position in the list shown, if any
    model_version TEXT,              -- model that produced the list, if any
    created_at    TEXT    NOT NULL   -- UTC, ISO 8601
);
```

Endpoints (`uvicorn api.main:app --reload`, docs at http://localhost:8000/docs):

```bash
curl -X POST http://localhost:8000/log_interaction \
  -H "Content-Type: application/json" \
  -d '{"club_id": 583, "player_id": 342229, "action": "shortlisted", "rank": 2}'
```

```bash
curl "http://localhost:8000/interactions?club_id=583"
```

## 5. App feedback

"Shortlist" and "Dismiss" in the app are logged by `/log_interaction`. For each (club, player) pair only the last action counts. They are used in two ways.

**At once, in the API (main mechanism).** `/recommend` hides the players the club shortlisted, signed or dismissed, and moves the club vector toward the liked players and away from the dismissed ones (Rocchio update, `Recommender.club_query` in `src/predict.py`, strength 0.25). No retraining is needed, and players similar to a shortlisted one move up.

**At the next retraining (optional).**

```bash
python -m src.export_model --feedback
```

Retrains the served model with the logged actions (`src/feedback.py`) and exports it to `models/serving/` (model version `lightgcn-2026-07-01+feedback<N>`). Then restart the API.

| Last action | Use in retraining |
| --- | --- |
| signed | positive edge, weight 1 in the BPR loss (same as a real transfer) |
| shortlisted | positive edge, weight 0.5 in the BPR loss |
| dismissed | hard negative for that club: 5% of its negatives per dismissed player, at most 30% |
| viewed | ignored |

The weights only scale the loss: in the LightGCN propagation a feedback edge is a full neighbour, like a transfer.

**Check** (`python -m eval.feedback_check`, [results](../eval/results/feedback_check.md)). On 5 clubs, a scout shortlists the players ranked 30, 40 and 50 and dismisses the top 2. These players are then hidden, so moving them would only show memorisation; the check measures their 20 nearest neighbours instead (median rank change, positive = up):

| Group | Serving-time update | Retraining | New seed, no feedback |
| --- | --- | --- | --- |
| Neighbours of shortlisted players | +41 | -136 | -226 |
| Neighbours of dismissed players | -85 | -276 | -372 |
| All other players | +1 | -6 | +17 |
| Top 10 kept | 80% | 12% | 10% |

The serving-time update does what is expected and keeps most of the list. But this check is close to true by construction: the neighbours are chosen in the same embedding space as the update, so it shows that the code works, not that the players moving up are what a scout needs.

**Offline check on real recruitments** (`python -m eval.feedback_offline_check`, [results](../eval/results/feedback_offline_check.md)). Validation season 2024/25, 251 clubs: the first 2 recruitments of each club act as shortlisted players, and NDCG@10 is measured on its later recruitments.

| Club vector | NDCG@10 (3 seeds) |
| --- | --- |
| No update | 0.0094 ± 0.0009 |
| Updated toward its first 2 signings (strength 0.25) | 0.0094 ± 0.0005 |
| Updated toward 2 random players (control) | 0.0089 ± 0.0004 |

The update does not make the list better at predicting the next signings; it only hurts less than a random update. A likely reason: after signing a striker, a club looks for other positions, so its next needs are complementary rather than similar. What we can claim is that the list adapts to the scout (more players like the ones shortlisted, the rejected ones gone), not that it becomes more accurate. The retraining cannot be told apart from the noise of a new training run (the top 10 changes about as much with another seed, see issue oscarico92/Recommender-systems#5), so it is not used for the demo.

## Tests

```bash
python -m pytest -q
```
