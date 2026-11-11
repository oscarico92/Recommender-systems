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
| `data/processed/interactions.parquet` | One row per recruitment: `club_id` (user), `player_id` (item), `from_club_id`, `transfer_date`, `transfer_season`, `fee_eur`, `market_value_eur`, `is_loan`, `is_free`, `league_id` |
| `data/processed/clubs.parquet` | The selected clubs: `club_id`, `club_name`, `league_id`, `n_transfers` |
| `data/processed/cleaning_report.json` | Number of rows left after each step (to quote in the status reports) |

### Cleaning rules

A transfer is kept as an interaction only if it is a recruitment choice made by a first team.

| Step | Rule | Why |
| --- | --- | --- |
| 1 | Drop rows with a missing player, club or date, exact duplicates, and transfers dated after the run date | Unusable or not done yet |
| 2 | Detect loans: A to B then B to A within 760 days, both without a fee. The first move is kept with `is_loan = True`, the return is dropped | A return from loan is not a recruitment |
| 3 | Drop moves to a pseudo club (`Retired`, `Without Club`, `Unknown`, `Career break`, ...) | End of career, not a recruitment. Signing a player who was without club is kept |
| 4 | Drop moves to a youth or reserve team (`U19`, `U21`, `B`, `II`, `Castilla`, ...) and promotions from a club's own academy to its first team | Internal decisions, not market choices. A player bought from another club's academy is kept |
| 5 | Keep only recruiting clubs from the 10 selected leagues (`src/config.py`) | Good data coverage |
| 6 | Keep only clubs with at least 10 recruitments | Enough history to learn a club's habits |

The loans are detected on the full history, before the other filters, so that a loan to a club outside the selected leagues is still recognised.

### Known limitations

- **Loans are inferred.** In the public table, a loan, an end of loan and a free transfer all have `transfer_fee = 0`. A loan made permanent (option to buy) or a loan longer than 760 days is seen as a normal transfer. A player who leaves for free and comes back within two years is seen as a loan.
- **The league of a club is its current one** (`clubs.domestic_competition_id`). A club relegated before the snapshot is not in the selection, even for its older seasons.
- **`transfer_fee` is null when unknown**, so `fee_eur` must not be read as 0 in that case.

## 3. New interactions (SQLite)

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

## Tests

```bash
python -m pytest -q
```
