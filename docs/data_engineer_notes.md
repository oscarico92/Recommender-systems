# Data Engineer part: handover and showcase notes

Owner: Simon Gallais. Everything below is finished, tested and merged (or in the week 4 PR).

## What this part does

| Piece | File | Command |
| --- | --- | --- |
| Download the Transfermarkt tables | `src/download_data.py` | `python -m src.download_data` |
| Clean transfers into recruitments | `src/clean_transfers.py` | `python -m src.clean_transfers` |
| Player features at a cutoff | `src/features.py` | built by the training scripts |
| SQLite storage of app actions | `src/db.py` | created on first use |
| `POST /log_interaction`, `GET /interactions` | `api/interactions.py` | part of the API |
| Feedback at once in the API | `src/predict.py` (`club_query`), `api/main.py` | automatic |
| Retraining with app feedback (optional) | `src/feedback.py`, `src/export_model.py` | `python -m src.export_model --feedback` |
| Checks | `eval/feedback_check.py`, `eval/loan_weight_check.py` | see `eval/results/` |
| Tests | `tests/test_clean_transfers.py`, `test_features_leakage.py`, `test_log_interaction.py`, `test_feedback.py` | `python -m pytest -q` |

Rules, numbers and limits: [data/README.md](../data/README.md).

## Key numbers

- 175,165 raw transfers, 26,369 recruitments after cleaning, 322 clubs in 10 leagues, 9,877 players, 1999 to the snapshot of 6 July 2026.
- Biggest cuts: returns from loan (33,000 rows), youth, reserve and internal moves (42,000), clubs outside the 10 leagues (69,000).
- 31% detected loans; recent loans are flagged `possible_loan` because their return is not observed yet.
- 17,628 candidate players at the test cutoff, 48 features each.

## Questions the professor may ask, and short answers

**How do you know the cleaning is right?** Unit tests for each rule, row counts after each step, and a manual check: Manchester United's 2023 signings come out right (Mount, Onana, Højlund bought; Evans free; Sabitzer, Weghorst, Butland, Reguilón, Amrabat loans). Running on real data also revealed false positives ("B. Leverkusen", "Willem II" taken for reserve teams), fixed with a test.

**How are loans detected if the data has no loan flag?** The public table gives a fee of 0 to loans, ends of loan and free transfers. A loan is a move A to B followed by B to A within 760 days, both without fee. The return is dropped, the loan is kept with `is_loan`. Limit: a loan made permanent or longer than 760 days is seen as a transfer, and recent loans cannot be confirmed (`possible_loan`).

**Do loans count less?** No. We tested a weight of 0.5 in the BPR loss on the validation season: NDCG@10 0.0116 ± 0.0020 against 0.0106 ± 0.0025 with weight 1, within one standard deviation, so no evidence of a gain. A loan is still a recruitment decision.

**How do you avoid leakage in the features?** Every feature uses only rows dated strictly before the cutoff (valuations, appearances, transfers). The fields of `players` that describe today (current club, current value, contract end) are never used. `tests/test_features_leakage.py` adds future rows and checks the features do not change.

**Why a fixed snapshot date?** So the cleaned data does not depend on the day the script is run (asked in code review).

**What happens when a scout clicks Shortlist or Dismiss?** `/log_interaction` writes a row in SQLite (ids checked against the served model). At once, `/recommend` hides that player and moves the club vector toward the shortlisted players and away from the dismissed ones (Rocchio update). Check on 5 clubs: the 20 nearest neighbours of shortlisted players move up (median +41 ranks), those of dismissed players move down (-85), the rest does not move (+1), and 80% of the top 10 stays.

**Why not just retrain with the feedback?** It is possible (`python -m src.export_model --feedback`, shortlisted as weighted positive edges, dismissed as hard negatives), but the first check only showed memorisation: the shortlisted players themselves jumped to the top, which is useless since the club already has them. On their neighbours, the effect of retraining cannot be told apart from the noise of a new training run. Code review pointed this out; the serving-time update is the honest answer.

**Does the app get more accurate with clicks?** No, and say it plainly. Offline check on 2024/25 (251 clubs): using each club's first 2 signings as a shortlist leaves NDCG@10 on its later signings unchanged (0.0094 with and without the update), while a random shortlist lowers it (0.0089). The list adapts to the scout (similar players up, rejected ones gone), but the next real signings are often in other positions, so similarity does not predict them. Pitch wording: "the list adapts to your choices", not "it learns and gets more accurate".

**Why SQLite?** One file, no server, enough for the demo. The API opens one connection per request. For production with many users, PostgreSQL with the same table.

## Points for the rest of the team

- **Run-to-run instability**: two trainings of the served model with different seeds share only about 10 to 30% of each club's top 10 (`eval/results/feedback_check.md`, issue oscarico92/Recommender-systems#5). The averaged metrics are stable over 3 seeds, but the lists shown in the demo depend on the run. For the showcase, keep the committed model in `models/serving/` and do not retrain right before the demo.
- After a retraining with feedback, restart the API so it loads the new model.
