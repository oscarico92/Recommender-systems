# Retraining with app feedback: does it generalise?

Serving model (cutoff 2026-07-01). For each of 5 clubs (Manchester United, Benfica, Celtic, RC Lens, Brighton), the players ranked 30, 40 and 50 are shortlisted and the top 2 are dismissed. These 5 players are then hidden from the club's list (the API does the same), so the table measures their neighbours: the 20 closest players to each of them in the embedding space of the model without feedback. Ranks are among about 6,000 warm candidates. Positive = moved up.

Three columns:

- **Serving-time update** (what the API does): the club vector moves toward the shortlisted players and away from the dismissed ones (Rocchio, strength 0.25), same model, no retraining.
- **Retraining**: shortlisted players become positive edges (weight 0.5 in the BPR loss) and dismissed players hard negatives (`python -m src.export_model --feedback`), same seed.
- **New seed**: retraining without any feedback with another seed, the run-to-run noise the retraining column must be compared with.

| Group | Players (5 clubs) | Serving-time update | Retraining | New seed, no feedback |
| --- | --- | --- | --- | --- |
| neighbours of shortlisted | 292 | +41 | -136 | -226 |
| neighbours of dismissed | 172 | -85 | -276 | -372 |
| all other players | 37255 | +1 | -6 | +17 |

Top 10 kept (mean over the clubs): serving-time update 80%, retraining 12%, new seed 10%.


Per club:

| Club | Group | Players | Serving-time update | Retraining | New seed |
| --- | --- | --- | --- | --- | --- |
| Manchester United | neighbours of shortlisted | 54 | +24 | -14 | -75 |
| Manchester United | neighbours of dismissed | 29 | -11 | -51 | -30 |
| Manchester United | all other players | 7492 | +0 | +1 | +10 |
| Benfica | neighbours of shortlisted | 60 | +84 | -134 | -484 |
| Benfica | neighbours of dismissed | 39 | -38 | -138 | -378 |
| Benfica | all other players | 7432 | -14 | +4 | -6 |
| Celtic | neighbours of shortlisted | 60 | +48 | -506 | -203 |
| Celtic | neighbours of dismissed | 32 | -80 | -357 | -118 |
| Celtic | all other players | 7443 | +11 | -2 | +15 |
| RC Lens | neighbours of shortlisted | 58 | +2 | +34 | -28 |
| RC Lens | neighbours of dismissed | 35 | -66 | -303 | -146 |
| RC Lens | all other players | 7432 | -8 | -28 | +17 |
| Brighton | neighbours of shortlisted | 60 | +46 | -62 | -340 |
| Brighton | neighbours of dismissed | 37 | -231 | -531 | -1189 |
| Brighton | all other players | 7456 | +16 | -8 | +49 |
