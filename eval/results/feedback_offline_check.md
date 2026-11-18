# Serving-time feedback: offline check on the validation season

Season 2024/25, model trained before 2024-07-01, 251 clubs with at least 3 recruitments. The first 2 recruitments of each club act as shortlisted players, NDCG@10 and Recall@20 are computed on its later recruitments. Mean ± std over 3 seeds. Control: the same update toward 2 random players.

| Club vector | Strength | NDCG@10 | Recall@20 | Clubs better / worse than no update (seed 0) |
| --- | --- | --- | --- | --- |
| no update | 0.0 | 0.0094 ± 0.0009 | 0.0180 ± 0.0017 | - |
| real first signings | 0.25 | 0.0094 ± 0.0005 | 0.0171 ± 0.0032 | 8 / 7 |
| random players (control) | 0.25 | 0.0089 ± 0.0004 | 0.0165 ± 0.0029 | 6 / 6 |
| real first signings | 0.5 | 0.0093 ± 0.0012 | 0.0173 ± 0.0049 | 11 / 9 |
| random players (control) | 0.5 | 0.0082 ± 0.0009 | 0.0149 ± 0.0008 | 12 / 6 |
| real first signings | 1.0 | 0.0090 ± 0.0006 | 0.0150 ± 0.0037 | 12 / 9 |
| random players (control) | 1.0 | 0.0061 ± 0.0011 | 0.0119 ± 0.0012 | 10 / 9 |
