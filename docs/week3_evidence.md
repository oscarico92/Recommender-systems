# Week 3: evidence for the risk re-assessment

Each answer points to a file of the repository. Numbers come from `eval/results/`.

| Question | Answer | Evidence |
| --- | --- | --- |
| Q1 Training | Yes | LightGCN trained on the full V1 data. `docs/figures/loss_curves.png`: train BPR loss goes from 0.69 to 0.01 with no NaN, validation BPR loss drops to 0.25 by epoch 10 and stays flat (0.25 to 0.28). Early stopping keeps epoch 35, the best validation NDCG@10. Log: `eval/results/val_training_log.csv`. |
| Q2 Sanity floor | Yes | Same split and harness (`eval/results/test_results.md`): LightGCN NDCG@10 0.0098 ± 0.0014 and Recall@20 0.0173 ± 0.0036 against 0.0018 / 0.0026 for most popular over all history, 0.0017 / 0.0038 for most popular over the last 2 seasons, and 0.0004 / 0.0016 for random. |
| Q3 Leakage | Yes | See the audit below. |
| Q4 Innovation proof | Yes | Own metrics, with the same model switched off as ablation (`eval/results/test_results.md`): cold-start NDCG@10 on the new talents list 0.0303 ± 0.0021 against 0.0067 for highest market value; share of the top 10 within budget 98% with the re-ranking against 76% without. The NDCG@10 differences between LightGCN variants are within one standard deviation, so no accuracy gain is claimed for the re-ranking. |
| Q5 Live system | Yes | `docs/figures/app_screenshot.png`: Streamlit calls the API, which serves the exported model (`models/serving/`), answers in 20 to 40 ms. A "Shortlist" click is written to SQLite and returned by `GET /interactions`. |
| Q6 Reproducibility | To do | Dry run on a clean virtual environment: `pip install -r requirements.txt` in 26 s, tests and API working in under 1 minute. Still needs a team member who did not write the code to follow the README on their own machine. |
| Q7 Schedule | Back on schedule | Week 2 and week 3 code tasks are done. Remaining: Q6 test, popularity bias, report and showcase. |

## Leakage audit

| Component | Rule | Where it is enforced |
| --- | --- | --- |
| Split | Train before 2024-07-01, validation 2024/25, test from 2025-07-01 | `src/split.py`, `tests/test_eval_and_rerank.py::test_temporal_split_has_no_overlap` |
| Graph | Only interactions before the cutoff (`history`) | `src/graph.py`, built from `setup.history` |
| Features | Valuations, appearances and transfers strictly before the cutoff; snapshot fields of `players` (current club, value, contract) never used | `src/features.py`, `tests/test_features_leakage.py` |
| Club budget | 90th percentile of values recruited before the cutoff | `eval/harness.py::club_budgets` |
| Resale signal | Value trend over the 12 months before the cutoff | `src/features.py` (`value_trend`) |
| Tuning | Grid and early stopping on the validation season only | `src/tune.py` |
| Test run | Retrained on everything before 2025-07-01 with the tuned epochs, test targets never read during training | `eval/run_experiments.py` (`monitor=False`) |
| Value change metric | Uses valuations after the cutoff, for evaluation only, never as an input | `eval/harness.py::build_setup` |

## Honest limits

- **Value and age**: every model list loses value in the following year (-0.11 to -0.24) while the real signings gain (+0.19). The top 10 average 28 to 30 years against 24.5 for the real signings: the model favours experienced players, who lose value. The resale bonus reduces it, an age-aware correction is planned for week 4.

- **Popularity bias**: players who change clubs often (experienced forwards) appear for several clubs. Coverage is still 70 times that of most popular, but a popularity penalty is planned for week 4.
- **Cold players**: "unknown to the graph" means no recruitment by a club of the 10 leagues in the last 10 years. The new talents list therefore also contains established players who have not moved in 10 years.
- **Mixing**: giving 2 places of the top 10 to new talents lowered NDCG@10 on the validation season, so it is off by default and the new talents are shown as a separate list.
