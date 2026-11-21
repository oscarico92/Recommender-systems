# Stability between training seeds

Test cutoff 2025-07-01; 322 clubs; config from validation; tuned re-ranking (resale 1.0, age 1.0, mobility 0.5). Same train/test split for all seeds.

Overlap is the number of shared players in two top-10 lists divided by 10, averaged across clubs. It measures list stability, not ranking quality.

| Seeds | Mean overlap | Median club overlap |
| --- | ---: | ---: |
| 0 vs 1 | 31.7% | 30.0% |
| 0 vs 2 | 31.8% | 30.0% |
| 1 vs 2 | 31.2% | 30.0% |

Mean pairwise overlap: 31.6%.

## Score gaps

Warm model score gap between ranks 10 and 11: median 0.028795, 10th percentile 0.00473976, 90th percentile 0.099026 (966 club-seed pairs). Small gaps make the boundary sensitive to changes in embeddings; this is descriptive, not a causal test.

The issue #5 figure used the served model at the 2026-07-01 cutoff. These results use the test cutoff, so they are a comparable diagnostic rather than a replication.
