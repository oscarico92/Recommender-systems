# Test results (season 2025/26, cutoff 2025-07-01)

322 clubs, 17628 players in the catalogue, 2990 target recruitments (90.0% in the catalogue), 10549 catalogue players unknown to the graph at the cutoff. Mean ± std over 3 seeds. Config: `{"dim": 64, "layers": 2, "lr": 0.005, "reg": 0.0001, "years": 10, "epochs": 35, "batch": 2048, "eval_every": 5, "patience": 6, "warm_mix": 0.0, "cold_slots": 0, "seed": 0}`.

## Main list (all players)

| Model | ndcg@10 | recall@20 | cold_ndcg@10 | warm_ndcg@10 | coverage@10 | within_budget@10 | value_change@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 0.0004 ± 0.0003 | 0.0016 ± 0.0006 | 0.0003 ± 0.0004 | 0.0001 ± 0.0002 | 0.1664 ± 0.0006 | 0.7846 ± 0.0081 | 0.0687 ± 0.0144 |
| Most popular | 0.0018 ± 0.0002 | 0.0026 ± 0.0002 | 0.0000 | 0.0021 ± 0.0003 | 0.0008 ± 0.0000 | 0.8048 ± 0.0006 | -0.4594 ± 0.0028 |
| Highest market value | 0.0003 | 0.0009 | 0.0013 | 0.0000 | 0.0008 | 0.0000 | -0.1700 |
| LightGCN alone | 0.0098 ± 0.0014 | 0.0173 ± 0.0036 | 0.0000 | 0.0123 ± 0.0016 | 0.0573 ± 0.0006 | 0.7587 ± 0.0020 | -0.2069 ± 0.0039 |
| Feature encoder alone | 0.0094 ± 0.0013 | 0.0198 ± 0.0031 | 0.0005 ± 0.0007 | 0.0109 ± 0.0015 | 0.0306 ± 0.0004 | 0.8065 ± 0.0247 | -0.2392 ± 0.0243 |
| LightGCN + 2 places for new players | 0.0099 ± 0.0015 | 0.0196 ± 0.0021 | 0.0044 ± 0.0009 | 0.0101 ± 0.0017 | 0.0543 ± 0.0011 | 0.7718 ± 0.0023 | -0.2035 ± 0.0095 |
| LightGCN + budget re-ranking | 0.0108 ± 0.0017 | 0.0238 ± 0.0037 | 0.0000 | 0.0136 ± 0.0020 | 0.0575 ± 0.0003 | 0.9827 ± 0.0004 | -0.1993 ± 0.0044 |
| LightGCN + budget + resale (w=0.5) | 0.0117 ± 0.0010 | 0.0243 ± 0.0023 | 0.0000 | 0.0145 ± 0.0011 | 0.0594 ± 0.0003 | 0.9827 ± 0.0004 | -0.1552 ± 0.0095 |
| LightGCN + budget + resale (w=1.0) | 0.0125 ± 0.0027 | 0.0239 ± 0.0008 | 0.0000 | 0.0156 ± 0.0027 | 0.0597 ± 0.0007 | 0.9827 ± 0.0004 | -0.1121 ± 0.0110 |

## New talents list (only players unknown to the graph at the cutoff)

These players had no recruitment by a club of the 10 leagues in the 10 years before the cutoff, so LightGCN cannot score them. NDCG@10 is computed on the recruitments of such players only.

| Model | ndcg@10 | coverage@10 | within_budget@10 | value_change@10 |
| --- | --- | --- | --- | --- |
| Random | 0.0006 ± 0.0006 | 0.1574 ± 0.0007 | 0.8742 ± 0.0049 | 0.1661 ± 0.0220 |
| Highest market value | 0.0067 | 0.0007 | 0.0165 | -0.0880 |
| Cold-start encoder (ours) | 0.0312 ± 0.0026 | 0.0185 ± 0.0003 | 0.8090 ± 0.0151 | 0.1009 ± 0.0412 |
