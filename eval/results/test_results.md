# Test results (season 2025/26, cutoff 2025-07-01)

322 clubs, 17628 players in the catalogue, 2990 target recruitments (90.0% in the catalogue), 10549 catalogue players unknown to the graph at the cutoff. Mean ± std over 3 seeds. Config: `{"dim": 64, "layers": 2, "lr": 0.005, "reg": 0.0001, "years": 10, "epochs": 35, "batch": 2048, "eval_every": 5, "patience": 6, "warm_mix": 0.0, "cold_slots": 0, "loan_weight": 1.0, "hard_negative_rate": 0.05, "hard_negative_max": 0.3, "seed": 0}`.

## Main list (all players)

| Model | ndcg@10 | recall@20 | cold_ndcg@10 | warm_ndcg@10 | coverage@10 | within_budget@10 | value_change@10 | age@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random | 0.0004 ± 0.0003 | 0.0016 ± 0.0006 | 0.0003 ± 0.0004 | 0.0001 ± 0.0002 | 0.1664 ± 0.0006 | 0.7846 ± 0.0081 | 0.0687 ± 0.0144 | 25.7278 ± 0.0402 |
| Most popular (all history) | 0.0018 ± 0.0002 | 0.0026 ± 0.0002 | 0.0000 | 0.0021 ± 0.0003 | 0.0008 ± 0.0000 | 0.8048 ± 0.0006 | -0.4594 ± 0.0028 | 33.0637 ± 0.0363 |
| Most popular (last 2 seasons) | 0.0017 ± 0.0003 | 0.0038 ± 0.0008 | 0.0000 | 0.0025 ± 0.0003 | 0.0007 | 0.4107 ± 0.0191 | -0.0586 ± 0.0876 | 24.6487 ± 0.5278 |
| Highest market value | 0.0003 | 0.0009 | 0.0013 | 0.0000 | 0.0008 | 0.0000 | -0.1697 | 23.4343 |
| LightGCN alone | 0.0098 ± 0.0014 | 0.0173 ± 0.0036 | 0.0000 | 0.0123 ± 0.0016 | 0.0573 ± 0.0006 | 0.7587 ± 0.0020 | -0.2069 ± 0.0039 | 29.1457 ± 0.1531 |
| Feature encoder alone | 0.0094 ± 0.0001 | 0.0161 ± 0.0036 | 0.0010 ± 0.0015 | 0.0110 ± 0.0013 | 0.0307 ± 0.0007 | 0.8110 ± 0.0216 | -0.2330 ± 0.0075 | 30.0262 ± 0.1294 |
| LightGCN + 2 places for new players | 0.0099 ± 0.0017 | 0.0203 ± 0.0017 | 0.0043 ± 0.0004 | 0.0100 ± 0.0015 | 0.0543 ± 0.0006 | 0.7707 ± 0.0061 | -0.1959 ± 0.0034 | 28.9460 ± 0.1722 |
| LightGCN + budget re-ranking | 0.0108 ± 0.0017 | 0.0238 ± 0.0037 | 0.0000 | 0.0136 ± 0.0020 | 0.0575 ± 0.0003 | 0.9827 ± 0.0004 | -0.1993 ± 0.0044 | 29.5028 ± 0.1992 |
| LightGCN + budget + resale (w=0.5) | 0.0117 ± 0.0010 | 0.0243 ± 0.0023 | 0.0000 | 0.0145 ± 0.0011 | 0.0594 ± 0.0003 | 0.9827 ± 0.0004 | -0.1552 ± 0.0095 | 28.9347 ± 0.1537 |
| LightGCN + budget + resale (w=1.0) | 0.0125 ± 0.0027 | 0.0239 ± 0.0008 | 0.0000 | 0.0156 ± 0.0027 | 0.0597 ± 0.0007 | 0.9827 ± 0.0004 | -0.1121 ± 0.0110 | 28.4338 ± 0.0943 |
| LightGCN + budget + resale 0.5 + age (w=0.5) | 0.0131 ± 0.0023 | 0.0245 ± 0.0036 | 0.0000 | 0.0160 ± 0.0027 | 0.0624 ± 0.0009 | 0.9827 ± 0.0004 | -0.0667 ± 0.0113 | 27.4520 ± 0.1431 |
| LightGCN + budget + resale 0.5 + age (w=1.0) | 0.0152 ± 0.0027 | 0.0254 ± 0.0033 | 0.0000 | 0.0190 ± 0.0034 | 0.0627 ± 0.0014 | 0.9827 ± 0.0004 | -0.0063 ± 0.0174 | 26.3945 ± 0.1193 |
| LightGCN + budget + resale 0.5 + mobility (w=0.5) | 0.0127 ± 0.0021 | 0.0233 ± 0.0022 | 0.0000 | 0.0153 ± 0.0024 | 0.0681 ± 0.0005 | 0.9827 ± 0.0004 | -0.0944 ± 0.0080 | 28.1531 ± 0.1286 |
| LightGCN + budget + resale 1.0 + age 1.0, no mobility | 0.0166 ± 0.0032 | 0.0233 ± 0.0022 | 0.0000 | 0.0204 ± 0.0038 | 0.0625 ± 0.0011 | 0.9827 ± 0.0004 | -0.0019 ± 0.0150 | 26.4144 ± 0.0899 |
| LightGCN + budget + resale 1.0 + age 1.0 + mobility 0.5 | 0.0162 ± 0.0025 | 0.0237 ± 0.0019 | 0.0000 | 0.0198 ± 0.0029 | 0.0664 ± 0.0017 | 0.9827 ± 0.0004 | 0.0218 ± 0.0115 | 26.1434 ± 0.1197 |
| LightGCN + budget + tuned re-ranking (resale 1.0, age 2.0, mobility 0.0) | 0.0162 ± 0.0024 | 0.0249 ± 0.0017 | 0.0000 | 0.0205 ± 0.0027 | 0.0627 ± 0.0013 | 0.9827 ± 0.0004 | 0.0399 ± 0.0157 | 25.5848 ± 0.0921 |

Reference, players actually signed after the cutoff (2453 in the catalogue): mean age 24.5, mean value change +0.186. The whole catalogue: mean age 25.8, mean value change +0.061.

## New talents list (only players unknown to the graph at the cutoff)

These players had no recruitment by a club of the 10 leagues in the 10 years before the cutoff, so LightGCN cannot score them. NDCG@10 is computed on the recruitments of such players only.

| Model | ndcg@10 | coverage@10 | within_budget@10 | value_change@10 | age@10 |
| --- | --- | --- | --- | --- | --- |
| Random | 0.0006 ± 0.0006 | 0.1574 ± 0.0007 | 0.8742 ± 0.0049 | 0.1661 ± 0.0220 | 24.9481 ± 0.0930 |
| Highest market value | 0.0066 | 0.0007 | 0.0165 | -0.1235 | 21.8785 |
| Cold-start encoder (ours) | 0.0325 ± 0.0022 | 0.0184 ± 0.0003 | 0.8008 ± 0.0049 | 0.1121 ± 0.0257 | 24.6491 ± 0.0828 |
