# Cube Memory-Seed probe-cost offline reranking

| Condition | Probe top1 ever | CEM mean ever | Top5-any ever | Top5 uniform expected ever | Gate |
| --- | --- | --- | --- | --- | --- |
| red | 8/12 | 9/12 | 11/12 | 7.400/12 | FAIL |
| blue_v2 | 7/12 | 8/12 | 10/12 | 7.200/12 | FAIL |
| yellow_v2 | 6/12 | 7/12 | 9/12 | 6.000/12 | FAIL |

## Physical-best and stored/recomputed diagnostics

| Condition | JEPA top1 agree | Spearman median | Top30 overlap median | Top30 Jaccard median | Max rank displacement | Physical-min best probe rank median | Physical-min best stored rank median | Physical-min best recomputed rank median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| red | 12/12 | 1.000000 | 30.0 | 1.000000 | 0 | 35.5 | 62.0 | 62.0 |
| blue_v2 | 12/12 | 1.000000 | 30.0 | 1.000000 | 0 | 106.5 | 116.0 | 116.0 |
| yellow_v2 | 12/12 | 1.000000 | 30.0 | 1.000000 | 0 | 66.5 | 105.5 | 105.5 |

Stored/recomputed JEPA ordering statistics are diagnostics; they do not claim full exact ordering.

A top-five mean action is **not** inferable from stored outcomes and requires a new simulator rollout.
