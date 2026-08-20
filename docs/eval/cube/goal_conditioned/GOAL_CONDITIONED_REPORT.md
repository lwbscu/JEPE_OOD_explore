# Cube Goal-Conditioned Retrieval Report

Date: 2026-08-15

## Executive conclusion

目标条件过滤在离线层面有效，但没有超过现有 T2 在线基线。

- **G1-T2:** Red/Blue-v2/Yellow-v2 = `86/84/86%`，宏平均 `85.33%`；相对 T2 的 `88/88/86%` 为 `-2/-4/0 pp`，宏平均 `-2.00 pp`。
- **G1-T1:** `82/80/80%`，宏平均 `80.67%`；相对原 T1 的 `72/70/70%`，三色均 `+10 pp`，且逐 env 没有相对原 T1 的反向翻转；但相对 T2 仍为 `-6/-8/-6 pp`。
- 六个离线单元均通过 `median E_roll <= 40 mm` 门禁。G1-T2 的三色中位想象误差均低于对应 T2，G1-T1 仍维持约 `11–13 mm` 的低误差。

因此，锚点与目标不对齐是 T1 失败的重要因素，但不是唯一因素。目标对齐使全锚热启动明显恢复，却仍不及保留 270 条自由探索候选的 T2；覆盖多样性具有独立必要性。生成层 §A 的当前定版应保留 **T2：10 条专家种子 + 20 条小扰动 + 270 条自由探索**，而不是 G1-T2 或全锚 G1-T1。

## Frozen protocol

- Checkpoint: MaskedAug `weights_final.pt`, SHA256 `d64501aa8e7dac1205d3a134c5bd7c160361e16d6da54c79e21e974cdc953117`.
- CEM: 300 samples, top-30 update, 10 iterations, horizon/receding-horizon 5, action block 5, legacy updated elite-mean selector, seed 42.
- Retrieval query: the existing normalized 9D current-state feature.
- Goal: the fixed privileged block XYZ at the formal episode's `row+25`; the same goal is used for every replan in that env.
- Candidate neighborhood: exact top-100 raw anchors after globally excluding all 50 formal evaluation episodes, with closed-ball tie completion and stable `(distance, dataset_row)` ordering.
- Alignment score: `||block_pos(row)-goal|| - ||block_pos(row+25)-goal||`, computed in float64; strict positive scores are accepted first.
- Selection: positive-score anchors in state-distance order, with 10 distinct source episodes; missing slots are filled from the same unfiltered top-100 while preserving distinct episodes.
- G1-T2 changes retrieval only relative to T2: 10 selected seeds, 20 fixed per-cycle raw-action-domain Gaussian perturbations (`sigma=0.1`, clipped to `[-1,1]`), 270 free samples.
- G1-T1 changes retrieval only relative to T1: nearest selected aligned seed as the full initial mean, initial sampling std `0.2`, no persistent injected slots.

### Protocol-difference disclosure

The prior T2 evaluator excluded only the current evaluation episode. The present task explicitly required every selected source episode to be outside the full fixed-50 set, so G1 globally excludes all 50. Old T2 first-cycle captures had no cross-evaluation-episode hit, but one later Red replan used another formal episode. Consequently, G1-vs-T2 is overwhelmingly a goal-filter comparison at the frozen first-cycle gate, but the full online delta also includes this small leakage-discipline correction and should not be attributed exclusively to goal filtering.

## Offline gate and alignment

Each cell was captured in full 50-env solver order, retaining the fixed 12 pools. Every new 12x300 pool received fresh MuJoCo endpoint truth and Masked-probe scoring; no legacy candidate labels were joined by candidate index.

The first-cycle retrieval statistics are identical across protocols and colors because the current states, goals, and retriever are paired:

- Raw positive anchors: `2887/5000 = 57.74%`.
- Selected aligned seed slots: `463/500 = 92.60%`.
- Fallback slots: `37/500 = 7.40%`; 8/50 queries needed fallback.
- All selected sources are from distinct episodes outside the fixed-50 evaluation set; all `row+25` endpoints are valid.

| Protocol | Color | Median / p90 / p95 E_roll (mm) | E_roll >40 mm | Same-color reference median | Delta vs reference | Gate |
|---|---|---:|---:|---:|---:|---:|
| G1-T2 | Red | 19.296 / 173.255 / 232.344 | 33.56% | T2 21.115 | -1.819 mm | PASS |
| G1-T2 | Blue-v2 | 21.247 / 154.860 / 217.319 | 33.67% | T2 22.672 | -1.425 mm | PASS |
| G1-T2 | Yellow-v2 | 20.035 / 180.255 / 230.534 | 36.14% | T2 23.143 | -3.108 mm | PASS |
| G1-T1 | Red | 11.450 / 20.777 / 24.513 | 3.56% | T1 10.763 | +0.687 mm | PASS |
| G1-T1 | Blue-v2 | 13.195 / 24.636 / 27.094 | 2.17% | T1 11.910 | +1.285 mm | PASS |
| G1-T1 | Yellow-v2 | 11.538 / 22.060 / 25.649 | 3.61% | T1 10.692 | +0.846 mm | PASS |

G1-T2 lowers the median relative to T2 on all colors, but retains a large `>40 mm` tail. G1-T1 remains tightly on-manifold. The offline `57.74%` raw-positive rate is well above the requested 30% sparsity diagnostic; the pool is not too sparse at the first cycle.

## Online matrix

All rows use the same 50 formal envs. Earlier rows are frozen prior results and were not rerun.

| Method | Red | Blue-v2 | Yellow-v2 | Macro average |
|---|---:|---:|---:|---:|
| Original mean | 72% | 64% | 62% | 66.00% |
| Memory Seed | 88% | 68% | 66% | 74.00% |
| Route2 global ColorAug | 80% | 64% | 78% | 74.00% |
| MaskedAug mean | 74% | 76% | 74% | 74.67% |
| Seed x ColorAug | **92%** | 68% | 80% | 80.00% |
| Seed x MaskedAug | 84% | 86% | 84% | 84.67% |
| T1 state-only full anchor | 72% | 70% | 70% | 70.67% |
| **T2 state-only mixed pool** | 88% | **88%** | **86%** | **87.33%** |
| **G1-T2 goal-filtered mixed pool** | 86% | 84% | **86%** | 85.33% |
| **G1-T1 goal-filtered full anchor** | 82% | 80% | 80% | 80.67% |

G1-T2 does not exceed the T2 macro record: `85.33% < 87.33%`. G1-T1 improves its matched T1 control by `+10 pp` on every color, but does not fully revive to T2 level.

## Paired env flips versus T2

Notation: `F->S` means T2 failure becomes G1 success; `S->F` means T2 success becomes G1 failure. IDs are zero-based formal env indices, with dataset row in parentheses.

### G1-T2

- Red F->S: none; S->F: `27(1294136)`.
- Blue-v2 F->S: none; S->F: `27(1294136), 38(1570913)`.
- Yellow-v2 F->S: none; S->F: none.

### G1-T1

- Red F->S: `11(556268), 14(808839), 17(891249)`; S->F: `0(128267), 6(257500), 12(712588), 34(1523727), 35(1529852), 46(1795165)`.
- Blue-v2 F->S: `11(556268), 14(808839), 17(891249)`; S->F: `0(128267), 6(257500), 12(712588), 27(1294136), 34(1523727), 35(1529852), 46(1795165)`.
- Yellow-v2 F->S: `11(556268), 14(808839), 17(891249), 38(1570913)`; S->F: `0(128267), 6(257500), 12(712588), 27(1294136), 34(1523727), 35(1529852), 46(1795165)`.

## G1-T1 diagnostic versus original T1

Goal filtering produced `+10 pp` in every color with no paired regression relative to original T1:

- Red F->S: `5(189273), 10(456735), 16(882105), 23(1058181), 43(1725130)`; S->F: none.
- Blue-v2 F->S: `5(189273), 10(456735), 23(1058181), 33(1478821), 43(1725130)`; S->F: none.
- Yellow-v2 F->S: `5(189273), 10(456735), 23(1058181), 33(1478821), 43(1725130)`; S->F: none.

This is strong paired evidence that source-goal mismatch materially hurt the original full-anchor T1 arm. The remaining `6.67 pp` macro gap to T2 shows that selecting a better single anchor still cannot replace broad candidate coverage.

## Interpretation and section-A decision

### a. Did goal alignment improve the best macro result?

No. G1-T2 reduced median imagination error by `1.4–3.1 mm` relative to T2, yet online success fell by `2/4/0 pp`. This is another direct counterexample to using pool median imagination error as a sufficient planner-quality objective. The G1-T2 paired changes contain no F->S flips and three S->F flips.

### b. Did G1-T1 revive?

Partially and materially. It gained 10 pp in every color over T1 with no paired regression, so state-only anchor mismatch was a major T1 failure source. It still reached only `82/80/80%`, below T2's `88/88/86%`. The results are consistent with two necessary ingredients: target relevance and coverage diversity. They do not show that either factor alone is sufficient.

### c. Final generation-layer conclusion

For the current experiment matrix, retain **T2 state-only mixed-pool planning** as the default generation-layer method: it remains the best single-policy macro result (`87.33%`) and the best Blue-v2/Yellow-v2 result (`88/86%`). Goal filtering is useful as a diagnostic and clearly helps a narrow full-anchor planner, but the current hard `score>0` filter should not replace T2.

The observations are consistent with the hard filter removing some useful fragments even when their net 25-step block displacement is not positive, while median on-manifold accuracy alone cannot identify which candidates will support the CEM elite update. A future retrieval revision should therefore be evaluated as a soft goal-aware ranking or mixture component, not as a hard replacement for free exploration. This is a mechanism hypothesis, not an isolated causal identification.

## Artifact integrity

- Formal roots: `outputs/eval/cube/goal_conditioned/{G1T2,G1T1}/{red,blue_v2,yellow_v2}`.
- Gate captures: `outputs/eval/cube/goal_conditioned/gate_capture/{G1T2,G1T1}/{red,blue_v2,yellow_v2}`.
- Physical truth: `outputs/eval/cube/goal_conditioned/physical_cache/{G1T2,G1T1}/{red,blue_v2,yellow_v2}`.
- Imagination scoring: `outputs/eval/cube/goal_conditioned/imagination_error/{G1T2,G1T1}/{red,blue_v2,yellow_v2}`.
- Six formal cells contain 300 videos, 300 cost JSON files, 300 cost NPZ files, 351 planning cycles, and 3,510 iteration records. All 300 videos decode to 50 frames at 736x288.
- Offline artifacts contain 72 new physical cases and 21,600 new candidate score rows.
- Formal first-cycle pools numerically match the gate captures: maximum candidate drift is `4.77e-7` and maximum latent-cost drift is `0.003632`; top-1 IDs and top-30 sets remain unchanged.
- New source files: `le-wm/eval_goal_conditioned.py`, `le-wm/tools/cube_goal_conditioned_common.py`, and `le-wm/tools/cube_goal_conditioned_offline.py`. No site-package or original evaluation script was modified.
