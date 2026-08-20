# Subgoal Intervention Diagnosis

## Scope and Frozen Inputs

This is a zero-simulation analysis of the 36 STALLED interventions in the frozen B2 rule run. Inputs are the rule trace, goal-switch/intervention artifacts, cost histories, and the 50 recorded videos under `outputs/eval/cube/longhorizon/rule_offset100/`. No new environment rollout was run.

The row-level table is [interventions.csv](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/diagnosis_work/interventions.csv), SHA256 `5f81d09d76e2342e576f738bdc8a93f62bbdebcd597822bf94829c5850f27fc5`. The aggregate evidence is [summary.json](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/diagnosis_work/summary.json), SHA256 `1d7b1b50616a3d3cf786b27708efff3b9eb3c5e4eb1bbc5aaa2d5f895a50e74e`.

## Classification

| Type | Count | Interpretation |
|---|---:|---|
| Construction / continuity mismatch | 36/36 | Primary type: planner cost improved, but the physical block did not move toward the injected real-frame goal. The retrieved frame is usually a discontinuous scene. |
| Window insufficient | 0/36 | Not supported as the primary cause. Exact timeout traces show essentially zero block displacement. |
| Repeated-switch jitter (descriptive overlap) | 34/36 | 13/15 intervention episodes had multiple interventions, but there were zero switches inside an active window. This is retry churn after timeout, not physical back-and-forth evidence. |

The classifications overlap: 34 interventions are both construction-type and in multi-intervention episodes. The primary diagnosis is construction/continuity, not the retry schedule.

## Evidence

### 1. Window sufficiency

For the 30 complete timeout interventions with privileged terminal positions, median net block progress toward the subgoal was `-1.33e-11 m`, maximum absolute positive progress was `2.58e-10 m`, and `0/30` exceeded 1 mm. The six budget-censored interventions were estimated from video with affine calibration (median calibration error 3.82 mm, p90 9.59 mm); none exceeded 1 cm estimated progress.

The initial current-to-subgoal distance had median `0.1632 m`. A non-held-out expert reference moved a mean `0.1201 m` in 25 steps across all segments (`0.1575 m` among segments moving more than 1 mm), corresponding to naive estimates of 34.0 or 25.9 steps. Those estimates would motivate an adaptive window in isolation, but the measured zero motion rules out “the window was merely a little short” as the main explanation.

### 2. Planner cost versus physical progress

All 36 injected-goal CEM solves reduced their within-solve best cost. The median absolute reduction was `130.62` cost units and the median relative reduction was `52.1%`. Nevertheless, the 30 exact timeout traces show no physical block progress. Cost improvement therefore reflects an internal goal-image/planner response, not successful execution toward the real-frame subgoal. Costs across different goal images are only compared within each solve; no cross-goal absolute-cost claim is made.

### 3. Frame continuity and visual audit

The retrieved frame's EE pose was more than 10 cm from the current EE pose in `33/36` interventions. The EE jump median was `0.1813 m`, mean `0.2006 m`, and maximum `0.4939 m`. This is direct evidence that the real-frame goal image often describes a mechanically disconnected configuration.

Six representative current/dataset/goal panels are preserved for manual inspection:

- [comparison_00_env31_int27.png](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/diagnosis_work/comparisons/comparison_00_env31_int27.png)
- [comparison_01_env08_int26.png](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/diagnosis_work/comparisons/comparison_01_env08_int26.png)
- [comparison_02_env40_int20.png](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/diagnosis_work/comparisons/comparison_02_env40_int20.png)
- [comparison_03_env09_int23.png](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/diagnosis_work/comparisons/comparison_03_env09_int23.png)
- [comparison_04_env08_int13.png](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/diagnosis_work/comparisons/comparison_04_env08_int13.png)
- [comparison_05_env44_int29.png](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/diagnosis_work/comparisons/comparison_05_env44_int29.png)

Each panel contains the contemporaneous agent view, the retrieved dataset frame, and the injected goal panel. The principal visual check is continuity of the arm/gripper pose, not merely matching block coordinates.

### 4. Switching / jitter

There were 66 total goal switches: 36 final-to-subgoal and 30 timeout subgoal-to-final. There were no goal switches inside an open active window. Repeated interventions occurred in 13 of 15 affected episodes (`34/36` interventions), so the system retries often, but the trace does not support physical oscillation as the cause of failure. It is secondary bookkeeping churn caused by repeated timeouts.

## Repair Decision

1. **Use continuity-constrained retrieval.** Require the selected real frame's EE position to be within 10 cm of the current EE position, then rerank remaining valid real frames by the existing geometric query. This directly targets the measured `33/36` EE discontinuities and preserves the real-frame/no-synthetic-goal rule.
2. **Do not use adaptive subgoal budget as the first repair.** The distance/reference calculation alone suggests longer than 25 steps in some cases, but exact privileged traces show no physical motion at all. Extending a window would hide the construction failure and spend more compute without evidence it can help.
3. **Keep one active subgoal and the existing cooldown.** This is already true in the B2 trace; no in-window switching was observed. The retry count will remain a diagnostic, not the primary repair variable.
4. **Do not switch to target-seed injection in this retry.** The failure evidence identifies discontinuous goal frames, so changing the intervention semantics would confound the directly supported repair.

## Repair-to-Evidence Mapping

| Repair | Evidence addressed | Status |
|---|---|---|
| EE continuity cap `<=0.10 m` before real-frame landing | 33/36 EE jumps above 10 cm; internal cost improved while physical progress was zero | Selected |
| Keep only real HDF5 frames and revalidate after selection | All goal images must remain grounded; prevents a numeric candidate from bypassing continuity | Selected |
| Adaptive budget | Window estimate alone; contradicted by zero physical movement | Not selected |
| Cooldown / one active goal | No in-window oscillation; repeated switches are secondary timeout churn | Frozen, not changed |
| Goal-directed seed injection | Would change the causal intervention and is not required to test continuity | Not selected |

## Gate for Retry

The repaired rule arm must first pass a 2-episode smoke showing at least one subgoal reaches its real-frame target, with the same true-frame landing and next-policy-replan provenance checks used by B2. Only then will the frozen 50-episode rule evaluation run. If the repaired rule remains at or below the 72% baseline, the brain line will be archived and the LLM arm will not be rerun.
