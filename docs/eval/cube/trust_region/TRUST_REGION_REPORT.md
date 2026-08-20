# Cube Trust-Region CEM Report

Date: 2026-08-15

## Executive conclusion

Both trust-region variants passed the preregistered offline gate (`median E_roll <= 40 mm`) on all three colors, but only T2 improved closed-loop task success.

- **T1 nearest-seed warm start, initial std 0.2:** `72/70/70%` (Red/Blue-v2/Yellow-v2), macro `70.67%`.
- **T2 10 exact + 20 perturbed memory slots, legacy zero/unit CEM:** `88/88/86%`, macro `87.33%`.
- Frozen Seed×MaskedAug baseline: `84/86/84%`, macro `84.67%`.

T2 improves the baseline by `+4/+2/+2 pp` and raises the macro average by `+2.67 pp`. It does not beat the Red record of `92%` from Seed×ColorAug, but it sets the best Blue-v2 result (`88%`), the best Yellow-v2 result (`86%`), and the best single-policy macro average in the current matrix.

The causal statement must be narrower than “lower imagination error causes higher success.” T1 produced the lowest and tightest imagination error distribution, yet performed worst online. The supported conclusion is:

> Off-policy imagination error is a real planning failure mode, but lower imagination error alone is insufficient. The observed pattern is consistent with candidate target relevance and search coverage also mattering: constraining the whole search around one current-state nearest expert trajectory lowers hallucination while T1 success falls, whereas T2 adds expert-neighborhood support without discarding the legacy search distribution and performs better. The present compound interventions do not isolate those factors causally.

## Frozen protocols

- Checkpoint: MaskedAug `weights_final.pt`, SHA256 `d64501aa8e7dac1205d3a134c5bd7c160361e16d6da54c79e21e974cdc953117`.
- CEM: 300 samples, top-30 update, 10 iterations, horizon/receding-horizon 5, action block 5, legacy updated elite-mean selector, seed 42.
- Retrieval: current-state 9D exact nearest neighbor; current evaluation episode excluded; 10 neighbors from 10 distinct source episodes.
- T1: closest retrieved seed is the full initial mean; initial sampling std `0.2`; no persistent injected slots.
- T2: legacy mean 0/std 1; slots 1–10 are exact seeds; slots 11–30 are two raw-action-domain `sigma=0.1` perturbed copies per seed, clipped to `[-1,1]`; perturbations are fixed per planning cycle and do not advance the solver RNG.
- Offline gate: full 50-env first-solve RNG order, fixed 12 env retained, 12×300 MuJoCo numerical replay, Masked probe endpoint error; pass iff pooled median `E_roll <= 40 mm` per cell.

## Offline gate

The fair reference is the old unseeded pool measured with the same Masked checkpoint/probe, not the official-checkpoint value near 78 mm.

| Protocol | Color | Old unseeded median | New median / p90 / p95 (mm) | New >40 mm | Median reduction | Gate |
|---|---|---:|---:|---:|---:|---:|
| T1 | Red | 85.720 | 10.763 / 19.891 / 23.538 | 3.56% | 87.4% | PASS |
| T1 | Blue-v2 | 112.476 | 11.910 / 22.919 / 26.192 | 2.17% | 89.4% | PASS |
| T1 | Yellow-v2 | 123.356 | 10.692 / 21.175 / 25.031 | 3.61% | 91.3% | PASS |
| T2 | Red | 85.720 | 21.115 / 161.502 / 227.729 | 32.78% | 75.4% | PASS |
| T2 | Blue-v2 | 112.476 | 22.672 / 164.567 / 224.411 | 33.94% | 79.8% | PASS |
| T2 | Yellow-v2 | 123.356 | 23.143 / 188.161 / 235.193 | 37.94% | 81.2% | PASS |

Encoder-floor medians for T1 were `5.807/11.735/8.282 mm`; for T2 they were `6.835/16.185/11.059 mm` (Red/Blue/Yellow). T2 passes the median gate but retains a large hallucination tail, so it should not be described as keeping the entire pool on-manifold.

## Online matrix

All entries use the same fixed 50 evaluation rows. Earlier rows are frozen prior results and were not rerun.

| Method | Red | Blue-v2 | Yellow-v2 | Macro average |
|---|---:|---:|---:|---:|
| Original mean | 72% | 64% | 62% | 66.00% |
| Memory Seed | 88% | 68% | 66% | 74.00% |
| Route2 global ColorAug | 80% | 64% | 78% | 74.00% |
| MaskedAug mean | 74% | 76% | 74% | 74.67% |
| Seed×ColorAug | **92%** | 68% | 80% | 80.00% |
| Seed×MaskedAug | 84% | 86% | 84% | 84.67% |
| **T1 warm trust region** | 72% | 70% | 70% | 70.67% |
| **T2 perturbed memory pool** | 88% | **88%** | **86%** | **87.33%** |

Against Seed×MaskedAug, T1 changes by `-12/-16/-14 pp`; T2 changes by `+4/+2/+2 pp`.

## Paired env flips versus Seed×MaskedAug

Notation: `F→S` means baseline failure becomes success; `S→F` means baseline success becomes failure. IDs are zero-based formal env indices, with dataset row in parentheses.

### T1

- Red F→S: `1(136513), 11(556268), 14(808839), 17(891249)`
- Red S→F: `5(189273), 6(257500), 10(456735), 12(712588), 16(882105), 23(1058181), 34(1523727), 35(1529852), 43(1725130), 46(1795165)`
- Blue-v2 F→S: `11(556268), 14(808839), 17(891249), 26(1269622)`
- Blue-v2 S→F: `0(128267), 5(189273), 6(257500), 10(456735), 12(712588), 23(1058181), 27(1294136), 33(1478821), 34(1523727), 35(1529852), 43(1725130), 46(1795165)`
- Yellow-v2 F→S: `1(136513), 11(556268), 14(808839), 17(891249), 38(1570913)`
- Yellow-v2 S→F: `0(128267), 5(189273), 6(257500), 10(456735), 12(712588), 23(1058181), 27(1294136), 33(1478821), 34(1523727), 35(1529852), 43(1725130), 46(1795165)`

### T2

- Red F→S: `0(128267), 1(136513)`; S→F: none.
- Blue-v2 F→S: `26(1269622)`; S→F: none.
- Yellow-v2 F→S: `1(136513)`; S→F: none.

## Interpretation

### a. Records

T1 does not approach the existing records. T2 reaches `88%` on Red, below the `92%` Seed×ColorAug ID record. T2's macro `87.33%` exceeds the previous best single-policy macro `84.67%`, and its `88/86%` are the new Blue-v2/Yellow-v2 bests.

### b. Does lower imagination error track success?

Not monotonically. T1 reduced median error by 87–91% and nearly eliminated the >40 mm tail, but lost 12–16 pp versus Seed×MaskedAug. T2 reduced the median less and retained a 33–38% >40 mm tail, yet gained 2–4 pp with no paired regressions.

This rejects the strongest one-variable causal claim. It motivates the following mechanism hypothesis, rather than identifying each component causally:

1. The legacy unconstrained pool contains severe off-policy hallucinations.
2. Expert-neighborhood candidates can improve elite updates.
3. The planner may need goal-relevant diversity. Current-state-only nearest retrieval is not goal-conditioned, so a narrow T1 neighborhood can be physically plausible without being useful for the current goal; source-goal mismatch was not directly measured here.

### c. Is off-policy predictor retraining necessary?

It remains justified, but it is not the immediate prerequisite for improvement. T2 already improves the best deployable baseline while leaving a large hallucination tail, showing that planner-side support shaping has remaining value. The next lightweight experiment should be goal-conditioned retrieval or a multi-neighbor mixture warm start with a less aggressive trust radius. If those saturate, simulator-generated off-policy trajectories should be used to retrain the predictor, targeting the T2 tail rather than replacing expert-distribution training wholesale.

## Artifact integrity

- Formal roots: `outputs/eval/cube/trust_region/{T1,T2}/{red,blue_v2,yellow_v2}`.
- Gate roots: `outputs/eval/cube/trust_region/imagination_error/{T1,T2}/{red,blue_v2,yellow_v2}`.
- Physical truth: `outputs/eval/cube/trust_region/physical_cache/{T1,T2}/{red,blue_v2,yellow_v2}`.
- Six formal cells contain 300 videos, 300 cost JSON files, 300 cost NPZ files, 364 planning cycles, and 3,640 iteration records. All videos decode to 50 frames at 736×288.
- Cross-process CUDA cost drift reached at most `0.00414`; candidate values, top-1 IDs, and top-30 sets remained consistent. The previously exact compressed-NPZ post-check was therefore changed to report-only diagnostics while all pre-run PASS/provenance/checkpoint/physical-truth bindings remain fail-closed.
