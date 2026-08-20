# Off-policy predictor fine-tuning: final verdict

## Decision

**CLOSE THE LINE.** No additional training retry, offline gate, or online T2 evaluation is authorized.

V3 did not produce a scientific model result. Both executions ended before the first step-500 realtime expert stopline, with no checkpoint or completed training artifact. This is an infrastructure/process-execution failure, not evidence that the V3 loss isolation itself passed or failed. The user designated V3 as the final attempt, so the absence of a valid V3 checkpoint closes the route operationally.

## Evidence across V1, V2, and V3

| Version | Intervention | Expert-manifold result | Candidate imagination result | Decision |
|---|---|---|---|---|
| V1 | Synthetic Gaussian / T2-perturbation / AR1 data; one-step loss | Held-out expert teacher loss `0.003402 -> 0.005822` (`+71.1%`) | Median `E_roll`: Red `85.72 -> 43.15` mm, Blue-v2 `112.48 -> 94.88` mm, Yellow-v2 `123.36 -> 78.30` mm; all three hard gates failed | No online run |
| V2 main | Real planner-in-the-loop data; rollout loss on both expert and V2; `96E/32V2` | `0.003295 -> 0.012589` (`+282.0%`) | Not evaluated because the expert stopline failed | No offline/online run |
| V2 retry | Same objective with `104E/24V2` | `0.003295 -> 0.043268` (`+1213.0%`) | Not evaluated because the expert stopline failed | No further model retry |
| V3 | Real planner data; rollout loss isolated to V2; `96E/32V2` | No scientific measurement: both executions stopped before step 500 | No valid checkpoint, so no candidate gate | Route closed |

V1 established that off-policy supervision can reduce candidate-distribution rollout error, especially on Red, but it neither met the `<40 mm` / half-tail gate nor preserved the expert manifold. V2 established that applying the five-step rollout objective to expert samples is destructive in this setup; increasing the expert fraction did not repair it. V3 correctly isolated that objective in code, but the infrastructure did not sustain a run long enough to test the hypothesis.

## V3 protocol and implementation verification

- Base: MaskedAug checkpoint, not V1 or V2 weights.
- Frozen training contract: `96` expert + `32` V2 samples, batch `128`, LR `1e-5`, BF16, maximum `5000` steps.
- Expert loss path: teacher prediction plus shared SIGReg only; it never invokes autoregressive rollout.
- V2 loss path: teacher prediction plus shared SIGReg plus `0.5 x` five-step autoregressive rollout loss.
- Realtime stopline: paired `34` batches / `4352` clean expert clips every `500` optimizer steps; exact baseline `0.0032953384798020124`; strict relative increase `>10%` would stop training.
- CPU synthetic and real-model tests confirmed one rollout call for the V2 batch of 32, five-step BPTT, 93 trainable tensors / 11,740,484 parameters, frozen encoder/projector identity, and exact-zero expert rollout total and depth-1 through depth-5 fields.

Frozen code identities:

- `le-wm/cube_offpolicy_v3.py`: `8e46ff7239601650fba1fbcc04e232beb71f20f17ee937d6665642da5932374f`
- `le-wm/train_cube_offpolicy_v3.py`: `ddbcfa1656cada92c8df2cf3b07e001cd788d02999b69fa18a3acdcdaa5b25d9`
- `le-wm/tools/evaluate_cube_offpolicy_v3.py`: `5a0dbf46acc235515db7b60444fef7a29ec03fffa4cfc0c8740a1e22a399df6e`
- `le-wm/eval_offpolicy_v3.py`: `efbb5762a5909dab892e4184bc03bcbf7ec096ed5264b6c2e685108c1d9a10f2`
- Loss contract: `5ab8c7da32b1accf667af0c9fb917a6dcd3332c72f5243ea63a3e325130cac9c`

## V3 execution record

| Execution | Last persisted step | Realtime stopline | Expert rollout logs | Model artifacts | Classification |
|---|---:|---|---|---|---|
| Initial canonical run | `140` (console reached `150`) | Step 0 baseline exact; no step-500 record | Total and depths 1-5 exact zero at every logged point | None | Exit `137`; no cgroup OOM; infrastructure interruption |
| One allowed identical recovery | `160` | Step 0 baseline exact; no step-500 record | Total and depths 1-5 exact zero at every logged point | None | PTY closed with runner status `1`, no captured traceback and no OOM; second unexpected interruption |

The recovery used the same MaskedAug base, seed, data, split, code, run id, and default arguments. It was not a scientific or hyperparameter retry. Because it also ended before producing a stopline decision, a third launch is prohibited.

Evidence:

- First incident: `outputs/train/offpolicy_v3/INTERRUPTED_EXIT137_20260817T161551Z.md`
- Recovery incident: `outputs/train/offpolicy_v3/INTERRUPTED_RECOVERY_20260817T162502Z.md`
- Archived first partial output: `outputs/train/offpolicy_v3/offpolicy_v3_pred_seed3072.interrupted_exit137_20260817T161551Z/`
- Preserved recovery partial output: `outputs/train/offpolicy_v3/offpolicy_v3_pred_seed3072/`

## Gate and online status

| Gate | Status | Reason |
|---|---|---|
| Realtime/final expert stopline | **NOT MEASURED** | Neither run reached step 500 and no final checkpoint exists |
| Three-color 12x300 candidate imagination gate | **NOT RUN** | Requires a stopline-authorized V3 checkpoint |
| Expert-action Measurement 1 (`<=8 mm`) | **NOT RUN** | Requires a valid V3 checkpoint |
| T2 online 3x50 | **NOT RUN** | Aggregate offline authorization was never available |

There are therefore no V3 success rates or per-environment flips to report. The existing T2 baseline remains Red `88%`, Blue-v2 `88%`, Yellow-v2 `86%`.

## Final scientific assessment

The off-policy line is **plausible but unvalidated as a deployable improvement**:

1. V1 gives evidence that candidate-distribution data can reduce hallucination, but its gains were incomplete and accompanied by expert-manifold degradation.
2. V2 shows that a shared expert/V2 multi-step objective is unsafe for expert dynamics in the current optimization regime.
3. V3's loss isolation is the right controlled test of the V2 failure diagnosis, and the early logged contract behaved correctly, but there is no trained V3 model from which to infer retention, imagination accuracy, reranking, or online control.

Accordingly, this route is closed without claiming that V3 scientifically failed. The demonstrated bottleneck is twofold: objective interference in completed V1/V2 experiments, and inability to obtain a durable V3 run under the current execution infrastructure.

## Preconditions for any future reopening

Reopening requires an explicit new user decision; it must not happen automatically. Minimum prerequisites are:

1. A durable training supervisor that preserves the actual stderr and exit signal and survives client/session turnover.
2. A short infrastructure qualification run long enough to cross at least one step-500 callback without changing the frozen V3 model, data, loss, or hyperparameters.
3. More frequent recoverable infrastructure checkpoints may be added only as an execution safeguard, not used for checkpoint selection or hyperparameter search.
4. The first scientific run after reopening should remain the same frozen V3 arm. Only its expert stopline result should determine whether candidate gates are allowed.

Until those prerequisites and explicit authorization exist, the production recommendation remains the established planning-side solution rather than further predictor fine-tuning.
