# No-Augmentation Control and Probe-Goal Cost Report

Date: 2026-08-20 (UTC)

## Executive conclusion

The no-augmentation control rejects the simple undertraining explanation. At the
same `12,732`-step budget as MaskedAug, the control reaches `86/74/74%` on
Red/Blue-v2/Yellow-v2, versus `88/88/86%` for MaskedAug + T2. Extending the
control to `16,732` cumulative steps raises only Red to `88%` while Blue-v2 and
Yellow-v2 fall to `72/72%`. Robust-v1, at the corresponding cumulative compute,
reaches `92/92/86%`. More unaugmented expert optimization therefore does not
recover the color-OOD gains; the targeted augmentation attribution remains
supported.

The strict robust-v1 XYZ probe passes its quality gate with `3.3206 mm` median
test error. Replacing image-goal latent cost with privileged target-coordinate
probe cost raises success from robust latent `32/18/12%` to `50/58/52%` on
in-box/+5cm/fallback-max targets. This is a large partial repair, especially
outside the target box, but the pre-registered in-box criterion was `>=70%` and
the result is only `50%`. The 26% in-box continuity failure is therefore partly
goal-cost representational, but not solely so; imagined trajectory quality and
closed-loop navigation remain limiting.

All success counts and paired flips below were independently recomputed from the
formal `results.json` episode vectors rather than copied from prose reports.

## Phase 0: authentication and secret boundary

Authentication passed before experiments began:

- GitHub identity `lwbscu` has push/admin permission on the public
  `lwbscu/JEPE_OOD_explore` repository.
- Hugging Face identity `scilwb` can access the public, non-gated dataset
  repository `scilwb/JEPE_OOD_explore`.
- `/root/autodl-tmp/ailab/llm_api.md` is a forbidden credential source. It is
  excluded from both staging trees and must never be copied, linked, quoted,
  committed, or uploaded.
- Upload authorization remains conditional on the final byte-stream secret scan
  of the exact GitHub/HF staging manifests and the Git index.

The auditable records are:

- `/root/autodl-tmp/ailab/outputs/release/PHASE0_AUTH_AND_SECRET_REPORT.md`
- `/root/autodl-tmp/ailab/outputs/release/phase0_auth_status.json`

No credential value appears in this report.

## Part 1: no-augmentation control

### Training contract

The control starts from the original Quentinll LeWM-Cube checkpoint, uses the
original expert data, applies no pixel augmentation (including no MaskedAug), and
excludes the frozen 50 formal evaluation episodes (`9,100` clips). Both phases
use batch `128`, AdamW, learning rate `1e-5`, weight decay `0.001`, bf16 mixed
precision, SIGReg weight `0.09`, gradient clipping `1.0`, and seed `3072`.

The run intentionally has two optimizer phases:

| phase | warm start | phase steps | cumulative step | optimizer schedule | wall time |
| --- | --- | ---: | ---: | --- | ---: |
| A | original Quentinll weights | 12,732 | 12,732 | fresh AdamW, 12,732-step cosine schedule | 5,449.93s (1h30m49.93s) |
| B | exported phase-A weights | 4,000 | 16,732 | fresh AdamW, fresh 4,000-step cosine schedule | 1,823.65s (30m23.65s) |
| total | - | 16,732 | 16,732 | two phases | 7,273.59s (2h01m13.59s) |

This is not one uninterrupted 16,732-step optimizer trajectory. The reset is
deliberate: phase A matches the MaskedAug schedule, while phase B matches the
fresh 4,000-step robust-v1 continuation schedule. This improves compute/schedule
alignment but must be retained as an interpretation boundary.

### Training curves and checkpoints

TensorBoard scalars are logged every 20 steps. Values below are the first and
last logged points in each phase.

| phase | logged steps | pred loss first -> last | SIGReg first -> last | total first -> last | terminal validation pred loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 19 -> 12,719 | 0.007968 -> 0.004774 | 0.960938 -> 0.878906 | 0.094394 -> 0.083876 | 0.002559 |
| B | 19 -> 3,999 | 0.005884 -> 0.005163 | 0.945312 -> 0.855469 | 0.090845 -> 0.082311 | not emitted before max-step stop |

The curves are healthy and finite. Phase B improves its own logged endpoint but
does not improve the OOD online results, so a lower within-distribution training
loss should not be read as color-OOD robustness.

| checkpoint | path | SHA-256 |
| --- | --- | --- |
| 12,732 | `/root/autodl-tmp/ailab/checkpoints/lewm-cube-control_noaugment/control_noaugment_seed3072/weights_step_12732.pt` | `5f994fc004f8bd11e241c881707f01213522d398adea010fef7c77d7c3433240` |
| 16,732 | `/root/autodl-tmp/ailab/checkpoints/lewm-cube-control_noaugment/control_noaugment_seed3072/weights_final.pt` | `2cfe36941194652d46ef46d849ece0d2f14b296a19607793a1044d9ad073b808` |

Training provenance:

- `/root/autodl-tmp/ailab/outputs/train/control_noaugment/control_noaugment_seed3072/run_plan.json`
- `/root/autodl-tmp/ailab/outputs/train/control_noaugment/control_noaugment_seed3072/phase_a/completed.json`
- `/root/autodl-tmp/ailab/outputs/train/control_noaugment/control_noaugment_seed3072/phase_b/completed.json`
- `/root/autodl-tmp/ailab/logs/tensorboard/control_noaugment/control_noaugment_seed3072/`
- training entry point SHA-256: `e49c3501730372a1a4e754812b877475944774ddaf78750e95c6c4aa52ad9c4c`

### Frozen T2 protocol

All four rows below use the same 50 formal rows, seed `42`, goal offset `25`,
budget `50`, 300 CEM candidates, 10 iterations, and T2's `10` exact memory
seeds + `20` sigma-0.1 perturbations + `270` free candidates.

| model/checkpoint | Red | Blue-v2 | Yellow-v2 | macro |
| --- | ---: | ---: | ---: | ---: |
| MaskedAug + T2 | 44/50 (88%) | 44/50 (88%) | 43/50 (86%) | 87.33% |
| robust-v1 + T2 | 46/50 (92%) | 46/50 (92%) | 43/50 (86%) | 90.00% |
| no-aug, step 12,732 + T2 | 43/50 (86%) | 37/50 (74%) | 37/50 (74%) | 78.00% |
| no-aug, step 16,732 + T2 | 44/50 (88%) | 36/50 (72%) | 36/50 (72%) | 77.33% |

The control evaluation roots are
`/root/autodl-tmp/ailab/outputs/eval/cube/control_noaugment/step_12732/`
and
`/root/autodl-tmp/ailab/outputs/eval/cube/control_noaugment/step_16732/`.
Their aggregate records are `evaluation_summary_step_12732.json` and
`evaluation_summary_step_16732.json` in the parent directory.

### Paired environment flips

Indices are zero-based formal environment indices. `F->S` means the named
current arm repairs a MaskedAug + T2 failure; `S->F` is the reverse.

| current arm | condition | F->S | S->F |
| --- | --- | --- | --- |
| robust-v1 | Red | `[22, 31]` | `[]` |
| robust-v1 | Blue-v2 | `[22, 31]` | `[]` |
| robust-v1 | Yellow-v2 | `[31]` | `[27]` |
| no-aug 12,732 | Red | `[]` | `[46]` |
| no-aug 12,732 | Blue-v2 | `[31]` | `[4, 10, 27, 34, 36, 37, 38, 42]` |
| no-aug 12,732 | Yellow-v2 | `[11, 31]` | `[1, 10, 33, 34, 36, 37, 43, 45]` |
| no-aug 16,732 | Red | `[31]` | `[46]` |
| no-aug 16,732 | Blue-v2 | `[11, 31]` | `[4, 10, 27, 34, 36, 37, 38, 42, 43, 45]` |
| no-aug 16,732 | Yellow-v2 | `[11, 31]` | `[1, 10, 27, 33, 34, 36, 37, 43, 45]` |

The second no-aug phase changes the 12,732-step arm as follows: Red repairs env
`31` with no regression; Blue-v2 repairs env `11` but regresses envs `[43, 45]`;
Yellow-v2 has no repair and regresses env `27`. Thus the extra 4,000 no-aug steps
produce `+2/-2/-2 pp`, not a delayed color-OOD catch-up.

### Control verdict

The preregistered outcomes were: comprehensive catch-up would support
underfitting, whereas Red-only improvement with stagnant Blue/Yellow would
support augmentation causality. The latter is observed. At equal 12,732-step
compute, no-aug is `-2/-14/-12 pp` behind MaskedAug + T2. At equal cumulative
16,732-step compute, no-aug is `-4/-20/-14 pp` behind robust-v1. More expert
optimization recovers Red but not the two recolored conditions. This is strong
evidence that targeted visual intervention, rather than training duration alone,
drives the color-OOD gains.

## Part 2: robust-v1 probe-goal cost

### Probe contract and quality

The probe is trained specifically on robust-v1 encoder embeddings and predicts
only block XYZ. The 400,000-row dataset is split by episode into `320,000`
train / `40,000` validation / `40,000` test rows across `1,593/200/200`
episodes. Splits are episode-disjoint, and the frozen 50 formal evaluation
episodes are excluded.

| test metric | value |
| --- | ---: |
| median XYZ error | 3.3206mm |
| mean XYZ error | 4.0678mm |
| p90 / p95 | 7.7999 / 9.7120mm |
| XYZ RMSE | 4.9709mm |
| quality gate | PASS (`3.3206 < 15mm`) |

Artifacts:

- probe: `/root/autodl-tmp/ailab/models/probes/cube_robust_v1_xyz/robust_v1.pt`, SHA-256 `caa7a92435c01df358382222d22d09313b2b06e45945cd893607764b2db28792`
- probe metadata: `/root/autodl-tmp/ailab/models/probes/cube_robust_v1_xyz/metadata.json`
- embedding metadata: `/root/autodl-tmp/ailab/outputs/probe/cube_robust_v1/dataset/metadata.json`
- robust-v1 weights: `/root/autodl-tmp/ailab/checkpoints/lewm-cube-robust_v1/lewm-cube-robust_v1/weights_final.pt`, SHA-256 `cffe41b70ed743c7ecf63610b0ebad2be64d6903572ec31e0379f95800072eed`

### Cost intervention

The latent arm encodes the retrieved real goal frame and ranks terminal imagined
states by latent distance. The probe arm removes goal pixels from the rollout
cost path and ranks the terminal imagined block XYZ against the privileged target
XYZ. The predictor, encoder, T2 sampling policy, physical environment, target
poses, real H5 target-frame selection, fixed 50 starts, seed, and budget remain
frozen.

The formal output root is
`/root/autodl-tmp/ailab/outputs/eval/cube/probe_goal_cost/`; its provenance and
aggregate records are `run_manifest.json` and `summary.json`.

### Three-tier results

| tier | actual target distance | existing MaskedAug latent | robust-v1 latent rerun | robust-v1 probe cost | probe delta vs robust latent | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| in-box | 0.00cm | 13/50 (26%) | 16/50 (32%) | 25/50 (50%) | +18pp | FAIL `<70%` gate |
| +5cm | median 5.00cm | 11/50 (22%) | 9/50 (18%) | 29/50 (58%) | +40pp | large partial repair |
| fallback-max | median 5.57cm, max 7.02cm | 6/50 (12%) | 6/50 (12%) | 26/50 (52%) | +40pp | large partial repair |

`fallback-max` reuses the prior `+10cm` target selection because eligible real
H5 frames do not reach 10cm after exclusions. It is not a true 10cm or 20cm
measurement. The selected population spans approximately 5.43-7.02cm and has a
5.57cm median.

Robust-v1 itself changes latent-cost performance relative to the existing
MaskedAug benchmark by `+6/-4/0 pp`. This rerun is necessary: attributing the
entire probe result against the old 26% row would confound the world-model
checkpoint with the goal-cost intervention.

### Probe paired flips

Robust-v1 latent cost versus the existing MaskedAug latent benchmark:

| tier | F->S | S->F | net |
| --- | --- | --- | ---: |
| in-box | `[4, 5, 7, 23, 44]` | `[0, 21]` | +3 envs (+6pp) |
| +5cm | `[]` | `[0, 34]` | -2 envs (-4pp) |
| fallback-max | `[13]` | `[32]` | 0 envs (0pp) |

Probe cost versus the robust-v1 latent rerun:

| tier | F->S | S->F | net |
| --- | --- | --- | ---: |
| in-box | `[0, 9, 10, 12, 18, 19, 24, 25, 37, 41, 43, 45]` | `[35, 36, 44]` | +9 envs (+18pp) |
| +5cm | `[0, 1, 2, 4, 6, 7, 12, 16, 19, 22, 24, 29, 31, 33, 34, 35, 36, 37, 41, 44, 45]` | `[42]` | +20 envs (+40pp) |
| fallback-max | `[1, 5, 7, 18, 19, 22, 23, 24, 29, 31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 46]` | `[]` | +20 envs (+40pp) |

For completeness, probe cost versus the existing MaskedAug latent benchmark:

| tier | F->S | S->F | net |
| --- | --- | --- | ---: |
| in-box | `[4, 5, 7, 9, 10, 12, 18, 19, 23, 24, 25, 37, 41, 43, 45]` | `[21, 35, 36]` | +12 envs (+24pp) |
| +5cm | `[1, 2, 4, 6, 7, 12, 16, 19, 22, 24, 29, 31, 33, 35, 36, 37, 41, 44, 45]` | `[42]` | +18 envs (+36pp) |
| fallback-max | `[1, 5, 7, 13, 18, 19, 22, 23, 24, 29, 31, 33, 34, 35, 36, 41, 42, 43, 44, 46]` | `[]` | +20 envs (+40pp) |

All three comparisons have identical `evaluated_rows` and `target_rows` within
each tier.

### Probe-goal verdict

The probe intervention is materially useful but does not pass the requested
repair criterion. In-box rises from robust latent 32% to 50%, still 20pp below
the `>=70%` threshold. The result rules out “goal-image continuity is irrelevant”
because direct coordinate cost repairs 9-20 net environments per tier. It also
rules out “goal-image continuity is the whole problem” because half of in-box
episodes still fail despite a 3.32mm probe and direct privileged target XYZ.

The remaining bottleneck is consistent with predictor/navigation quality:
candidate imagined endpoints can be decoded and ranked against the right target,
but the model-planner-controller chain still fails to turn that objective into
reliable physical success. A future restart should therefore improve multi-step
endpoint fidelity or candidate/control coverage rather than merely replacing the
goal representation again.

## Trace and causal boundaries

- T2 traces record all ten CEM iterations for the executed closed-loop plan. The
  independent perturbation RNG is derived from `[42, dataset_row,
  planning_cycle]`; the ten retrieved sources exclude the current evaluation
  episode.
- In probe outputs, `trust_trace.npz` retains the legacy array name
  `latent_costs`. For `cost_mode=probe`, that array contains XYZ probe costs,
  not latent distances. Each `results.json` records this explicitly in
  `trace_cost_field_note`.
- Changing cost changes elites, later CEM distributions, actions, physical
  states, and subsequent planning cycles. The paired arms therefore share starts,
  targets, protocol, and initial stochastic contract, but they are not an
  offline re-ranking of one immutable full trajectory pool.
- The probe evaluates predicted terminal latent states; it does not replace the
  predictor with privileged simulation. Its gains isolate a cost/goal-interface
  contribution, while remaining failures still combine predictor error,
  candidate coverage, CEM optimization, receding-horizon control, and physical
  execution.
- Fifty paired episodes give exact benchmark counts but limited statistical
  resolution (2pp per episode). The causal claims above are intentionally limited
  to these frozen conditions and target populations.

## Reproducibility

- control trainer: `/root/autodl-tmp/ailab/le-wm/train_cube_control_noaugment.py`, SHA-256 `e49c3501730372a1a4e754812b877475944774ddaf78750e95c6c4aa52ad9c4c`
- control evaluator: `/root/autodl-tmp/ailab/le-wm/eval_control_noaugment.py`, SHA-256 `e470f5fc191036ce828def26c95b18b88ecffe6bcaa767a2c462ec1a5957d8d7`
- strict probe trainer: `/root/autodl-tmp/ailab/le-wm/tools/train_cube_xyz_probe.py`, SHA-256 `3b41a198ada57892ab242157f068fbd28d32d35991b1017d08056d8c9d9a5fc6`
- probe-goal evaluator: `/root/autodl-tmp/ailab/le-wm/eval_probe_goal_ood.py`, SHA-256 `fcb35f71e4dee2c95958a628aa82a04222556e3eee6ec7fd0fcf8abef118dcd0`
