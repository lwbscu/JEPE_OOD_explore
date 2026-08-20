# Cube Off-policy Predictor Fine-tuning Report

## Decision

The experiment is a **useful offline negative result**.

- Off-policy fine-tuning attacked the intended failure mode: with the `1e-5`
  checkpoint, median candidate rollout error fell by 49.7% on Red, 15.6% on
  Blue-v2, and 36.5% on Yellow-v2.
- It did not meet the preregistered fail-stop gate in any color. The aggregate
  gate is `FAIL`, so no 50-env T2 evaluation was run.
- Lowering the learning rate once to `5e-6`, as authorized for expert
  validation degradation, did not fix retention and produced worse physical
  rollout errors than the `1e-5` run.
- The present data therefore improve imagination for part of the candidate
  distribution, but do not yet make the pool reliably rankable enough to claim
  a better `pool has a solution -> controller selects it` conversion rate.

## 1. Off-policy dataset

| Item | Frozen value |
|---|---:|
| Rollouts | 30,000 |
| Model-step transitions | 150,000 |
| Environment steps | 750,000 |
| Horizon | 5 model steps = 25 env steps |
| Frames per rollout | 6, at env steps 0/5/10/15/20/25 |
| Gaussian `N(0,1)` | 12,000 (40%) |
| Memory-T2 seed + raw sigma 0.1 | 9,000 (30%) |
| Stationary AR(1), rho 0.8 | 9,000 (30%) |
| Storage | 120 JPEG-HDF5 shards, 2.1909 GiB |
| Collection time | 4,859.72 s = 1.350 h, 6.173 rollouts/s |
| Seed | 424200 |
| Early termination | Disabled; every rollout executed 25 steps |

The 30,000 source rows are unique and all have `step_idx <= 175`. The fixed 50
evaluation episodes are excluded from source rows and all Memory-T2 retrievals;
retrievals also exclude the current source episode and use 10 distinct source
episodes. The mixture is exactly 12k/9k/9k. All 120 shard hashes, action
inverse/clip/normalize semantics, continuous rollout IDs, and the fixed-length
trajectory contract were validated. A cross-shard sample of 1,440 JPEG frames
decoded as RGB `224x224`.

Artifacts:

- Dataset root: `/root/autodl-tmp/ailab/datasets/offpolicy_cube_v1/`
- Manifest: `/root/autodl-tmp/ailab/datasets/offpolicy_cube_v1/manifest.json`
- Selection: `/root/autodl-tmp/ailab/datasets/offpolicy_cube_v1/selection.npz`
- Collection log: `/root/autodl-tmp/ailab/logs/data/offpolicy_cube_v1/formal.log`

## 2. Mixed predictor fine-tuning

Both runs start from the frozen MaskedAug checkpoint. Each batch contains
exactly 64 expert and 64 off-policy samples. Expert samples retain the 80/20
masked-hue augmentation; off-policy images are unchanged. Only
`predictor + action_encoder + pred_proj` are trainable (93 parameter tensors,
11,740,484 parameters); encoder and projector remain frozen and are bitwise
unchanged. Loss is the original `pred + 0.09 * SIGReg` objective.

| Run | LR | Steps | Expert samples | Off-policy samples | Duration | Final SHA256 |
|---|---:|---:|---:|---:|---:|---|
| Primary | `1e-5` | 4,000 | 256,000 | 256,000 | 4,365.09 s | `4fa2478bddc612cf521548027ea6ae92e38f8f3d7925b05680601dc156d014dc` |
| Retention retry | `5e-6` | 4,000 | 256,000 | 256,000 | 4,390.63 s | `ea201b78490c650a3ed291fc306c1fe66f4583c528bae75a42cc7848a96aacf1` |

Training-curve first/last 50 logged-window changes:

| Run | Combined pred | Off-policy pred | Expert pred | Total loss |
|---|---:|---:|---:|---:|
| `1e-5` | -15.53% | -18.98% | -0.20% | -6.04% |
| `5e-6` | -14.07% | -17.57% | +2.37% | finite; no divergence |

The original 4,000-step run stopped mid-epoch and therefore did not execute
Lightning's epoch-end validation. A read-only paired post-hoc validation was
added and run on the frozen rollout split: 50 batches, each 64 expert + 64
off-policy, identical batch provenance for base and final, no optimizer,
`inference_mode`, and checkpoint/model hashes unchanged before and after.

| Checkpoint | Expert pred base -> final | Off-policy pred base -> final | Combined pred base -> final |
|---|---:|---:|---:|
| `1e-5` | 0.003402 -> 0.005822 (+71.1%) | 0.125788 -> 0.089010 (-29.2%) | 0.064595 -> 0.047416 (-26.6%) |
| `5e-6` | 0.003402 -> 0.005879 (+72.8%) | 0.125788 -> 0.094591 (-24.8%) | 0.064595 -> 0.050235 (-22.2%) |

The expert degradation is consistent across all 50 paired batches for the
`1e-5` run. This triggered the single authorized lower-LR retry. The retry did
not improve expert retention and was worse on off-policy validation, so no
further retraining was performed.

Training artifacts:

- Primary run: `/root/autodl-tmp/ailab/outputs/train/offpolicy_v1/offpolicy_v1_pred_seed3072/`
- Primary checkpoint: `/root/autodl-tmp/ailab/checkpoints/lewm-cube-offpolicy_v1/offpolicy_v1_pred_seed3072/weights_final.pt`
- Lower-LR run: `/root/autodl-tmp/ailab/outputs/train/offpolicy_v1/offpolicy_v1_pred_lr5e6_seed3072/`
- Lower-LR checkpoint: `/root/autodl-tmp/ailab/checkpoints/lewm-cube-offpolicy_v1/offpolicy_v1_pred_lr5e6_seed3072/weights_final.pt`

## 3. Offline fail-stop gate

The evaluator uses exactly the same old unseeded three-color `12 env x 300`
candidate pools and cached physical terminal states. It recomputes MaskedAug
base and new-model latent rollouts on the same actions; stored latent costs from
the older official checkpoint are not reused. The same MaskedAug XYZ probe is
valid because the encoder/projector are bitwise identical. There are 21,600
finite score rows: 2 models x 3 colors x 12 env x 300 candidates.

The hard gate is per color and all colors are required:

1. new median `E_roll < 40 mm`; and
2. new `P(E_roll > 40 mm)` no more than half the same-pool MaskedAug baseline.

### Primary `1e-5` result

| Color | Base median / >40 | New median / >40 | Median change | Gate target | Status |
|---|---:|---:|---:|---:|---|
| Red | 85.72 mm / 62.44% | 43.15 mm / 51.97% | -49.7% | <40 mm / <=31.22% | FAIL |
| Blue-v2 | 112.48 mm / 77.00% | 94.88 mm / 71.61% | -15.6% | <40 mm / <=38.50% | FAIL |
| Yellow-v2 | 123.36 mm / 77.03% | 78.30 mm / 65.97% | -36.5% | <40 mm / <=38.51% | FAIL |

Full distribution:

| Color | Base median / p90 / p95 | New median / p90 / p95 | New >40 |
|---|---:|---:|---:|
| Red | 85.72 / 204.29 / 230.64 | 43.15 / 175.24 / 208.45 mm | 51.97% |
| Blue-v2 | 112.48 / 248.11 / 277.66 | 94.88 / 225.07 / 256.81 mm | 71.61% |
| Yellow-v2 | 123.36 / 251.97 / 282.92 | 78.30 / 233.48 / 254.44 mm | 65.97% |

The encoder readout floor is unchanged, as expected. Median `E_imag` fell from
82.32 to 36.89 mm on Red, 95.39 to 52.55 mm on Blue-v2, and 90.69 to 44.39 mm
on Yellow-v2. Thus predictor correction is real, but terminal physical error
remains above threshold, especially in OOD colors.

### Success/failure stratification (`E_roll` median)

| Color | Base final-success | New final-success | Base final-failure | New final-failure |
|---|---:|---:|---:|---:|
| Red | 19.97 | 19.44 | 130.34 | 72.57 mm |
| Blue-v2 | 64.35 | 63.70 | 137.88 | 110.11 mm |
| Yellow-v2 | 71.09 | 42.49 | 137.49 | 92.55 mm |

Most median improvement comes from physically failing candidates. Successful
candidate tails do not consistently improve: final-success p90 becomes
60.47/187.42/234.60 mm for Red/Blue/Yellow versus
39.68/140.31/198.82 mm at baseline. The worsened successful-candidate tails are
consistent with the absence of a stable ranking improvement; this analysis
does not isolate them as its cause.

### Lower-LR retry

| Color | `1e-5` median / >40 | `5e-6` median / >40 | Better checkpoint |
|---|---:|---:|---|
| Red | 43.15 / 51.97% | 47.24 / 53.83% | `1e-5` |
| Blue-v2 | 94.88 / 71.61% | 101.08 / 73.53% | `1e-5` |
| Yellow-v2 | 78.30 / 65.97% | 84.24 / 67.83% | `1e-5` |

The lower-LR retry also fails every gate and is uniformly worse on the primary
physical metrics. The `1e-5` checkpoint is therefore the main result, despite
its expert heldout degradation.

Offline artifacts:

- Primary: `/root/autodl-tmp/ailab/outputs/eval/cube/offpolicy_v1/offline/`
- Lower LR: `/root/autodl-tmp/ailab/outputs/eval/cube/offpolicy_v1/offline_lr5e6/`

## 4. Candidate reranking

Success@K is the number of the fixed 12 environments whose top K contains a
final-success candidate. MRR uses all 12 environments. Values below compare
the MaskedAug base with the primary `1e-5` checkpoint.

| Cost | Color | Base S@1/3/5/10/30 | New S@1/3/5/10/30 | Base -> new MRR |
|---|---|---:|---:|---:|
| Latent | Red | 5/6/6/6/7 | 6/6/6/6/8 | .465 -> .514 |
| Latent | Blue-v2 | 4/6/6/7/8 | 5/6/7/8/8 | .432 -> .476 |
| Latent | Yellow-v2 | 6/6/6/7/7 | 5/6/7/7/7 | .512 -> .479 |
| Probe | Red | 5/6/6/7/7 | 5/6/7/7/8 | .473 -> .483 |
| Probe | Blue-v2 | 5/6/6/7/8 | 3/5/8/8/8 | .474 -> .388 |
| Probe | Yellow-v2 | 6/7/7/7/7 | 5/6/7/7/7 | .542 -> .475 |

Ever-success sensitivity gives the same non-uniform conclusion:

| Cost | Color | Base S@1/3/5/10/30 | `1e-5` S@1/3/5/10/30 | Base -> new MRR |
|---|---|---:|---:|---:|
| Latent | Red | 6/6/6/6/7 | 6/6/6/6/8 | .507 -> .514 |
| Latent | Blue-v2 | 4/6/6/7/8 | 5/6/7/8/8 | .433 -> .476 |
| Latent | Yellow-v2 | 6/6/6/7/7 | 5/6/7/7/7 | .512 -> .480 |
| Probe | Red | 5/6/6/7/7 | 6/6/7/7/8 | .473 -> .524 |
| Probe | Blue-v2 | 5/6/6/7/8 | 4/5/8/8/8 | .475 -> .430 |
| Probe | Yellow-v2 | 6/7/7/7/7 | 6/6/7/7/7 | .543 -> .522 |

For final-success labels, median successful/failed candidate ranks and
physical-min/final-optimum ranks are:

| Cost/color | Success/failure rank, base -> new | Physical min/final rank, base -> new |
|---|---:|---:|
| Latent Red | 137/158 -> 133/160 | 99/112 -> 44/37.5 |
| Latent Blue-v2 | 125/160 -> 126/161 | 106.5/78.5 -> 82/53 |
| Latent Yellow-v2 | 130/160 -> 129/160.5 | 165/164 -> 112.5/43.5 |
| Probe Red | 134/159 -> 133.5/160 | 75/75 -> 32.5/43 |
| Probe Blue-v2 | 122/162 -> 125/160 | 67.5/26 -> 64.5/67 |
| Probe Yellow-v2 | 129.5/159.5 -> 128/160 | 120/80.5 -> 55.5/46.5 |

The latent cost improves Red and Blue-v2 but regresses Yellow-v2 at top 1.
Probe cost improves some deeper-K coverage while regressing Blue-v2 and
Yellow-v2 top-1/MRR. Therefore the error reduction does not translate into a
stable three-color selector improvement.

Median rank of the physically final-distance-optimal candidate under latent
cost improves Red `112 -> 37.5`, Blue-v2 `78.5 -> 53`, and Yellow-v2
`164 -> 43.5`. The improved median physical-optimum ranks indicate that some
endpoint-ordering signal changed favorably, while Success@K/MRR show that it is
not uniformly usable.

The `5e-6` reranking result is mixed rather than uniformly worse: final-success
latent MRR is `.471/.473/.479` and probe MRR is `.463/.419/.438` on
Red/Blue/Yellow. This does not alter the gate decision; it prevents describing
lower LR as uniformly worse outside the preregistered physical-error metrics.

## 5. Online evaluation

No online T2 run was performed. Both aggregate gates are `FAIL`, and the formal
entry point rejects the run before model/environment loading. Consequently:

- there is no new Red/Blue-v2/Yellow-v2 success-rate row;
- there are no paired environment flips to report;
- the existing T2 baseline remains 88% / 88% / 86%.

This is the intended fail-stop behavior, not a missing experiment.

## 6. Interpretation and next data iteration

### What is established

1. The dynamics-stack-only update (`predictor + action_encoder + pred_proj`,
   with encoder/projector frozen) reduces the targeted aggregate rollout-error
   metric. This establishes metric movement, not improved selection or
   closed-loop control.
2. The current V1 coverage is insufficient. Blue-v2 improves least, all three
   `>40 mm` rates remain far above their half-baseline targets, and ranking gains
   are not consistent across cost definitions and colors.
3. The present training trade-off repairs many failed candidates but damages the
   already-good expert/success manifold. Lower LR alone does not resolve this.
4. Because the offline gate fails, this experiment does not establish an
   improved pool-solution conversion rate or closed-loop success rate.

### Likely mismatch in V1

- V1 covers iid Gaussian, stationary AR(1), and seed perturbations, but not the
  adaptive distribution produced by 10 CEM update rounds. Late-round means,
  variances, cross-time correlations, and extreme candidate tails are missing.
- Collection clips raw actions to `[-1,1]`, while the frozen old audit pools
  contain many inverse-scaled candidate elements outside that range. This makes
  the supervised action distribution only an approximation to the exact audit
  replay distribution.
- Off-policy frames are identity images; only the expert half gets MaskedAug.
  This is consistent with weaker Blue-v2 transfer, though not by itself causal.
- The original one-step `pred + SIGReg` training objective is unchanged. It does
  not directly optimize 1-to-5-step autoregressive rollout error or terminal
  physical readout.

### Recommended V2

1. Collect a new, disjoint training/validation pool from actual T2/CEM iteration
   distributions: approximately 60% candidates sampled across CEM rounds with
   extra weight on late rounds, 20% memory seeds with several noise scales, 10%
   iid Gaussian, and 10% AR1/piecewise-constant actions.
2. Match deployment action semantics exactly. Either collect the actual
   unclipped planner-to-environment actions, or deliberately change and re-freeze
   planner/audit/collection clipping together; do not mix contracts.
3. Stratify collection by contact density, near-40-mm terminal cases, model
   disagreement/high predicted uncertainty, and states where the base model has
   high rollout inconsistency. Do not use the frozen 12-env outcomes to select
   training samples.
4. Apply the same masked color augmentation to off-policy frames, or collect
   matched red/blue/yellow live renders in a separate causal arm.
5. Protect the successful manifold with expert:off-policy `60:40`, a shorter
   first budget (about 2k steps), and checkpoint selection on a new disjoint
   physical validation pool. Lower LR by itself was not sufficient.
6. If coverage alignment still fails, add explicit multi-step autoregressive
   rollout and terminal-state/readout losses. Keep the existing old 12x300 pools
   as a one-time final test rather than iteratively tuning on them.

## Final conclusion

Off-policy fine-tuning produced a useful offline negative result but did not
earn authorization for online evaluation. It reduced aggregate candidate
rollout error and improved several physical-optimum ranks, supporting—but not
confirming—the hypothesis that better action-distribution coverage can reduce
off-policy imagination error. The full error -> ranking -> control causal chain
remains unverified. V1 does not reduce imagination-error tails enough, preserve
the expert/success manifold, or yield a stable three-color reranker. The correct
next step is not another learning-rate sweep; it is a better matched dataset
from actual multi-round CEM distributions plus explicit multi-step training and
a fresh physical validation split.
