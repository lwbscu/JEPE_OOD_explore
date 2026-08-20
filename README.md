# JEPE OOD Explore

Reproducible code, reports, and evidence for a controlled study of OOD behavior
in a Cube JEPA world-model planner. Large artifacts are hosted in the
[Hugging Face dataset repository](https://huggingface.co/datasets/scilwb/JEPE_OOD_explore).

## Project in one page

**Research question.** Which part of a visual world-model planner limits
closed-loop OOD control: goal construction, visual perception, candidate
generation, ranking, slow-loop intervention, or learned dynamics?

**Best deployable result.** Robust v1 + T2 reaches **92/92/86%** on paired
Red/Blue-v2/Yellow-v2 evaluation, macro **90.00%**. The 94% Red probe result is
a privileged-coordinate diagnostic, not a three-color deployment result.

| Line | Main intervention | Quantitative result | Final status |
|---|---|---|---|
| Goal and perception | Real frame + controlled recolor + EE continuity; MaskedAug then Robust v1 | 66.00% baseline to 90.00% macro | Improved; floor, light, and camera remain sensitive |
| Candidate generation | Memory Seed and T2 | 74.00% then 87.33% macro | Effective planning-side intervention |
| Ranking and navigation | top-1, blind LLM, probe XYZ cost, waypoint chain | Probe 32/18/12% to 50/58/52%, but 4 cm chain falls to 16/12/18% | Ranking-only and waypoint remedies rejected |
| Slow loop | B1/B2 rule and LLM intervention | 72% baseline versus 70% rule and 70% LLM | Archived |
| Dynamics training | Off-policy V1/V2/V3 and official Play v1 | Expert depth-5 5.16 mm; planner candidates 85.24/118.04/124.55 mm | Four-round line archived |

The final evidence supports a structural mismatch between the JEPA one-step
training objective and the multi-step prediction required by planning. See the
[final planning analysis](docs/eval/cube/PLANNING_PROBLEM_ANALYSIS.md),
[Play verdict](docs/eval/cube/PLAY_LINE_VERDICT.md), and
[waypoint report](docs/eval/cube/waypoint_probe/WAYPOINT_REPORT.md).

## Main three-color matrix

| Model / training arm | Red | Blue v2 | Yellow v2 | Macro |
|---|---:|---:|---:|---:|
| MaskedAug + T2 | 88% | 88% | 86% | 87.33% |
| Robust v1 + T2 | 92% | 92% | 86% | 90.00% |
| No augmentation, 12,732 steps | 86% | 74% | 74% | 78.00% |
| No augmentation, 16,732 cumulative steps | 88% | 72% | 72% | 77.33% |

### Intermediate probe-goal diagnostic

| Target tier | Robust latent cost | Probe XYZ cost | Delta |
|---|---:|---:|---:|
| In-box | 32% | 50% | +18 pp |
| +5 cm | 18% | 58% | +40 pp |
| Fallback support (median 5.57 cm) | 12% | 52% | +40 pp |

The probe changes the cost interface and improves each paired tier, but does
not solve navigation: a 4 cm waypoint chain scores 16/12/18% versus direct
probe 50/58/52%. Full paired flips and provenance are in
[`CONTROL_AND_PROBEGOAL_REPORT.md`](docs/eval/cube/CONTROL_AND_PROBEGOAL_REPORT.md)
and [`WAYPOINT_REPORT.md`](docs/eval/cube/waypoint_probe/WAYPOINT_REPORT.md).

## Repository map

- `code/`: LeWM experiment and tool scripts, preserving their original layout.
- `docs/`: experiment reports, verdicts, analyses, and validation records.
- `evidence/`: 100 retained MP4s plus comparison/contact-sheet images.
- `LICENSE` and `NOTICE`: upstream MIT terms and derivative-work attribution.

## Reproduction entry points

```bash
# Zero-augmentation control: 12,732 steps plus a fresh 4,000-step continuation
python code/train_cube_control_noaugment.py --run-id control_noaugment_seed3072 --phase all --num-workers 6

# T2 paired control evaluation (run once for each released control checkpoint)
python code/eval_control_noaugment.py --checkpoint <control.pt> --condition all --num-eval 50 --authorize-formal

# Robust-specific embedding dataset and strict XYZ probe
python code/tools/build_cube_probe_dataset.py --checkpoint <robust.pt> --output <embedding_dir> --max-frames 400000 --sampling-mode episode_blocks
python code/tools/train_cube_xyz_probe.py --dataset <embedding_dir> --device cuda

# Paired latent/probe goal-cost evaluation
python code/eval_probe_goal_ood.py --checkpoint <robust.pt> --probe <probe.pt> --probe-dataset-metadata <embedding_dir>/metadata.json --tier all --mode both --num-eval 50 --authorize-formal

# Official Play conversion, mixed one-step training, and fail-stop offline gate
python code/tools/prepare_cube_play_v1.py --help
python code/train_cube_play_v1.py --help
python code/tools/evaluate_cube_play_v1.py --help
```

The original runs used absolute `/root/autodl-tmp/ailab/...` paths. Historical
reports preserve those paths as provenance; map them to your checkout and data
root when reproducing. The 20,566 previously validated bulk videos were deleted
for disk recovery and can be regenerated with the corresponding evaluators.

## Data and limitations

- Fixed evaluation episodes are excluded from training, probes, memory lookup,
  and released off-policy generation where each protocol requires it.
- Real Cube frames support only about 7.02 cm outside the nominal target box.
  Requested +10 cm and +20 cm tiers therefore share a documented fallback
  support point with median distance about 5.57 cm; they are not claimed as
  true +10/+20 cm measurements.
- The original Cube HDF5, official Play source, PushT data, and Quentinll base
  checkpoint are not redistributed. Follow the source links in `NOTICE`.
- Play v1 passed expert-retention checks but failed all candidate-pool gates, so
  no online Play evaluation was authorized.
- Negative off-policy and long-horizon results are retained for scientific
  completeness.

## Future work

The next controlled direction is an explicitly multi-step training target that
matches the planner's joint off-policy action sequences while retaining the
expert manifold. Any slow loop should validate goal continuity and model
uncertainty before action search, instead of attempting repeated post-hoc
recovery after an unreliable plan is active.

## Links

- Code and reports: https://github.com/lwbscu/JEPE_OOD_explore
- Weights, derivative data, memory index, reports, and evidence: https://huggingface.co/datasets/scilwb/JEPE_OOD_explore
