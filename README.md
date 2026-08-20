# JEPE OOD Explore

Reproducible code and evidence for a controlled study of visual and target-space
OOD behavior in a Cube world-model planner. Large artifacts are hosted in the
[Hugging Face dataset repository](https://huggingface.co/datasets/scilwb/JEPE_OOD_explore).

## Headline results

| Model / training arm | Red | Blue v2 | Yellow v2 | Macro |
|---|---:|---:|---:|---:|
| MaskedAug + T2 | 88% | 88% | 86% | 87.33% |
| Robust v1 + T2 | 92% | 92% | 86% | 90.00% |
| No augmentation, 12,732 steps | 86% | 74% | 74% | 78.00% |
| No augmentation, 16,732 cumulative steps | 88% | 72% | 72% | 77.33% |

### Privileged-coordinate probe goal cost

| Target tier | Robust latent cost | Probe XYZ cost | Delta |
|---|---:|---:|---:|
| In-box | 32% | 50% | +18 pp |
| +5 cm | 18% | 58% | +40 pp |
| Fallback support (median 5.57 cm) | 12% | 52% | +40 pp |

The full causal interpretation, paired flips, probe quality, and training
curves are in
[`CONTROL_AND_PROBEGOAL_REPORT.md`](docs/eval/cube/CONTROL_AND_PROBEGOAL_REPORT.md).

## Repository map

- `code/`: LeWM experiment and tool scripts, preserving their original layout.
- `docs/`: experiment reports, verdicts, and diagnosis documents.
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
```

The original runs used absolute `/root/autodl-tmp/ailab/...` paths. Historical
reports preserve those paths as provenance; map them to your checkout and data
root when reproducing. The 20,566 previously validated bulk videos were deleted
for disk recovery and can be regenerated with the corresponding evaluators.

## Data and limitations

- Fixed evaluation episodes are excluded from training, probes, memory lookup,
  and released off-policy generation.
- Real Cube frames support only about 7.02 cm outside the nominal target box.
  Requested +10 cm and +20 cm tiers therefore share a documented fallback
  support point with median distance about 5.57 cm; they are not claimed as
  true +10/+20 cm measurements.
- The original Cube HDF5, PushT data, and Quentinll base checkpoint are not
  redistributed. Follow the source links in `NOTICE`.
- Negative off-policy and long-horizon results are retained to make the release
  scientifically complete.

## Links

- Code and reports: https://github.com/lwbscu/JEPE_OOD_explore
- Weights, derivative data, memory index, and evidence: https://huggingface.co/datasets/scilwb/JEPE_OOD_explore
