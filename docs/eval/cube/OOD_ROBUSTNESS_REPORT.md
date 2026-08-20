# OOD Robustness Report

## Executive Summary

The full-visual-axis finetune passed the expert stopline and improved the frozen T2
regression conditions from `88/88/86` to `92/92/86` (macro `87.33%` to `90.00%`,
`+2.67pp`). The gain is confined to red and blue; yellow is unchanged. Floor and
camera changes remain difficult, with camera being the weakest untrained axis.

The target-space curve falls from `26%` in-box to `22%` at a real `5cm` shell and
`12%` at the largest real target distance available. The requested `10cm` and
`20cm` tiers have no matching H5 frames after the fixed-episode exclusion; both
therefore use the same real fallback population (median `5.57cm`, maximum
`7.02cm`). They are not evidence for true 10/20cm performance.

## Part 0: Disk Cleanup

Cleanup details are in
`/root/autodl-tmp/ailab/outputs/eval/cube/OOD_DISK_CLEANUP_REPORT.md`.

The authorized video-only cleanup removed `20,566` MP4 files (`486,391,688`
bytes, about `0.45 GiB`). Reports, JSON/CSV, cost NPZ, physical outcomes,
manifests, checkpoints, datasets, and the three documented exception paths were
retained. The data disk has approximately `71 GiB` free. Releasing `40 GiB` was
not possible under the user's deletion rule; the remaining volume is primarily
immutable datasets and checkpoints.

## Part 1: Robust Finetune

Training started from the MaskedAug checkpoint
`/root/autodl-tmp/ailab/checkpoints/lewm-cube-maskedaug/route21_masked_hsv_seed3072/weights_final.pt`
and completed `4,000` steps (the practical frozen budget within the requested
`<=12,732` limit). Configuration was batch `128`, AdamW, learning rate `1e-5`,
bf16 mixed precision, seed `3072`, 90/10 episode split, and the existing masked
cube hue intervention plus:

- low-saturation/dark-background hue shift (`S < 0.40`, `V < 0.35`);
- full-frame gamma jitter in `[0.7, 1.4]`.

The checkpoint is
`/root/autodl-tmp/ailab/checkpoints/lewm-cube-robust_v1/lewm-cube-robust_v1/weights_final.pt`
(SHA256 `cffe41b70ed743c7ecf63610b0ebad2be64d6903572ec31e0379f95800072eed`).
Wall-clock training time was `10,113s` (about `2h48m33s`).
The run plan, completion record, TensorBoard curve, and QC are respectively in:

- `/root/autodl-tmp/ailab/outputs/train/robust_v1/lewm-cube-robust_v1/run_plan.json`
- `/root/autodl-tmp/ailab/outputs/train/robust_v1/lewm-cube-robust_v1/completed.json`
- `/root/autodl-tmp/ailab/logs/tensorboard/robust_v1/lewm-cube-robust_v1/`
- `/root/autodl-tmp/ailab/outputs/train/robust_v1/lewm-cube-robust_v1/qc/`

The QC contains 15 PNGs (five per visual axis) and a contact sheet. The logged
training `pred_loss` decreased from `0.13970` at the first logged point to
`0.00694` at step `4000`; total loss decreased from `0.26763` to `0.08311`.

The paired expert stopline used 34 batches, 140,180 clips, and 815 held-out
episodes. Base loss was `0.00329533848`; robust loss was `0.00305068938`, a
relative change of `-7.42%`, so the `+10%` fuse did not trigger. The measurement
record is
`/root/autodl-tmp/ailab/outputs/train/robust_v1/lewm-cube-robust_v1/expert_stopline.json`.

## Part 1 Results: T2 Visual Matrix

All nine conditions used the same seed-42 formal rows, goal offset `25`, budget
`50`, and the T2 seed/noise protocol. Formal results are in
`/root/autodl-tmp/ailab/outputs/eval/cube/robust_v1/` and the aggregate is
`evaluation_summary.json`.

| condition | axis | success | delta vs T2 baseline |
| --- | --- | ---: | ---: |
| red | regression | 46/50 (92%) | +4pp |
| blue_v2 | regression | 46/50 (92%) | +4pp |
| yellow_v2 | regression | 43/50 (86%) | 0pp |
| floor_red | floor | 24/50 (48%) | stress condition |
| floor_green | floor | 23/50 (46%) | stress condition |
| light_low | light | 30/50 (60%) | stress condition |
| light_high | light | 34/50 (68%) | stress condition |
| camera_minus | camera | 22/50 (44%) | stress condition |
| camera_plus | camera | 21/50 (42%) | stress condition |

Against the paired T2 baseline, red and blue each have `F->S = [22, 31]` and no
`S->F` flips. Yellow has `F->S = [31]` and `S->F = [27]`, net zero. No regression
condition dropped by the `>3pp` report threshold.

The result is a partial robustness win: the trained color axes improved, while
floor and especially camera remain sensitive. Camera was intentionally not
augmented; these results support a future render-based camera augmentation arm,
not a claim that pixel-space augmentation solves geometry/viewpoint shift.

## Part 2: Goal-Space OOD Curve

The evaluator used real H5 target frames, excluded all 50 fixed evaluation
episodes from target retrieval, and kept the same 50 initial rows for every tier.
Outputs are under
`/root/autodl-tmp/ailab/outputs/eval/cube/goal_ood_curve/`, including
`curve.csv`, `curve.json`, and `success_vs_ood_distance.png`.

| requested tier | median actual distance | fallback | success |
| --- | ---: | --- | ---: |
| in_box | 0.00cm | no | 13/50 (26%) |
| +5cm | 5.00cm | no | 11/50 (22%) |
| +10cm | 5.57cm (max 7.02cm) | yes | 6/50 (12%) |
| +20cm | 5.57cm (max 7.02cm) | yes | 6/50 (12%) |

The observed slope from in-box to the real 5cm shell is `-4pp / 5cm`, or
approximately `-0.8pp/cm`. From in-box to the largest fallback population it is
`-14pp / 5.57cm`, approximately `-2.5pp/cm`. There is no defensible slope past
7.02cm because the dataset contains no eligible 10cm or 20cm shell frames.

## Conclusion

The visual-axis finetune is worthwhile but incomplete. It improves the known
color regression axes by `+4pp` each without violating the expert stopline, but
does not solve floor or viewpoint OOD. The goal-space benchmark shows a clear
decline with real distance, while its nominal 10/20cm tiers are dataset-limited
fallback measurements.

Recommended next step: add render-based camera/viewpoint augmentation (or collect
play data with camera variation) before another checkpoint sweep. Do not treat
the +10/+20 fallback rows as evidence that the system has been tested at those
distances; a larger target-position dataset is required for that claim.

## Reproducibility

- robust augmentation: `le-wm/cube_robustaug.py`, SHA256
  `c5101aeda2e44001c7948aa9eb73a558b5b83ea6e29bf4e96e0596f411300701`
- robust training entry point: `le-wm/train_cube_robust.py`, SHA256
  `b857930cd0bde2306dd079db174525e3781cd892c7ebcdf23ee8698496223631`
- robust T2 evaluator: `le-wm/eval_cube_robust.py`, SHA256
  `0799f77e26479d0b7faeff4fd99baf82b9ab34bfd35045f867ba2c4106ffcfc7`
- goal OOD evaluator: `le-wm/eval_goal_ood.py`, SHA256
  `c1233a8aa27d7bed94234a2a99e39692869ef7245b8ac22a5f45434074cbcd0d`
