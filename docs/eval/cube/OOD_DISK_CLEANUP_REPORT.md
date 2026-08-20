# OOD Work Disk Cleanup

The authorized video cleanup removed 20,566 regenerated MP4 files after checking that each parent audit/evaluation directory retained report or JSON/CSV metadata. Removed bytes: 486,391,688 (about 0.45 GiB).

Retained: all reports, JSON, CSV, cost NPZ, physical outcomes, manifests, checkpoints, datasets, `outputs/eval/cube/ood/goal_compare_env0/`, `quentinll` paths, `longhorizon/diagnosis_work/comparisons/`, and 50 `pretrained` videos whose directory had no companion metadata for a safe deletion decision.

Post-cleanup MP4 count is 50, all under the retained `pretrained` directory. The disk reports approximately 71 GiB free (76,066,492,416 bytes). The requested 40 GiB release was not achievable under the stated video-only deletion rule: the authorized candidates were only about 0.45 GiB. The dominant space remains datasets (about 157 GiB), including the immutable Cube H5, and checkpoints (about 8 GiB); neither was deleted.

Deletion manifest: `/root/autodl-tmp/tmp/ood_video_delete_manifest.txt`.
