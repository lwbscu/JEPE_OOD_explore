# Cube Route 2.1 MaskedAug + Memory Seed Combination Report

## Executive conclusion

Route 2.1 passed its pre-registered Blue target. Masking the red cube and hue-shifting only masked pixels raised Blue-v2 from 64% to 76% (+12 percentage points), while Red remained within the regression gate at 74% versus 72%. This supports the low-contrast / cube-background hue hypothesis under the fixed 50-environment protocol.

The triggered Memory Seed × MaskedAug combination reached 84% / 86% / 84% on Red / Blue-v2 / Yellow-v2. It is the strongest single protocol across all three colors (macro average 84.67%). The best per-color results are 92% Red from Memory Seed × Route2 global hue, and 86% / 84% Blue / Yellow from Memory Seed × MaskedAug; this per-color oracle is descriptive, not a deployable selector.

## Complete 50-environment matrix

| Method | Red | Blue-v2 | Yellow-v2 | Macro average |
|---|---:|---:|---:|---:|
| Mean, pretrained | 36/50 (72%) | 32/50 (64%) | 31/50 (62%) | 66.00% |
| Memory Seed | 44/50 (88%) | 34/50 (68%) | 33/50 (66%) | 74.00% |
| Route2 global hue | 40/50 (80%) | 32/50 (64%) | 39/50 (78%) | 74.00% |
| Route2.1 MaskedAug | 37/50 (74%) | 38/50 (76%) | 37/50 (74%) | 74.67% |
| Memory Seed × Route2 | **46/50 (92%)** | 34/50 (68%) | 40/50 (80%) | 80.00% |
| Memory Seed × MaskedAug | 42/50 (84%) | **43/50 (86%)** | **42/50 (84%)** | **84.67%** |

All rows use the same frozen 50 initial states, seed 42, 50-step budget, 300 samples, top-30 distribution update, 10 CEM iterations, and updated elite-mean action selector.

## Route 2.1 training and QC

- Warm start: original `quentinll/lewm-cube` checkpoint, not Route2.
- Training: 1 epoch, 12,732 steps, batch 128, BF16, AdamW, learning rate 1e-5, SIGReg weight 0.09, seed 3072.
- Leakage control: the 50 formal evaluation episodes were excluded before split; 9,100 clips were removed.
- Strict mask: float64 HSV `hue > 0.9`, `saturation > 0.4`, `value > 0.15`.
- Augmentation: 80% random hue shift over the full hue circle, 20% identity; only mask pixels change and S/V are preserved.
- Runtime: 1,629,696 clips / 6,518,784 frames observed; 1,303,145 clips augmented (79.962%); 12,513 empty-mask frames (0.192%); 1,463,100,207 masked pixels processed.
- QC: 10 triptych PNGs plus a 2×5 contact sheet. All ten fixed QC masks were non-empty; background and gripper pixels outside the mask were elementwise identical before and after augmentation.
- Final train prediction loss: 0.007138; validation prediction loss: 0.003340.
- Checkpoint SHA256: `d64501aa8e7dac1205d3a134c5bd7c160361e16d6da54c79e21e974cdc953117`.

## Paired environment flips

Environment indices below are zero-based. `F→S` means baseline failure became success; `S→F` is the reverse.

### MaskedAug versus pretrained Mean

| Color | F→S | S→F |
|---|---|---|
| Red | `[34, 36]` | `[9]` |
| Blue-v2 | `[4, 9, 34, 36, 37, 42]` | `[]` |
| Yellow-v2 | `[6, 10, 29, 36, 37, 43]` | `[]` |

### MaskedAug versus Route2 global hue

| Color | F→S | S→F |
|---|---|---|
| Red | `[]` | `[9, 22, 26]` |
| Blue-v2 | `[4, 9, 33, 34, 37, 42, 43]` | `[26]` |
| Yellow-v2 | `[6]` | `[9, 22, 34]` |

### Memory Seed × MaskedAug versus MaskedAug

| Color | F→S | S→F |
|---|---|---|
| Red | `[6, 9, 26, 27, 38]` | `[]` |
| Blue-v2 | `[0, 1, 6, 27, 38]` | `[]` |
| Yellow-v2 | `[0, 9, 26, 27, 34]` | `[]` |

### Memory Seed × MaskedAug versus Memory Seed

| Color | F→S | S→F |
|---|---|---|
| Red | `[]` | `[0, 1]` |
| Blue-v2 | `[0, 4, 10, 27, 34, 36, 37, 38, 42, 45]` | `[26]` |
| Yellow-v2 | `[0, 10, 27, 34, 36, 37, 42, 43, 45]` | `[]` |

### Memory Seed × MaskedAug versus Memory Seed × Route2

| Color | F→S | S→F |
|---|---|---|
| Red | `[]` | `[0, 22, 24, 31]` |
| Blue-v2 | `[0, 4, 27, 33, 34, 37, 38, 42, 43]` | `[]` |
| Yellow-v2 | `[0, 6]` | `[]` |

## Interaction analysis

Using `interaction = Combo - Memory - Aug + Mean`:

| Combination | Red | Blue-v2 | Yellow-v2 | Macro average |
|---|---:|---:|---:|---:|
| Memory × Route2 global hue | -4 pp | 0 pp | -2 pp | -2 pp |
| Memory × MaskedAug | -6 pp | +6 pp | +6 pp | +2 pp |

Memory retrieval and masked color augmentation have positive interaction on both OOD colors. On Red, MaskedAug is an OOD-specialized intervention and interferes with the already-strong Memory baseline: 84% versus Memory-only 88% and Memory × Route2 92%.

## Artifact validation

- MaskedAug and the two combination matrices contain nine formal groups, each with the frozen 50 rows, 50 videos, 50 cost JSON files, and 50 cost NPZ files.
- All 450 videos decode as 50 frames at 736×288.
- All CEM arrays are finite and have the frozen shapes; returned actions equal the final updated elite mean.
- In every combination replan, the ten Memory sources are from ten different episodes and exclude the current evaluation episode.
- The first-cycle retrieval queries, sources, distances, and actions are identical to the original Memory Seed run; only the world-model checkpoint changes.
- Blue/yellow goal hashes match the frozen recolor arrays. No formal log contains Traceback, ERROR, OOM, or Killed.

## Interpretation and remaining gap

The Blue repair path is operational: MaskedAug fixes the perception-side failure (64%→76%), and Memory Seed adds generation coverage (76%→86%). Yellow shows the same favorable interaction and reaches 84%. The remaining gaps to 100% are 14 pp on Blue and 16 pp on Yellow under the single robust protocol; these are consistent with residual cost-ranking, control, or candidate-coverage failures, not the previously isolated background-contrast issue alone.

For one unified policy, Memory Seed × MaskedAug is recommended. If Red ID performance is the sole objective, Memory Seed × Route2 remains higher at 92%. Selecting a policy after observing the color-specific evaluation result would be an oracle choice and is not reported as a deployable method.
