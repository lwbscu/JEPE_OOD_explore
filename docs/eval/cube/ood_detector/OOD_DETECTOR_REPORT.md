# Cube OOD detector report

## Decision

The requested detector is **not identifiable from the frozen six-axis artifacts**. No simulator or new evaluation was run.  Reporting a hidden-space kNN AUC or a probe-imagination-error AUC would require substituting a different signal or broadcasting a condition-level/candidate-level value to episodes, so those primary AUC cells are reported as `NOT MEASURED` rather than fabricated.

## Requested scores

| Score | Status | Joinable episode scores | Evidence boundary |
|---|---:|---:|---|
| Current-frame control-latent kNN distance | NOT MEASURED | 0 | The shipped memory index is 9D privileged state; variation-specific formal images/latents were not persisted. |
| Probe imagination error | NOT MEASURED | 0 | Existing measurement-2 covers older checkpoints, three colours, and 12 audit environments—not robust_v1 six-axis formal episodes. |

Therefore the present artifacts cannot support the claim that the model can detect its own OOD failures.

## Diagnostic proxies available in the frozen logs

These values are reported only to quantify whether already-logged online quantities correlate with failure.  `privileged_state_knn_min_distance_proxy` uses ground-truth state and is not model self-knowledge.  `first_cycle_elite_mean_cost_proxy` is a goal-conditioned planning cost, not physical probe imagination error.  AUC uses failure as the positive class; 0.5 is chance.

| Axis | Privileged-state kNN proxy AUC | First-cycle CEM-cost proxy AUC | N |
|---|---:|---:|---:|
| color | 0.4775 | 0.6639 | 200 |
| camera | 0.3701 | 0.7431 | 200 |
| light | 0.4377 | 0.8289 | 200 |
| floor | 0.3844 | 0.5375 | 200 |
| size | 0.3827 | 0.9210 | 200 |
| action_noise | 0.4029 | 0.7454 | 200 |

The calibration figure uses five deterministic equal-count bins based only on score ranks; labels are not used to choose thresholds.  Tier-0 controls are repeated within each axis because each axis is interpreted as an independent paired benchmark.

## Instrumentation required for a real detector study

1. Persist the decision-time robust_v1 control latent (or exact current image bytes) for every formal episode before action selection.
2. Build and freeze a robust_v1 latent memory bank with formal50 episodes excluded; do not relabel the privileged 9D state index as latent space.
3. Persist per-episode executed-plan predicted terminal XYZ and matching physical terminal XYZ so probe imagination error has a causal join.
4. Freeze calibration thresholds on a separate split before evaluating six-axis success calibration.

These steps require new instrumentation/evaluation and were not executed because E3 was explicitly frozen to zero simulation and zero evaluation.

## Provenance

- Benchmark summary SHA-256: `58a06d4956c572b6ed8365ad3971139fe7083dc26415e9af6284ed6df2645856`
- Memory metadata SHA-256: `fdf7d064b20ae5ed3b3f013fd0aee314d44b8e1329325cfedac52570a23a37b3`
- Measurement-2 SHA-256: `0de6fdf998409b55dea854649b3f840b1f9cb65a3b966eecc1c636a39dd97dca`
- Analysis is read-only with respect to all benchmark inputs.
