# B2 Subgoal Diagnosis

This report is strictly offline. It consumes the completed B2 rule artifact, saved MP4 panels, the HDF5 expert dataset, and the frozen memory index; it does not instantiate a simulator or call a GPU.

## Conclusion

- Parsed exactly 36 rule interventions. All 36 had initial distance below the empirical expert p90 25-step reachability estimate; the evidence does not support a hard window-insufficient diagnosis.
- 32 interventions are labeled composition_mismatch: the planner reduced within-solve CEM cost while the physical subgoal distance made less than 25% progress. Four are negative; none are forced to unknown by missing video steps.
- 24 are labeled switching_oscillation under the predeclared 52-step/4cm target-reuse rule; 2 have no adjacent event with enough evidence.

## Evidence

- Outcomes: `{"aborted_budget_exhausted": 6, "timeout": 30}`.
- Initial distance median/p90: 0.1632/0.2387 m.
- Final distance median/p90: 0.1632/0.2387 m.
- Physical progress median: -9.509e-12 m; p90: 0.0080 m.
- Expert p90 25-step displacement: 0.2836 m (1751200 non-heldout windows).
- Video calibration validation error: median 28.0 mm, p90 61.7 mm; every intervention had a 1.00 visible interior-frame fraction.

## Risks

- Interior physical positions are estimates from the largest red HSV component in the saved agent panel, calibrated against non-heldout expert frames. The 61.7 mm p90 calibration error is material; exact initial/final logged distances are stronger evidence than the interior curve.
- A panel video can hold a visually stable red centroid while the cube is occluded or depth-shifted. No missing frame was silently interpolated; any unavailable measurement is encoded as null/unknown.
- The labels are overlapping operational diagnostics, not causal proof. Composition mismatch is defined by cost improvement plus weak physical progress, while switching oscillation is defined by target reuse and spacing.
