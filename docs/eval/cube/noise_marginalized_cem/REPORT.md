# Execution-noise marginalized CEM

M=8 independent candidate execution-noise rollouts; paired seed-42 fixed-50 evaluation.

| Color | sigma | Vanilla | Marginalized | Delta | F→S | S→F |
|---|---:|---:|---:|---:|---:|---:|
| red | 0.2 | 37/50 (74.0%) | 37/50 (74.0%) | +0.0pp | 3 | 3 |
| blue_v2 | 0.2 | 36/50 (72.0%) | 34/50 (68.0%) | -4.0pp | 3 | 5 |
| red | 0.3 | 31/50 (62.0%) | 31/50 (62.0%) | +0.0pp | 1 | 1 |
| blue_v2 | 0.3 | 31/50 (62.0%) | 28/50 (56.0%) | -6.0pp | 1 | 4 |

Every comparison uses identical formal rows and stateless executed-noise seeds on the common action-call prefix. Candidate-noise RNG is isolated from the legacy CEM generator.
