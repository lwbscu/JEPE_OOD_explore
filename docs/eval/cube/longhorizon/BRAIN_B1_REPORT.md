# B1 — Long-Horizon Goal + LLM Slow-Loop Supervisor

## Decision

**B1 produced no control or success-rate gain.** On the selected long-horizon arena (`goal_offset_steps=100`, `eval_budget=200`), the frozen T2 baseline and the DeepSeek slow-loop arm both achieved **36/50 = 72%**. The paired flip sets are empty in both directions.

This is not a trigger-coverage failure. The STALLED trigger fired in all 14 baseline-failure episodes, but all 70 valid LLM calls returned `CONTINUE`; there were no `SUBGOAL` or `RECOVER` decisions. Consequently, the supervisor never changed the formal planner goal and could not affect behavior.

## Frozen protocol

- World model: MaskedAug checkpoint `d64501aa8e7dac1205d3a134c5bd7c160361e16d6da54c79e21e974cdc953117`.
- Planner: frozen T2, with 10 exact memory seeds, 20 raw-domain sigma=0.1 perturbations, 270 free candidates, 300 samples, top 30, 10 CEM iterations, and seed 42.
- Evaluation isolation: the original 50 held-out episodes were retained. A new start step was sampled within each held-out episode for each offset. This avoids evaluating on episodes seen by MaskedAug training.
- Retrieval isolation: both T2 seed retrieval and subgoal-frame retrieval globally excluded all 50 held-out episodes and the current episode; every planning cycle used 10 distinct source episodes.
- LLM boundary: symbolic JSON only; no pixels, candidate actions, latent states, trajectories, or videos were exposed. The LLM did not participate in candidate ranking or selection.
- API: `deepseek-v4-flash`, OpenAI-compatible Chat Completions, `thinking={"type":"disabled"}`, temperature 0.1, strict JSON object, maximum 128 response tokens. These fields match the current [DeepSeek Chat Completions schema](https://api-docs.deepseek.com/api/create-chat-completion) and [thinking-mode toggle](https://api-docs.deepseek.com/guides/thinking_mode).

The implementation is frozen at:

- `le-wm/eval_brain_b1.py`: `1cdf3e8bddc8d3bd2415905bb2f9dbcee2aeb03050f27d437bf4a04100074774`
- `le-wm/tools/brain_supervisor.py`: `57e49844d4cc0d2981a03c9b99fa143cfd565e9f1737bcb79cb8af8d8efd862e`

## Stage 1 — difficulty selection

| Arena | Budget | Success | Rate | Role |
|---|---:|---:|---:|---|
| offset 75 | 150 | 38/50 | 76% | Reference; triggered the predefined harder arena because rate was strictly above 70% |
| offset 100 | 200 | 36/50 | 72% | Selected main arena |

The offset-75 selection row SHA-256 is `954ccf1ee9e6d8bee4ac224d1d0930f81ec4f2bf4f2ba9dda0c15883a82c384e`; the offset-100 row SHA-256 is `0cd9a6fd177d40f62c5d06d5632454f9cad4aeef357158dd393777db505a78ce`. Both contain 50 unique held-out episodes and zero retrieval overlap with the held-out set.

## Stage 2 — paired slow-loop result

| Method | Success | Rate | Delta vs baseline | F to S envs | S to F envs | Evaluator time |
|---|---:|---:|---:|---|---|---:|
| T2 baseline | 36/50 | 72% | — | — | — | 110.16 s |
| T2 + B1 supervisor | 36/50 | 72% | **0 pp** | `[]` | `[]` | 200.33 s |

The 50 evaluation rows and per-environment success vector are exactly paired. The brain result SHA-256 is `0188a837d88a9ca8a8a3362d94a0a9c61c899ef9a8e22f0eba32a1060f1c4ae0`.

## LLM behavior

| Measure | Result |
|---|---:|
| Episodes with at least one trigger/call | 14/50 |
| Baseline failures covered by a trigger | 14/14 |
| Trigger events | 1,097 STALLED; 0 DROPPED |
| Logical calls / HTTP attempts | 70 / 70 |
| Calls in each triggered episode | 5 |
| CONTINUE | 70/70 = 100% |
| SUBGOAL | 0 |
| RECOVER | 0 |
| Protocol failures / model mismatches | 0 / 0 |
| Cooldown-suppressed triggers | 224 |
| Episode-budget-suppressed triggers | 803 |
| Subgoal achievement rate | Not applicable; no subgoal was proposed |
| Success after RECOVER | Not applicable; no recovery was proposed |

The triggers covered exactly the 14 failed baseline environments: `[0, 3, 4, 6, 9, 17, 21, 22, 31, 32, 38, 40, 43, 44]`. Thus, insufficient trigger rate is ruled out for this batch. In fact, once an episode stalled, the persistent trigger rapidly exhausted the five-call allowance. The bottleneck is **decision collapse to CONTINUE**.

Because there were no formal goal switches, this experiment does not identify whether LLM-proposed subgoals would have been geometrically good or whether T2 could have executed them. Those branches were never reached. The symbolic payload was sufficient to represent the observed decision point—block and target pose, end-effector pose, gripper state, distance trend, and planner-cost trend all documented sustained stagnation—so there is no evidence here that vision was needed.

## API cost and runtime

| Measure | Result |
|---|---:|
| Total accounted tokens | 25,831 / 1,000,000 |
| Prompt / completion tokens | 25,341 / 490 |
| Mean calls per episode | 1.40 |
| Mean logical-call latency | 1.223 s |
| Median / p90 / p95 attempt latency | 1.162 / 1.589 / 1.704 s |
| Total logical-call latency | 85.635 s |
| Unknown-token reservation | 0 |

All 70 responses reported the exact requested model `deepseek-v4-flash`, `finish_reason=stop`, no reasoning content, and valid strict JSON. The model name and non-thinking capability are also documented in DeepSeek's [V4 release note](https://api-docs.deepseek.com/news/news260424/).

## Artifact integrity

- Baseline offset 75: 50 results, 50 decodable videos of 150 frames, 50 JSON + 50 NPZ cost histories, 111 planning cycles / 1,110 CEM iteration records.
- Baseline offset 100: 50 results, 50 decodable videos of 200 frames, 50 JSON + 50 NPZ cost histories, 151 planning cycles / 1,510 iteration records.
- Brain offset 100: 50 results, 50 decodable videos of 200 frames, 50 JSON + 50 NPZ cost histories, 151 planning cycles / 1,510 iteration records.
- All candidate costs were finite. Every retrieval cycle had 10 distinct source episodes; fixed-50 and current-episode leakage counts were zero.
- The real-API two-episode smoke additionally forced a diagnostic goal switch after a valid `CONTINUE`: the selected HDF5 frame matched byte-for-byte, the physical target remained unchanged, the action buffer was flushed, replanning occurred one step later, and the controller restored the final goal after reaching the subgoal. This validates the wiring even though the formal LLM never selected that branch.
- No API credential or authorization field was persisted. No evaluation, API, or GPU process remained after completion.

## Final assessment

B1's slow-loop architecture is operational and protocol-clean, but the tested policy is ineffective: **0 pp gain, zero paired flips, and roughly 90 seconds additional evaluator time**. The failure is not that the monitor missed failing episodes; it is that the LLM declined to intervene in every one of them.

For this configuration, the correct conclusion is:

1. Keep MaskedAug + T2 as the deployed controller.
2. Do not add B1 in its present form; it adds latency and token cost without behavioral change.
3. Do not add vision. The logged symbolic states already expose the stalls.
4. If this line is revisited, first replay the frozen 70 failure-state payloads offline and require non-degenerate, schema-valid intervention coverage before spending another 50-environment online run. A revised decision policy would need to distinguish when continued T2 exploration is justified from when a concrete recovery/subgoal is mandatory. That would be a new intervention-policy experiment, not evidence from B1 itself.

