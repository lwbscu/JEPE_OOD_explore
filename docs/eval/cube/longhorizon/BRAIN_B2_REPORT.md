# B2 Long-Horizon Monitor Comparison

## Judgment

The LLM monitor is equivalent to the deterministic rule monitor on this task. Both finish at 35/50 (70%), two percentage points below the frozen offset-100 baseline of 36/50 (72%), and their 50-element success vectors are identical. The LLM adds cost but no measurable success gain.

## Frozen Protocol

- Red environment, MaskedAug checkpoint, T2 planner, seed 42, offset 100, budget 200.
- The same frozen 50 episodes, rows, starts, goal rows, and baseline result are used by all arms.
- Memory retrieval excludes the complete fixed-50 episode set and the current episode. Every subgoal is a real HDF5 frame; no synthetic goal image is used.
- B2 evaluator: `eval_brain_b2.py` SHA256 `f46e5d1e64cef3b05ff6ce3c70d3abe5252c99955e7d90f976f58b01177b01b3`.
- Prompt supervisor: `brain_supervisor_v2.py` SHA256 `6c05a61a1312e85f3dedad8e94db2cca2656cffff3c19bcd9bd1281a64003f1e`.
- Prompt replay: `replay_brain_b2_prompts.py` SHA256 `77f638b489af078336ad7f18622ad963c43ec12904996b69f8c58c3b7e6a5be9`.

## Stage 0: Prompt Repair

The 70 frozen B1 trigger payloads were replayed against DeepSeek v4 flash with thinking disabled. Round 1 passed and was frozen; rounds 2 and 3 were not run.

| Check | Result |
|---|---:|
| Calls / valid JSON | 70 / 70 (100%) |
| Non-CONTINUE | 70 / 70 (100%) |
| In-box interventions | 70 / 70 (100%) |
| Decision distribution | SUBGOAL 70 |
| Accounted tokens | 78,131 |
| Mean latency | 997.7 ms |

All 70 calls selected candidate 1 (`progress_two_thirds`). This satisfies the preregistered acceptance gate but shows residual candidate-selection collapse; it is not evidence that the LLM found diverse strategies. The 70 source payloads were all STALLED, so Stage 0 does not empirically cover natural DROPPED decisions.

Artifacts: `outputs/eval/cube/brain_b2/offline_prompt_replay/round_1/`.

## Online Results

| Arm | Success | Rate | vs baseline | F->S | S->F |
|---|---:|---:|---:|---|---|
| Baseline | 36/50 | 72% | -- | -- | -- |
| Rule | 35/50 | 70% | -2 pp | none | env 8 |
| Brainv2 | 35/50 | 70% | -2 pp | none | env 8 |

Rule and Brainv2 success vectors are exactly equal. Relative to the rule arm, Brainv2 has no F->S or S->F flips.

### Rule behavior

- 36 STALLED triggers and 36 rule subgoal interventions across 15 episodes.
- 36 real-frame landings, all leakage-free and verified against HDF5 pixels/pose.
- 0/36 interventions achieved the subgoal within the complete intervention window; 30 timed out and 6 were right-censored by the budget at step 200.
- No DROPPED trigger occurred naturally in the formal 50 episodes.

### Brainv2 behavior

- 63 STALLED triggers; 38 SUBGOAL decisions and 5 CONTINUE decisions.
- 43 logical calls, 49 HTTP attempts, 5 transport failures, and 6 transport retries.
- Every successful response was strict valid JSON; the 5 transport failures were handled as CONTINUE according to the frozen policy.
- 38 real-frame subgoal landings, all leakage-free. Every selected candidate was candidate 1.
- 0/38 interventions achieved the subgoal; 31 timed out and 7 were right-censored by budget.
- No DROPPED trigger occurred naturally in the formal 50 episodes; RECOVER was validated by the forced smoke only.
- Accounted online tokens: 87,552. Reported provider usage was 42,408 with an unknown-token upper bound of 45,144; the supervisor budget accounting is the authoritative number.
- Mean HTTP-attempt latency: 8,908 ms. Calls per episode ranged from 0 to 4; the maximum stayed below the hard limit of 5.

Formal artifacts:

- Rule: `outputs/eval/cube/longhorizon/rule_offset100/`
- Brainv2: `outputs/eval/cube/longhorizon/brainv2_offset100/`
- Logs: `logs/eval/cube/longhorizon/formal/rule_offset100.log` and `logs/eval/cube/longhorizon/formal/brainv2_offset100.log`

Both arms produced 50 readable videos and 50 JSON+NPZ cost histories. No formal log contained `ERROR` or `Traceback`, and no GPU/evaluation process remained after completion.

## Conclusion

The result is **LLM approximately equal to rules**, not LLM greater than rules. The LLM did not recover any baseline failure that the rule arm missed, and it reproduced the rule arm's only regression (env 8). The additional model calls therefore do not justify their latency or token cost for this monitor.

The primary bottleneck is downstream execution: neither controller achieved a real-frame subgoal before its 25-step intervention window. The LLM also remained concentrated on candidate 1 despite the repaired prompt. Future work would need a validated execution-level subgoal controller or a materially different action/goal interface before another prompt iteration is useful. Prompt tuning on the 70 B1 failure payloads is an offline development bias and should not be interpreted as an independent generalization test.
