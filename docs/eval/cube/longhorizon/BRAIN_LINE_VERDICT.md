# 大脑线终审判决

## 判决

**大脑线正式归档。** 连续性修复版规则臂的 2-env 物理 smoke 未通过前置条件：6 次 STALLED 子目标干预中 `0/6` 到达子目标；若按全部干预（含 1 次 DROPPED 恢复）统计，则为 `0/7`。6 次 STALLED 中 1 次、1 次 DROPPED 均为 smoke 强制双事件诊断，另 5 次 STALLED 为自然触发。最终成功 `0/2`。因此按预注册执行序列，没有运行修复版规则 50 局，也没有运行条件性的 LLM 臂。

这不是编译门或静态门造成的停止。连续性筛选已真实用于 HDF5 帧选择，动态 goal 已进入 T2 重规划；停止依据是物理结果文件中 `intervention_achievement_rate=0.0`。

## 冻结协议与执行状态

- 环境/规划：MaskedAug 权重、T2、红色环境、`offset=100`、`budget=200`、`seed=42`。
- 评估清单：冻结的同一 50 个 held-out episode；smoke 使用其中 formal index `0` 与 `3`。
- 唯一修复：STALLED 真实帧先要求 HDF5 `proprio_effector_pos` 与 live EE 距离 `<=0.10m`，再沿原 B2 midpoint 几何距离与 row 稳定排序。
- 未采用：自适应窗口、目标导向种子、额外 prompt 修改或任何后续机制重试。
- 修复实现：[eval_brain_subgoal_fix.py](/root/autodl-tmp/ailab/le-wm/eval_brain_subgoal_fix.py)，SHA256 `3715e4a50a0f430be77677ffe3d585c60839436bf1f0e310861481add10932eb`。

## 三臂正式结果与修复臂状态

| Arm | Scope | Success | vs baseline | 子目标到达 | 状态 |
|---|---:|---:|---:|---:|---|
| offset100 baseline | 50 | 36/50 (72%) | reference | N/A | 已完成 |
| B2 rule | 50 | 35/50 (70%) | -2pp | 0/36 | 已完成 |
| B2 LLM | 50 | 35/50 (70%) | -2pp | 0/38 | 已完成 |
| continuity rule | 50 | NOT RUN | N/A | N/A | smoke 失败，按协议禁止 formal |
| continuity rule smoke | 2 | 0/2 | 非正式比较 | STALLED 0/6；全部干预 0/7 | FAIL |
| continuity LLM | 50 | NOT RUN | N/A | N/A | 规则 formal 未被授权，条件不成立 |

正式结果来源：

- [baseline results](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/baseline_offset100/results.json)
- [B2 rule results](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/rule_offset100/results.json)
- [B2 LLM results](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/brainv2_offset100/results.json)
- [continuity smoke results](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/smoke/rule_continuity_offset100/results.json)
- [continuity smoke outcomes](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/smoke/rule_continuity_offset100/intervention_outcomes.json)
- [continuity smoke retrievals](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/smoke/rule_continuity_offset100/subgoal_retrieval.json)

## 逐 env 配对翻转

以 baseline 的 50 局成功向量为参照：

| Arm | F -> S | S -> F |
|---|---|---|
| B2 rule | `[]` | `[8]` |
| B2 LLM | `[]` | `[8]` |
| continuity rule formal | NOT RUN | NOT RUN |

B2 rule 与 B2 LLM 的成功向量逐位相同。Continuity smoke 的两个 formal source indices `[0,3]` 在 baseline 与 smoke 中都失败，因此 smoke 子集翻转为双向空集；该 2 局结果不外推成 50 局成功率。

## 子目标到达率：修复前后

| Version | Interventions | Achieved | Timeout | Budget/other close |
|---|---:|---:|---:|---:|
| B2 rule | 36 | 0 (0%) | 30 | 6 |
| continuity smoke, STALLED 子目标 | 6 | 0 (0%) | 6 | 0 |
| continuity smoke, 全部干预（含 DROPPED） | 7 | 0 (0%) | 6 | 1 |

Continuity smoke 的触发分布为 `STALLED=6, DROPPED=1`，每局干预次数为 `[3,4]`；其中 `STALLED=1` 与 `DROPPED=1` 是 `--force-smoke-both-events` 的强制诊断，另外 5 个 STALLED 是自然触发。6 个 STALLED 选择均记录了真实 HDF5 EE 距离不超过 10cm；观测值约为 `8.86--9.97cm`。也就是说，代码没有静默回退到原先的 18cm 级断裂帧，但这一约束仍未使任何子目标在物理窗口内可达。

## 完整死因链

1. **B1：决策坍缩。** 事件触发和 API 链路正常，但 70/70 决策均为 CONTINUE，LLM 没有产生控制增量。
2. **B2：让干预发生后仍不落地。** prompt 修复后规则与 LLM 都实际切换 real-frame goal，但正式成功率均为 70%，且子目标到达率分别为 `0/36` 与 `0/38`。
3. **离线诊断：构图断裂。** [SUBGOAL_DIAGNOSIS.md](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/SUBGOAL_DIAGNOSIS.md) 显示原规则 36/36 内部 planner cost 下降而方块不动，目标帧 EE 跳变中位 18.1cm，33/36 超过 10cm；窗口不足不是主因。
4. **连续性修复：必要但不充分。** 新 selector 确实把 EE 跳变压到 10cm 内并重新规划，但 smoke 的 STALLED 子目标仍为 `0/6` achieved，全部干预为 `0/7` achieved。结果说明单点 EE 位置接近不能保证整张 goal 图与当前状态在抓持关系、机械臂姿态、方块/夹爪拓扑和可执行短程轨迹上连续。
5. **终审停止。** smoke 没有达到“至少一次真实子目标到达”，所以没有科学或协议依据烧 50 局 formal，更没有依据调用冻结 LLM 臂。

因此当前证据不支持继续通过 prompt、LLM 决策器或单个位置阈值延长这条线。若未来重启，前置条件应是一个能证明局部可达性的 goal 接口（例如带动作/状态连通性标签的短程真实帧对），而不是再次微调监督文本或扩大在线预算。

## Goal 协议最终版三规则

1. **真实帧：** goal 像素必须来自数据集真实 HDF5 帧，禁止合成场景构图。
2. **受控 recolor：** 仅允许已验证的目标身份换色；保持几何、遮挡和场景结构不变，并绑定源帧身份。
3. **EE 物理连续：** 候选真实帧必须满足当前到目标 EE 的连续性约束。此次终审同时证明：`EE <=10cm` 是必要筛选，不是可达性充分条件；未来接口还必须显式验证抓持/姿态/短程连通性。

## 产物完整性与未执行项

- 纯离线诊断表：[interventions.csv](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/diagnosis_work/interventions.csv)
- 纯离线汇总：[summary.json](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/diagnosis_work/summary.json)
- 六张并排构图核验图：[comparisons](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/diagnosis_work/comparisons)
- 修复 smoke 完整输出：[rule_continuity_offset100](/root/autodl-tmp/ailab/outputs/eval/cube/longhorizon/smoke/rule_continuity_offset100)
- `outputs/eval/cube/longhorizon/rule_continuity_offset100/` 未创建，证明 formal 没有运行。
- `brainv3_offset100/` 未创建，证明 LLM 条件臂没有运行。

Smoke manifest 的顶层运行合同明确为 `goal_offset_steps=100`、`eval_budget=200`、`seed=42`。其中继承的 `selection.fixed_manifest_selection.goal_offset=25` 只是 B1 held-out 清单的源选择元数据，不是本次实际 goal offset；该字段未用于本次停止判定，已在此披露以避免误读。
