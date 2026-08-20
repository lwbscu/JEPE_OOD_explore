# 几何航点链 + probe cost 正式评估

## 结论

**几何直线航点链未攻克跨 episode 目标的 50% 天花板，且在所有主考场都负增益。** 同一批 50 局配对下，4 cm 航点链把三档目标 OOD 的 probe 直射 `50/58/52%` 降到 `16/12/18%`；offset100 从新补跑的 probe 直射 `72%` 降到 `68%`；标准 red offset25 从 probe 直射 `94%` 降到 `70%`。没有一个主对比出现净成功翻转：long 为 `+0/-2`，red 为 `+0/-12`，OOD +5 cm 与 fallback 均为 `+0`。

本轮因此没有跨过 70% 门槛。失败主因不是坐标目标表达，也不是静态 probe 精度，而是局部导航闭环：4 cm 航点的首段仅有 `30%~60%` 到达率，25 步超时后大量回退；完整执行 25 步的 action-conditioned 想象 XYZ 误差中位仍为 `49.56~69.07 mm`（OOD/red 航点臂），已经大于 4 cm 航点间距和 2 cm 到达阈值。密集分段进一步消耗预算和引入重规划税。

## 协议与可追溯性

- 世界模型：robust_v1，checkpoint SHA-256 `cffe41b70ed743c7ecf63610b0ebad2be64d6903572ec31e0379f95800072eed`。
- XYZ probe：test median `3.3206 mm`，checkpoint SHA-256 `caa7a92435c01df358382222d22d09313b2b06e45945cd893607764b2db28792`，通过 `<15 mm` 门禁。
- evaluator：`le-wm/eval_waypoint_probe.py`，正式运行 SHA-256 `fc822cf7a848153eb960c4be9811e03d1503abc2c08f80ebefc46b62f2339e21`。
- T2 固定为 300 samples、10 CEM iterations、top-k 30、5 model steps × 5 env steps、seed 42。航点间距主臂 4 cm，到达阈值 2 cm，段超时 25 env steps；任一段超时后永久切为 probe 直射最终坐标。
- planner goal 仅是特权 XYZ；物理环境 target 始终是最终目标。goal 图在 policy info 中为零张量，并在每次 world-model rollout 前移除。
- OOD/red 延续既有 T2 的“当前 episode 排除”检索协议；offset100 延续 B1 的全 50 episode 全局排除。每个因果配对内部的排除协议相同。
- 9 个 smoke 均完成；7 个航点臂都验证了 waypoint switch、timeout fallback、buffer flush 后下一步真实重规划、段日志完整以及 goal 输入全零。2 个 direct 臂按设计没有 switch/fallback。smoke 视频位于 `outputs/eval/cube/waypoint_probe/smoke/<arm>/videos/`。
- 9/9 正式结果、日志及文件 SHA 汇总于 `aggregate_manifest.json`；机器可读总矩阵位于 `success_matrix.csv`。

正式评估采用逐臂串行调用，因此根目录原始 `run_manifest.json` 只描述最后一次 `ood_in_box_waypoint_6cm` 请求，不能当作九臂总 manifest。`aggregate_manifest.json` 专门修正这一汇总口径；`summary.json` 和各臂 `results.json` 是数值来源。

## 全矩阵成功率

| 场景 | 臂 | 成功 | 相对因果基线 | 判读 |
|---|---:|---:|---:|---|
| OOD in_box | 既有 latent cost | 16/50 (32%) | — | 同 rows/targets/model、不同 cost 的上下文参考 |
| OOD in_box | 既有 probe 直射 | 25/50 (50%) | — | 配对基线 |
| OOD in_box | waypoint 2.5 cm | 7/50 (14%) | -36 pp | 消融 |
| OOD in_box | waypoint 4 cm | 8/50 (16%) | -34 pp | 主臂失败 |
| OOD in_box | waypoint 6 cm | 11/50 (22%) | -28 pp | 最好航点间距，仍远低于直射 |
| OOD +5 cm | 既有 latent cost | 9/50 (18%) | — | 同 rows/targets/model、不同 cost 的上下文参考 |
| OOD +5 cm | 既有 probe 直射 | 29/50 (58%) | — | 配对基线 |
| OOD +5 cm | waypoint 4 cm | 6/50 (12%) | -46 pp | 主臂失败 |
| OOD fallback ~5.57 cm | 既有 latent cost | 6/50 (12%) | — | 同 rows/targets/model、不同 cost 的上下文参考 |
| OOD fallback ~5.57 cm | 既有 probe 直射 | 26/50 (52%) | — | 配对基线 |
| OOD fallback ~5.57 cm | waypoint 4 cm | 9/50 (18%) | -34 pp | 主臂失败 |
| offset100 | 新 probe 直射 | 36/50 (72%) | — | 航点因果基线 |
| offset100 | waypoint 4 cm | 34/50 (68%) | -4 pp | 未过 70% |
| red offset25 | 新 probe 直射 | 47/50 (94%) | — | 航点因果基线 |
| red offset25 | waypoint 4 cm | 35/50 (70%) | -24 pp | 明显回归 |

三档既有 latent cost 结果为 `16/50 (32%)、9/50 (18%)、6/50 (12%)`，与对应 OOD 臂共享 rows、targets 和模型，但 cost 不同，因此也只作上下文参考。旧的 offset100 Masked+goal-image 结果为 `36/50 (72%)`，旧的 red robust+goal-image 结果为 `46/50 (92%)`；前者连模型与 cost 都不同，后者 cost 不同。航点增益的因果基线始终是 probe 直射。

## 配对逐 env 翻转

记 `+` 为基线失败→当前成功，`-` 为基线成功→当前失败；env 是固定 50 局向量的 0-based 下标。

| 当前臂 vs 配对基线 | `+` env | `-` env | 净变化 |
|---|---|---|---:|
| in_box d4 vs probe 直射 | 35 | 0,1,4,5,9,10,12,16,19,24,31,33,37,41,42,43,45,46 | -34 pp |
| +5 cm d4 vs probe 直射 | 无 | 1,2,4,6,9,13,16,18,22,24,25,29,31,33,34,35,36,37,41,43,44,45,46 | -46 pp |
| fallback d4 vs probe 直射 | 无 | 5,7,9,13,18,19,23,24,29,31,32,34,36,37,44,45,46 | -34 pp |
| offset100 d4 vs 新 probe 直射 | 无 | 8,20 | -4 pp |
| red d4 vs 新 probe 直射 | 无 | 0,1,6,9,14,17,22,24,29,31,35,46 | -24 pp |
| in_box d2.5 vs probe 直射 | 无 | 1,4,5,9,10,16,19,22,24,31,33,34,37,41,42,43,45,46 | -36 pp |
| in_box d6 vs probe 直射 | 21 | 1,5,9,10,12,16,19,22,31,33,37,42,43,45,46 | -28 pp |
| in_box d2.5 vs d4 | 0,12 | 22,34,35 | -2 pp |
| in_box d6 vs d4 | 0,4,21,24,41 | 22,35 | +6 pp |

上下文对齐也完整保留：offset100 新 probe 直射 vs 旧 goal-image 为 `+21,38,43 / -29,30,46`，净 0 pp；red 新 probe 直射 vs旧 goal-image 为 `+14,17,24 / -26,27`，净 +2 pp。完整表见 `success_matrix.csv`。

## 航点执行统计

“planned”是 50 局按几何距离预生成的总段数；“activated”是预算内实际开始追踪的段数。STALLED 是活动坐标目标距离在连续 8 步下降严格小于 1 cm 的 false→true 事件数，不是 episode 数。

| 臂 | planned | activated | reached | 段到达率 | 到达耗时中位 | timeout | fallback episodes | STALLED events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| in_box d2.5 | 566 | 91 | 42 | 46.15% | 5 steps | 41 | 41/50 | 108 |
| in_box d4 | 361 | 73 | 23 | 31.51% | 7 steps | 46 | 46/50 | 107 |
| in_box d6 | 254 | 71 | 23 | 32.39% | 12 steps | 41 | 41/50 | 110 |
| +5 cm d4 | 501 | 75 | 25 | 33.33% | 7 steps | 46 | 46/50 | 108 |
| fallback d4 | 559 | 78 | 28 | 35.90% | 12 steps | 46 | 46/50 | 108 |
| offset100 d4 | 198 | 58 | 31 | 53.45% | 1 step | 23 | 23/50 | 99 |
| red d4 | 187 | 61 | 27 | 44.26% | 2 steps | 25 | 25/50 | 100 |
| offset100 probe 直射 | — | — | — | — | — | 0 | 0/50 | 50 |
| red probe 直射 | — | — | — | — | — | 0 | 0/50 | 54 |

间距消融揭示了明确的预算/切换税：2.5 cm 把实际到达段数提高到 `42`，但要规划 `566` 段，最终成功率反而最低（14%）；6 cm 只规划 `254` 段，成功率回升至 22%，却仍比直射低 28 pp。换言之，“更容易到达单段”没有转化为“更容易完成最终任务”。

### 失败断点

| 航点臂 | 最终失败的 timeout 断点（segment 0/1/2） | budget exhausted | fallback 后最终成功 |
|---|---:|---:|---:|
| in_box d2.5 | 27 / 8 / 3 | 5 | 3/41 |
| in_box d4 | 30 / 7 / 5 | 0 | 4/46 |
| in_box d6 | 30 / 5 / 1 | 3 | 5/41 |
| +5 cm d4 | 30 / 9 / 1 | 4 | 6/46 |
| fallback d4 | 25 / 11 / 2 | 3 | 8/46 |
| offset100 d4 | 14 / 2 / 0 | 0 | 7/23 |
| red d4 | 12 / 3 / 0 | 0 | 10/25 |

三档 OOD d4 都有 25~30 个最终失败直接断在第 0 段；进入后续段并不是主要瓶颈。即使触发了直射回退，OOD 也只挽回 `4/46、6/46、8/46`，说明前 25 步的错误局部干预已经消耗了半个 50-step 预算。

## 完整 25 步想象误差

这里测量的是 CEM 精确返回的 5×5 动作序列：probe(预测末端隐向量) 的 XYZ 与无中断执行 25 个 env steps 后物理 XYZ 的欧氏误差。发生航点切换、提前成功或预算截断的 solve 均标为 censored，不混入误差统计。

| 臂 | aligned / total | censored | median / p90 (mm) | 成功局 aligned n/median | 失败局 aligned n/median |
|---|---:|---:|---:|---:|---:|
| in_box d2.5 | 70/131 | 61 | 66.32 / 355.57 | 3 / 59.71 | 67 / 77.61 |
| in_box d4 | 77/118 | 41 | 67.35 / 356.28 | 4 / 52.85 | 73 / 78.30 |
| in_box d6 | 71/111 | 40 | 69.07 / 342.23 | 5 / 68.95 | 66 / 69.38 |
| +5 cm d4 | 76/121 | 45 | 68.85 / 407.01 | 6 / 33.97 | 70 / 87.58 |
| fallback d4 | 72/123 | 51 | 49.56 / 511.15 | 8 / 34.83 | 64 / 53.87 |
| offset100 probe 直射 | 114/149 | 35 | 245.05 / 440.69 | 2 / 29.24 | 112 / 248.96 |
| offset100 d4 | 133/177 | 44 | 199.23 / 421.99 | 7 / 37.22 | 126 / 220.94 |
| red probe 直射 | 7/53 | 46 | 113.09 / 180.88 | 1 / 25.65 | 6 / 116.28 |
| red d4 | 38/86 | 48 | 58.30 / 215.05 | 11 / 37.85 | 27 / 70.43 |

Censor 计数（switch/termination/budget）依次为：in_box d2.5 `39/7/15`、d4 `22/8/11`、d6 `21/11/8`、+5 cm `25/6/14`、fallback `27/9/15`、offset100 direct `0/35/0`、offset100 d4 `8/34/2`、red direct `0/46/0`、red d4 `10/35/3`。

成功 episode 经常提前终止，所以成功层 aligned 样本很少；成功/失败分层只能作为方向性诊断，不能据此做显著性推断。但总体证据仍一致：静态 probe 只有 3.32 mm 误差，action-conditioned 物理 rollout 却在局部航点臂达到约 50~69 mm 中位，误差尺度已超过航点本身。

## 最终归因与决策

1. **目标表达已经解决。** 全程零 goal 图，直接坐标 cost 在 red 达到 94%，在 offset100 达到 72%；因此本轮失败不能再归因于跨 episode goal 图连续性。
2. **主要是导航/动力学段，不是静态感知段。** probe 的 held-out XYZ 中位误差 3.32 mm，但完整动作序列的物理预测误差是其一个数量级以上；失败首先集中在 segment 0。
3. **固定直线分段与 25-step timeout 会放大误差。** OOD d4 有 92% episode 触发 fallback；原本有效的直射前 25 步被局部目标替代。更密的 2.5 cm 航点提高段到达率，却增加段数和重规划次数，最终更差。
4. **航点链没有产生新的解。** 三个 OOD 主臂中只有 in_box 多救回 1 局，却分别丢失 18/23/17 局；long 与 red 都是零正向翻转。

因此本机制应判为**负结果并停止扩展**：不继续调直线间距、段超时或增加更多航点。若未来重启，建议先在独立的单航点基准上探索性验证：4~6 cm 局部目标的 25 步到达率显著高于当前水平，完整 25 步 physical XYZ error 中位目标 `<20 mm`（由 2 cm 到达阈值导出的工程目标，非本轮预注册门禁），并确认单航点注入不损伤 red probe-direct 94% 回归。满足这些条件前，问题不在“缺少航点”，而在 predictor 对候选动作的局部可达性排序还不够可靠。

## 产物索引

- 九臂聚合身份：`outputs/eval/cube/waypoint_probe/aggregate_manifest.json`
- 总成功率矩阵：`outputs/eval/cube/waypoint_probe/success_matrix.csv`
- evaluator 汇总：`outputs/eval/cube/waypoint_probe/summary.json`
- 每臂完整结果：`outputs/eval/cube/waypoint_probe/<arm>/results.json`
- 每段 trace：`outputs/eval/cube/waypoint_probe/<arm>/segments.json`
- 想象误差：`outputs/eval/cube/waypoint_probe/<arm>/imagination_error.json`
- CEM cost/trace：`outputs/eval/cube/waypoint_probe/<arm>/cost_history/` 与 `trust_trace.{json,npz}`
- 正式日志：`logs/eval/cube/waypoint_probe/formal/<arm>.log`
