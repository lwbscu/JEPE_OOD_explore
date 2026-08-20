# 最终主文档独立校验记录

- 校验对象：`outputs/eval/cube/PLANNING_PROBLEM_ANALYSIS.md`
- 对象 SHA-256：`7c88415d8802989d88feaef8b8dedbec65dc7959ff423394c3f6440780ef6c00`
- 校验日期：2026-08-21
- 权限边界：只读复算主文档及其证据；未修改主文档、代码或 staging，未上传。
- 最终结论：**PASS**。

## 1. 总检查表

| Check | Status | Source / independent method |
|---|---|---|
| 主文档身份 | PASS | 对文件字节直接计算 SHA-256，等于冻结值。 |
| 正式 11 行三色总表 | PASS | 从每臂 50 位 `episode_successes`（原始 Red 用证据文本中的向量）重算成功数、成功率和三色不加权宏平均。 |
| 六节点冻结轨迹 | PASS | 从 11 行重取指定六行并按文档冻结精度格式化。 |
| G/H/I/J 关键数字 | PASS | 逐个读取正式 `results.json`、probe/control 汇总和 Play 离线 gate；未使用主文档数字反推。 |
| 训练 elapsed | PASS | Control/Play 取 `completed.json` 原始秒数；robust 取权威训练报告记录。 |
| 正式评估 elapsed | PASS | 仅求和正式 result 的 elapsed 字段，明确排除 smoke。 |
| LLM token 分账 | PASS | B1/B2 权威报告与 blind-reranker 最终双 provider 响应逐文件求和。 |
| 主文档引用路径 | PASS | 解析全部反引号内项目路径；22 个唯一引用，22/22 当前存在。 |
| 比喻残留 | PASS | 全文精确扫描；禁止词无命中，仅保留用户允许的两个结构标签。 |
| PushT 56G 边界 | PASS | 主文档明确标为用户运维记录；仓内不能核验删除前体积或删除事件。 |

## 2. 正式 11 行三色总表复算

每色均为 50 env；成功率为 `success_count / 50 * 100`，Macro 为三列成功率的不加权平均。

| 配置 | Red | Blue-v2 | Yellow-v2 | 复算 Macro | 主文档 |
|---|---:|---:|---:|---:|---|
| Pretrained mean | 36/50 = 72% | 32/50 = 64% | 31/50 = 62% | 66.00% | MATCH |
| Pretrained top-1 | 32/50 = 64% | 30/50 = 60% | 32/50 = 64% | 62.67% | MATCH |
| Memory Seed | 44/50 = 88% | 34/50 = 68% | 33/50 = 66% | 74.00% | MATCH |
| Global hue | 40/50 = 80% | 32/50 = 64% | 39/50 = 78% | 74.00% | MATCH |
| MaskedAug mean | 37/50 = 74% | 38/50 = 76% | 37/50 = 74% | 74.67% | MATCH |
| Memory x global hue | 46/50 = 92% | 34/50 = 68% | 40/50 = 80% | 80.00% | MATCH |
| Memory x MaskedAug | 42/50 = 84% | 43/50 = 86% | 42/50 = 84% | 84.67% | MATCH |
| MaskedAug + T2 | 44/50 = 88% | 44/50 = 88% | 43/50 = 86% | 87.33% | MATCH |
| robust-v1 + T2 | 46/50 = 92% | 46/50 = 92% | 43/50 = 86% | 90.00% | MATCH |
| No-aug step 12,732 + T2 | 43/50 = 86% | 37/50 = 74% | 37/50 = 74% | 78.00% | MATCH |
| No-aug step 16,732 + T2 | 44/50 = 88% | 36/50 = 72% | 36/50 = 72% | 77.33% | MATCH |

数值来源：

- Pretrained：`outputs/eval/cube/pretrained/evidence/ogb_cube_results.txt` 与 `outputs/eval/cube/ood/{blue_v2,yellow_v2}_cem10/results.json`。
- Top-1：`outputs/eval/cube/ood_select/{red,blue_v2,yellow_v2}_top1/results.json`。
- Memory Seed：`outputs/eval/cube/memory_seed/{red,blue_v2,yellow_v2}_seeded/results.json`。
- Global/Masked：`outputs/eval/cube/{coloraug,maskedaug}/{color}/results.json`。
- 两个组合：`outputs/eval/cube/combo_seedX{coloraug,maskedaug}/{color}/results.json`。
- T2/robust/control：`outputs/eval/cube/trust_region/T2/{color}/results.json`、`outputs/eval/cube/robust_v1/{color}/results.json`、`outputs/eval/cube/control_noaugment/step_{12732,16732}/{color}/results.json`。

所有机器向量均核验长度 50，且向量求和等于各 JSON 的 `metrics.success_count`。

## 3. 六节点冻结轨迹

| 节点 | 独立源值 | 冻结显示值 | Status |
|---|---:|---:|---|
| Pretrained mean | 99/150 = 66.0000% | 66% | MATCH |
| Memory Seed | 111/150 = 74.0000% | 74% | MATCH |
| Route2 global hue | 111/150 = 74.0000% | 74% | MATCH |
| Memory x MaskedAug | 127/150 = 84.6667% | 84.7% | MATCH |
| MaskedAug + T2 | 131/150 = 87.3333% | 87.33% | MATCH |
| robust-v1 + T2 | 135/150 = 90.0000% | 90.00% | MATCH |

冻结轨迹 `66% -> 74% -> 74% -> 84.7% -> 87.33% -> 90.00%` 与主文档一致；它是指定成果节点，不被解释为单 checkpoint 的连续训练 lineage。

## 4. G/H/I/J 关键数字

### G. No-augmentation control

| Checkpoint | Red | Blue-v2 | Yellow-v2 | Macro | Status |
|---|---:|---:|---:|---:|---|
| step 12,732 | 86% | 74% | 74% | 78.00% | MATCH |
| step 16,732 | 88% | 72% | 72% | 77.33% | MATCH |

来源：`outputs/eval/cube/control_noaugment/step_{12732,16732}/{color}/results.json` 的成功向量。

### H. Probe goal cost

| Tier | Masked latent | Robust latent | Robust probe | Probe vs robust | Status |
|---|---:|---:|---:|---:|---|
| in-box | 13/50 = 26% | 16/50 = 32% | 25/50 = 50% | +18pp | MATCH |
| true +5cm | 11/50 = 22% | 9/50 = 18% | 29/50 = 58% | +40pp | MATCH |
| fallback 5.57cm | 6/50 = 12% | 6/50 = 12% | 26/50 = 52% | +40pp | MATCH |

来源：Masked 为 `outputs/eval/cube/goal_ood_curve/{in_box,plus_05cm,plus_10cm}/results.json`；robust latent/probe 为 `outputs/eval/cube/probe_goal_cost/{in_box,plus_05cm,fallback_max}/{latent,probe}/results.json`。Probe test median `3.3206mm` 与 `outputs/eval/cube/CONTROL_AND_PROBEGOAL_REPORT.md` 一致。

### I. Waypoint

| 场景 | Probe direct | Waypoint 4cm | 其他正式消融 | Status |
|---|---:|---:|---:|---|
| OOD in-box | 50% | 16% | 2.5cm 14%；6cm 22% | MATCH |
| OOD +5cm | 58% | 12% | N/A | MATCH |
| OOD fallback | 52% | 18% | N/A | MATCH |
| offset100 | 72% | 68% | N/A | MATCH |
| Red offset25 | 94% | 70% | N/A | MATCH |

来源：`outputs/eval/cube/waypoint_probe/*/results.json` 九个正式臂；成功向量和 `metrics.success_count` 一致。主文档把 probe Red `94%` 仅作 privileged 单色诊断、不纳入正式三色 Macro，口径正确。

### J. Play-v1

| Check | 复算值 | Status |
|---|---:|---|
| Expert stopline relative increase | -16.338259% | MATCH / PASS |
| Red median / tail >40mm | 85.236376mm / 63.0000% | MATCH / FAIL |
| Blue-v2 median / tail >40mm | 118.042575mm / 77.6111% | MATCH / FAIL |
| Yellow-v2 median / tail >40mm | 124.548578mm / 77.3056% | MATCH / FAIL |
| Measurement-1 depth-5 median | 5.161476mm | MATCH / PASS |
| Online groups | 0, NOT RUN | MATCH |

来源：`outputs/eval/cube/play_v1/offline/gate.json` 的 `training_stopline`、`colors.*`、`expert_manifold` 和 `authorization`；候选池 elapsed 取同目录 `summary.json:elapsed_seconds_candidate_pools`。

## 5. 训练与评估 elapsed

| 项目 | 原始复算秒数 | 主文档显示 | Status |
|---|---:|---:|---|
| robust-v1 training | 10,113 | 10,113s = 2h48m33s | MATCH |
| Control phase A | 5,449.934491902590 | 5,449.934492s | MATCH |
| Control phase B | 1,823.654701352119 | 1,823.654701s | MATCH |
| Control total | 7,273.589193254709 | 7,273.589193s | MATCH |
| Play-v1 training | 6,534.750051788986 | 6,534.750052s | MATCH |
| F robust visual, 9 groups | 208.248542547226 | 208.248543s | MATCH |
| F goal-OOD, 4 groups | 129.768700122833 | 129.768700s | MATCH |
| F total, 13 groups | 338.017242670059 | 338.017243s | MATCH |
| G control, 6 groups | 130.111619710922 | 130.111620s | MATCH |
| H probe, 6 groups | 202.843739509583 | 202.843740s | MATCH |
| I waypoint, 9 groups | 448.752599239349 | 448.752599s | MATCH |
| G-I total, 21 groups | 781.707958459854 | 781.707958s | MATCH |
| J candidate pool, 6 model-color cells | 14.705492258072 | 14.705492s | MATCH |

训练来源：robust 为 `outputs/eval/cube/OOD_ROBUSTNESS_REPORT.md:43`；Control 为两个 `outputs/train/control_noaugment/.../completed.json:wall_seconds`；Play 为 `outputs/train/play_v1/play_v1_dyn_seed3072/completed.json:duration_seconds`。评估只累加正式 JSON，排除所有 `smoke/`；Waypoint elapsed 位于 JSON 顶层，其余位于 `metrics.elapsed_seconds`。Play Measurement-1 没有持久化 elapsed，主文档未估算，口径正确。

## 6. LLM token 分账

| 项目 | 独立复算 | Status |
|---|---:|---|
| Blind reranker 最终 DeepSeek batch | 1,877,674 | MATCH |
| Blind reranker 最终 OpenAI batch | 1,632,006 | MATCH |
| Blind reranker 最终双 provider 合计 | 3,509,680 | MATCH `约 3,500,000` |
| B1 online | 25,831 = 25,341 prompt + 490 completion | MATCH |
| B2 online authoritative accounted | 87,552 | MATCH |
| B2 provider / unknown upper | 42,408 / 45,144 | MATCH |
| B1 + B2 online | 113,383 | MATCH |
| B2 offline prompt iteration | 78,131 | MATCH，未混入 online |

Blind reranker 由 `outputs/rerank_pilot/responses/rerank_gpt55_deepseek_20260814_final/{deepseek,openai}/cube_*.json` 各 36 文件的 `usage.total_tokens` 求和；`约 3.5M` 的可核范围是该最终双-provider批次，不代表把所有保留的探索/重试目录重复相加。B1/B2 来源分别为 `outputs/eval/cube/longhorizon/BRAIN_B1_REPORT.md:63-73` 与 `BRAIN_B2_REPORT.md:16-29,50-59`。

## 7. 引用路径与语言扫描

主文档反引号中的 22 个唯一项目路径全部存在：robust checkpoint；11 份 cube 主报告/判决；4 份专题结果/证据目录；PushT 结果证据；rerank 报告；control 训练目录。逐项解析结果为 `22/22 exists, 0 missing`。

扫描结果：`米缸=0`、`秤=0`、字符 `米=0`。唯一相关结构标签是第 59 行 `瓶颈地图（问题分层表）` 和第 397 行 `大脑死因链（失败因果序列）`；二者均为用户明确允许的结构标签，不是待清理比喻。

## 8. PushT 与磁盘记录边界

- `outputs/eval/cube/OOD_DISK_CLEANUP_REPORT.md:3-7` 可核：删除 20,566 个 MP4、486,391,688 bytes（约 0.45GiB），清理后约 71GiB 空闲。
- `logs/eval/pusht_pretrained_console.log:4-6` 只能证明 2026-08-13 曾读取 `datasets/pusht_expert_train.h5`；该 H5 当前不存在。
- 仓内没有删除前文件大小或删除事件记录，因此 PushT `约 56G` 只能标记为**用户提供的运维记录，未由产物核验**。主文档正是这一口径，未将 56G 伪装为仓内审计数字。

## 9. 最终判定

主文档冻结 SHA 对应的正式矩阵、成果轨迹、G/H/I/J 数字、耗时、LLM 分账、路径和语言边界均通过独立复算。未发现会改变结论、表格或成本口径的错误。**FINAL: PASS**。
