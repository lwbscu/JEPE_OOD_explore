# Cube-double 零样本迁移报告

## 结论

`robust-v1 + T2` 在 OGBench `cube-double-play-v0` 状态上进行纯零样本迁移，正式结果为 **4/50（8%）**。成功 env 为 `[4, 17, 27, 28]`。本实验没有训练或微调；`block_0` 是唯一显式 target coordinate 与物理 success 对象。`block_1` 没有独立显式 target coordinate，也不参与物理 success，但存在于 full-state current/goal pixels，可能影响整图 latent cost。

该结果证明单方块模型在双物体场景中存在非零迁移能力，但 `8%` 远低于单方块三色正式结果 `92/92/86%`，不能声明组合泛化已建立。由于任务、数据源和目标构造均不同，这个差值是能力边界描述，不是与单方块结果的配对回归量。

权威结果：`outputs/eval/cube/cube_double_transfer/formal/results.json`，SHA-256 `c7310757c679323507a520eed18b09cf88f1cfccf13fdf723e03035f8e2b7b73`。

## 数据获取与身份

- 数据集：官方 OGBench `cube-double-play-v0`；发布索引约为 train `284 MB`、val `28 MB`。
- Berkeley 官方端点在本机出现 TLS EOF；按冻结 fallback 从 Hugging Face mirror 获取同名文件，revision 为 `0290b1be6721a8750c77334c316aca998ba4aa8b`。文件符合预期 NPZ schema，且评估前没有本地内容变换；由于未获得 Berkeley 端点文件或官方 checksum，与 Berkeley 端点的逐字节一致性未独立验证。
- 正式评估只读取 train NPZ 的 `qpos`、`qvel` 与 `terminals`；本地 train 文件为 `297,435,656` bytes，SHA-256 `a73d1a33d029cedb8bc170ef94791ec585fa2d9450096f4f2a02b8cfbcf608c9`。
- 下载与来源身份完整记录在 `outputs/eval/cube/cube_double_transfer/prepared/pair_manifest.json` 的 `source` 字段。

## 50 对选择协议

| 项目 | 冻结值 | 实测 |
|---|---:|---:|
| seed | 42 | 42 |
| source episode | 互异 | 50/50 互异 |
| raw goal offset | +125 steps | 50/50 精确 +125 |
| model goal offset | +25 steps | 25 |
| `block_0` 位移 | >=5 cm | 最小/中位/最大 5.762/26.424/50.906 cm |
| `block_1` 位移 | <=2 cm | 最大 1.381 cm |
| 候选 source episodes | - | 997 |
| 候选 pairs | - | 274,866 |
| 正式 pairs | 50 | 50 |

选择顺序为 source episode 的 seed-42 permutation；每个 episode 只取一个 seed-42 选中的合格 start，随后取前 50 个合格 episode。prepared NPZ 为 `outputs/eval/cube/cube_double_transfer/prepared/cube_double_eval_pairs.npz`，SHA-256 `14192599c9d09d37fdb73ce73c4ba057dd5cf6a205a0fab099ca9227c0c03748`。manifest SHA-256 为 `d4033dc843b152bf8dd828943aeebdb743fb02a001f5d50e11bc956914b07b9f`。

## 正式评估协议

- checkpoint：冻结的 robust-v1，SHA-256 `cffe41b70ed743c7ecf63610b0ebad2be64d6903572ec31e0379f95800072eed`。
- world：`swm/OGBCube-v0`，`env_type=double`，`mode=data_collection`，`permute_blocks=false`。
- planner：T2；CEM `300 samples / 10 iterations / top-30`，五步 horizon，selector 为 updated elite mean；memory slots `1..10`，noise slots `11..30`，noise sigma `0.1`。
- budget：50 world steps；成功判据为 `block_0` 到目标的欧氏距离 `<=4 cm`。
- goal：prepared pair 的真实 future pixels/state；只以 `block_0` 为目标。
- 训练：**零训练、零微调**；沿用 single-cube robust-v1 checkpoint 与 memory index。

## 正式结果

| 指标 | 结果 |
|---|---:|
| 成功率 | **4/50（8%）** |
| 成功 env | `[4, 17, 27, 28]` |
| 首次成功步 | `[5, 46, 13, 12]` |
| 执行 world steps | 50 |
| T2 planning cycles / iteration records | 97 / 970 |
| evaluator elapsed | `34.889 s` |
| 视频 | 50/50 |

全部 50 条成功向量、最终 `block_0` 距离、`block_1` 距离、逐 env cost history、物理 trace、T2 trace 和视频均在 `outputs/eval/cube/cube_double_transfer/formal/`。成功向量由 `results.json.metrics.episode_successes` 独立重算得到，四个真值位置与 `rows[].success` 一致。

## 能力边界

1. 双方块场景并非完全不可迁移：50 局中有 4 局达到 `block_0 <=4 cm`。
2. `8%` 表明单方块表示、single-cube memory seed 与五步 T2 规划不足以稳定处理第二个可交互物体带来的组合分布变化。
3. `block_1 <=2 cm` 是 start-goal pair 的数据选择约束；在线 success 不约束 `block_1`，因此结果只回答“能否把 block_0 推到目标”，不回答双目标控制。
4. 这是一组 play-derived、`+125` raw-step、block-1 近静止、block-0-only 的任务；没有同源 single-cube paired control，不能把 `8%` 分解为纯视觉干扰、接触动力学或 memory support 的单一贡献。
5. 后续若研究组合泛化，必须冻结新的 cube-double 配对集，并优先引入多步 action-conditioned 训练目标；本结果不支持直接微调后与本 50 局混用。

## 证据索引

- 条件门：`outputs/eval/cube/cube_double_transfer/part3_gate.json`，SHA-256 `ffba4e6b68ed26da2b631b2b0a617dfd47aca7c578f9294673cbb9bd43057ec6`
- pair manifest：`outputs/eval/cube/cube_double_transfer/prepared/pair_manifest.json`
- prepared pairs：`outputs/eval/cube/cube_double_transfer/prepared/cube_double_eval_pairs.npz`
- 正式结果：`outputs/eval/cube/cube_double_transfer/formal/results.json`
- 逐 env 证据：`outputs/eval/cube/cube_double_transfer/formal/cost_history/`、`trust_trace.json/npz`、`physical_trace.npz`、`videos/`
