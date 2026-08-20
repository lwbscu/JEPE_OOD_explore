# Cube 颜色增广微调（Route 2）报告

## 结论

颜色增广微调通过 Red 回归门禁，并显著改善 Yellow-v2，但没有改善 Blue-v2：

- Red：72% → 80%（+8pp）
- Blue-v2：64% → 64%（0pp）
- Yellow-v2：62% → 78%（+16pp）

因此，“颜色增广能普遍修复所有颜色 OOD”不成立；它对黄色非常有效，对纯蓝没有净收益。模型内的 OOD 缺口为 Blue 相对 Red -16pp、Yellow 相对 Red -2pp。

## 微调协议

| 项目 | 配置 |
| --- | --- |
| 初始化 | `quentinll/lewm-cube` 原权重 |
| 训练范围 | encoder + predictor 全模型 |
| 正式预算 | 1 epoch，12,732 optimizer steps |
| 数据泄漏纪律 | 固定 50 个正式评估 episode 在训练、验证与 normalizer 中全部排除 |
| 增广 | 每个 4-frame clip 共享同一 hue shift；20% identity，80% Uniform[-180°, 180°] |
| HSV | 只改 H，S/V 不变；作用于整帧 |
| Loss | prediction MSE + 0.09 × SIGReg |
| Optimizer | AdamW，lr=1e-5，weight decay=1e-3 |
| Scheduler | LinearWarmupCosineAnnealingLR，127-step warmup |
| Precision | BF16 mixed；manual norm clipping=1.0 |
| Seed | 3072 |

训练用时约 2 h 34 min。最终训练 pred loss 为 `0.00854`，留出验证 pred loss 为 `0.00341`；最终权重已完成本地重新加载验证。

## 50-env 闭环结果

三组均使用固定 seed42 的同一 50 个 dataset rows、50-step budget、CEM10（300 samples、topk30）以及 legacy updated elite-mean selector。

| 条件 | 原模型 | ColorAug | 变化 | 失败→成功 env | 成功→失败 env |
| --- | ---: | ---: | ---: | --- | --- |
| Red | 36/50 = 72% | 40/50 = 80% | +8pp | 22, 26, 34, 36 | — |
| Blue-v2 | 32/50 = 64% | 32/50 = 64% | 0pp | 26, 36 | 33, 43 |
| Yellow-v2 | 31/50 = 62% | 39/50 = 78% | +16pp | 9, 10, 22, 29, 34, 36, 37, 43 | — |

Red 门禁要求成功率不低于 69%；实际为 80%，通过。

## 与 Memory Seed 的关系

| 条件 | ColorAug | Memory Seed | 差值（ColorAug - Memory） |
| --- | ---: | ---: | ---: |
| Red | 80% | 88% | -8pp |
| Blue-v2 | 64% | 68% | -4pp |
| Yellow-v2 | 78% | 66% | +12pp |

两者不是同一实验臂：Memory Seed 改候选生成，ColorAug 改表示。当前结果说明生成层检索对 Red/Blue 更有价值，而颜色增广对 Yellow 更有效。由于 Route 1 的 probe-cost 离线门禁失败，本轮没有合法的“ColorAug × probe cost”组合组，也不能用上表相加推断组合成功率。

## 产物与完整性

- Checkpoint：`checkpoints/lewm-cube-coloraug/route2_hsv_seed3072/weights_final.pt`（SHA256 `acb060577ce83b310ec0edbf3a8fa8c614a90504c254e9982ad84db1cfa31433`）
- 训练记录：`outputs/train/route2_coloraug/route2_hsv_seed3072/`
- TensorBoard：`logs/tensorboard/route2_coloraug/route2_hsv_seed3072/`
- 三色正式结果：本目录下 `red/`、`blue_v2/`、`yellow_v2/`

三色均有 50 个视频、50 份 cost JSON 与 50 份 NPZ；150 个视频全部逐帧解码为 50 帧、736×288，共 7,500 帧。所有 CEM cost、均值、方差与最终候选数组均为有限值且形状符合协议。

## 下一步

1. Blue-v2 应优先检查“整帧 hue shift”是否把背景/夹爪一并变色而引入不必要的域偏移；下一版可做 cube-only hue augmentation 或训练色分布加权。
2. Yellow 已基本消除相对 Red 的 OOD 缺口（仅 -2pp），可以保留当前增广策略。
3. 若要测试真正叠加，应组合 `Memory Seed × ColorAug encoder`；这比已失败的原 encoder probe-cost 路线更有数据依据，但属于下一实验，不应从本轮结果直接外推。
