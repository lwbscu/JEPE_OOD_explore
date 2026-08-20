# Play 数据治 off-policy 动力学：终局判决

## 结论

**小脑深造线正式归档。** OGBench 官方 `cube-single-play-v0` 混合微调成功避免了专家遗忘，也通过了专家动作回归门，但没有降低真实 CEM 候选上的五步 off-policy 动力学误差。三色候选门全部失败，因此按预注册的 fail-stop 协议，在线 T2 与 probe-red 评估均未运行。

这轮给出了比 V3 更明确的科学结论：基础设施已稳定完成 5,000 步，三条训练铁律全部满足；失败不再能归因于训练中断或专家灾难性遗忘。官方 play 数据配合单步 teacher forcing 仍未把动力学能力迁移到规划器的 off-policy 五步候选分布。

## 一、决策总表

| 门 | 预注册要求 | Play-v1 结果 | 状态 |
|---|---:|---:|---|
| 实时/最终专家止损 | 相对 step 0 涨幅 `<=10%` | `-16.34%` | PASS |
| Red 候选五步误差 | 中位 `<40mm` 且 `>40mm` 率 `<=31.22%` | `85.24mm` / `63.00%` | FAIL |
| Blue-v2 候选五步误差 | 中位 `<40mm` 且 `>40mm` 率 `<=38.50%` | `118.04mm` / `77.61%` | FAIL |
| Yellow-v2 候选五步误差 | 中位 `<40mm` 且 `>40mm` 率 `<=38.51%` | `124.55mm` / `77.31%` | FAIL |
| 专家动作 Measurement-1 | depth-5 中位 `<=8mm` | `5.16mm` | PASS |
| 聚合门 | 上述全部通过 | `FAIL` | **禁止在线** |

机器判决位于 `outputs/eval/cube/play_v1/offline/gate.json`；其中授权字段为 `fail-stop: no online evaluation authorized`。独立 `validate-gate` 复验按设计拒绝该 checkpoint，日志为 `logs/eval/cube/play_v1/validate_gate.log`。

## 二、Play 数据获取与体检

### 来源与规模

- 官方一手索引：`https://rail.eecs.berkeley.edu/datasets/ogbench/`，数据名 `cube-single-play-v0`。
- 本机访问 Berkeley 站点发生 SSL 失败，实际字节仅经传输镜像 `https://huggingface.co/datasets/ryanhoangt/ogbench_data` 的固定 revision `0290b1be6721a8750c77334c316aca998ba4aa8b` 获取；镜像不被表述为官方数据仓。
- 原始 train/val 共 `1,100` episodes、`1,101,100` 帧，每局 `1,001` 帧；文件仅约 `269MB`，低于 60GB 下载上限。
- Play NPZ 没有 `success` 或 `reward` 字段，因此成功帧占比为 **N/A**；`terminals` 只用于 episode 边界，未被伪作成功标签。

### 分布体检

| 项目 | Play | Expert 对照 |
|---|---|---|
| episodes / frames | `1,100 / 1,101,100` | `10,000 / 2,010,000` |
| Block 范围 m | `[0.2451,-0.3572,0.0126] -> [0.6062,0.3500,0.3471]` | `[0.2444,-0.3556,0.0130] -> [0.6108,0.3550,0.3480]` |
| EE 范围 m | `[0.2443,-0.3500,0.0132] -> [0.6049,0.3501,0.3500]` | `[0.2410,-0.3534,0.0138] -> [0.6063,0.3517,0.3458]` |

五维动作直方图平均重叠系数为 `0.959406`，JS divergence 为 `0.001769 bits`。这说明 play 虽为独立探索采集，但其边际动作和状态覆盖与 expert 很接近；体检本身不能证明它覆盖了 CEM 的困难联合序列。

### 转换与排除

- Train H5：`datasets/ogbench_play/cube_single_play_train_phase0.h5`，`1,000` episodes、`201,000` 帧、`198,000` 窗口，SHA256 `2f26e4231f9442c74147becfcdce07c9cf27e72d7106d88fc7eb951e5b8976a0`。
- Val H5：`datasets/ogbench_play/cube_single_play_val_phase0.h5`，`100` episodes、`20,100` 帧、`19,800` 窗口，SHA256 `d19f919e5a837ad866e792fae5f698ee2cfedc076b6dae9af3de314802537eee`。
- `220,000` 个非末 action block 与原始连续五动作精确一致；qpos、qvel、observation、episode layout 和几何派生值全量通过。
- 与全 expert、固定 50、Measurement-1 的完整 episode hash 交集均为 `0`；三者的 `1e-3` 量化头中尾签名交集也均为 `0`。
- 跨 EGL context 的像素复验只作为 report-only：20 帧中 17 帧 byte-exact，其余最大通道差 1、每帧最多 3 个像素变化；它不参与数据有效性判定。

完整体检见：

- `datasets/ogbench_play/PLAY_DATA_HEALTH.md`
- `datasets/ogbench_play/health_report.json`
- `datasets/ogbench_play/manifest.json`
- `datasets/ogbench_play/validation.json`

## 三、混合微调与三条铁律

### 冻结配置

- Warm start：`checkpoints/lewm-cube-robust_v1/lewm-cube-robust_v1/weights_final.pt`。
- 唯一训练臂：`80 expert + 48 play`，即 `62.5% / 37.5%`；batch `128`，LR `1e-5`，BF16 mixed precision，AdamW，`5,000` optimizer steps，seed `3072`。
- Expert 使用既有 MaskedAug；play 使用 identity pixel transform。
- 仅 `predictor + action_encoder` 可训练，共 `87` tensors / `10,947,716` parameters。
- `encoder + projector + pred_proj` 均冻结并强制 eval；`pred_proj` 的 BN buffers 也纳入哈希。
- 损失为 `0.625*expert_teacher + 0.375*play_teacher + 0.09*shared_target_SIGReg`。训练产物记录 `model_rollout_calls=0`、`rollout_depth=0`；SIGReg 只作用于冻结 target embeddings，对动力学栈梯度为零。
- 实际来源样本精确为 `400,000 expert + 240,000 play`。

### 冻结哈希

| 模块 | 训练前 | 训练后 |
|---|---|---|
| encoder | `45bd98f8...9e553e` | 相同 |
| projector | `40607e1d...85d9e` | 相同 |
| pred_proj | `26545360...a5b6d` | 相同 |

最终权重：`checkpoints/lewm-cube-play_v1/play_v1_dyn_seed3072/weights_final.pt`，SHA256 `12a7ef9d607b2d98f5ab177d8772c3e9b5b4162747890e7d8ec1a51a36089e4c`。

### 实时止损曲线

固定 panel 为同一 `4,352` 个 expert clips、34 batches、无 shuffle、BF16；provenance SHA `ff350adbe655ea648c687924e78afae5aed9c3b6c2b33e74ec0667f82ef70dc0`。

| Step | Teacher pred loss | 相对 step 0 |
|---:|---:|---:|
| 0 | 0.00304231 | 0.00% |
| 500 | 0.00277244 | -8.87% |
| 1,000 | 0.00269973 | -11.26% |
| 1,500 | 0.00266858 | -12.28% |
| 2,000 | 0.00259782 | -14.61% |
| 2,500 | 0.00260691 | -14.31% |
| 3,000 | 0.00258149 | -15.15% |
| 3,500 | 0.00256595 | -15.66% |
| 4,000 | 0.00255532 | -16.01% |
| 4,500 | 0.00254360 | -16.39% |
| 5,000 | 0.00254525 | -16.34% |

曲线：`outputs/train/play_v1/play_v1_dyn_seed3072/stopline_curve.png`。训练记录：`run_plan.json`、`loss_curve.jsonl`、`stopline_history.json`、`frozen_integrity.json`、`completed.json` 位于同目录。

`loss_curve.jsonl` 的前 500 步与末 500 步采样均值对比为：expert pred `-12.26%`、play pred `-29.94%`、加权 pred `-19.84%`；三条 rollout 列全程精确为 0。Total loss 约 `+0.30%` 是因为数值被 `0.09*SIGReg` 主导，而该项按冻结合同对动力学栈梯度为零，不能用 total loss 的绝对值替代 pred/stopline 判断。

### Step-0 基础设施事件

首次启动在 optimizer step 0 的固定面板取数处停止：expert episode 末 clip 的第四个、未被 teacher forcing 使用的 action block 含 terminal padding NaN，旧 finite 检查误查了四块。现场完整归档在 `outputs/train/play_v1/infra_failures/20260820T200449Z_step0_unused_action_gate/`；没有 checkpoint、权重、history 或 optimizer 更新。修复只把 finite 检查限定为真正消费的前三块，保留 terminal clip、未做 `nan_to_num`、样本数与 provenance 均不变；前三块 NaN 的负测仍拒绝。随后同一科学配置从头完成 5,000 步。这是基础设施修复，不是模型或超参数重试。

## 四、Fail-stop 离线门禁

### 同一 3 x 12 x 300 候选池

候选覆盖为两模型 × 三色 × 12 env × 300，共 `21,600` 行，键集合精确、无重复、数值全 finite。Masked 列是冻结历史参考，Robust 与 Play-v1 在同一池上本轮配对重算。

| 条件 | Masked 历史中位 mm | Robust 基座中位 mm | Play-v1 中位 mm | Play 相对 Robust | Play `>40mm` | 半尾阈值 | 结果 |
|---|---:|---:|---:|---:|---:|---:|---|
| Red | 85.72 | 85.29 | 85.24 | -0.05 | 63.00% | 31.22% | FAIL |
| Blue-v2 | 112.48 | 112.08 | 118.04 | +5.96 | 77.61% | 38.50% | FAIL |
| Yellow-v2 | 123.36 | 121.82 | 124.55 | +2.72 | 77.31% | 38.51% | FAIL |

相对 Robust 的 `>40mm` 比例变化分别为 Red `-0.31pp`、Blue-v2 `+0.42pp`、Yellow-v2 `-0.06pp`。这不是接近门槛的失败：三色中位均远高于 40mm，尾率约为允许值的两倍。

门禁图：`outputs/eval/cube/play_v1/offline/offpolicy_gate_curves.png`。完整逐候选记录：`candidate_scores.csv`；聚合：`summary.json`。

### 成功/失败候选分层

| 条件 | 最终成功候选数 | 成功中位 / `>40mm` | 最终失败候选数 | 失败中位 / `>40mm` |
|---|---:|---:|---:|---:|
| Red | 1,202 | 18.65mm / 10.98% | 2,398 | 130.54mm / 89.07% |
| Blue-v2 | 1,031 | 68.87mm / 65.08% | 2,569 | 140.66mm / 82.64% |
| Yellow-v2 | 1,014 | 66.94mm / 63.21% | 2,586 | 136.43mm / 82.83% |

Red 的成功候选可被较好想象，但失败候选仍严重幻觉；Blue/Yellow 连成功候选的中位也超过 40mm。Play-v1 没有改变这一分层结构。

### Measurement-1 专家动作回归

固定 2,000 段 × 两模型 × 五深度，共 `20,000` 行；固定 50 排除、seed 42、动作 teacher forcing 均已绑定。

| 深度 | Robust 基座中位 E_roll mm | Play-v1 中位 E_roll mm |
|---:|---:|---:|
| 1 | 4.19 | 4.02 |
| 2 | 4.69 | 4.37 |
| 3 | 5.22 | 4.76 |
| 4 | 5.46 | 4.88 |
| 5 | 5.70 | **5.16** |

Play-v1 在专家流形上略有改善且通过 `<=8mm` 门。这同时排除了“训练总体没学到”与“专家流形被破坏”两种解释；失败集中在 planner off-policy 多步候选分布。

## 五、在线评估状态

| 在线臂 | 既有对照 | Play-v1 | 逐 env 翻转 |
|---|---:|---|---|
| T2 Red offset25 | Robust-v1 `46/50 = 92%` | NOT RUN | N/A |
| T2 Blue-v2 offset25 | Robust-v1 `46/50 = 92%` | NOT RUN | N/A |
| T2 Yellow-v2 offset25 | Robust-v1 `43/50 = 86%` | NOT RUN | N/A |
| Probe direct Red offset25 | Robust-v1 `47/50 = 94%` | NOT RUN | N/A |

`outputs/eval/cube/play_v1/` 下只有 `offline/`；没有颜色或 probe 在线结果目录。未运行不是缺项，而是 fail-stop 设计的正确结果。

## 六、四轮完整记录

| 轮次 | 数据/目标 | 专家流形 | Off-policy 候选 | 结论 |
|---|---|---|---|---|
| V1 | 合成 Gaussian/T2/AR1，单步损失 | `0.003402 -> 0.005822`，`+71.1%` | Red `85.72 -> 43.15mm`，Blue `112.48 -> 94.88mm`，Yellow `123.36 -> 78.30mm`；仍未过门 | 误差下降但灾难性遗忘，关在线 |
| V2 main | 真实规划数据，expert 与 V2 共用多步 rollout | `0.003295 -> 0.012589`，`+282.0%` | 止损后未测 | 失败 |
| V2 retry | 真实规划数据，提高 expert 比例 | `0.003295 -> 0.043268`，`+1213.0%` | 止损后未测 | 更差，停止模型重试 |
| V3 | rollout loss 仅施加于 V2 | 两次均未到 step 500 | 无有效 checkpoint | 基础设施失败，科学结论未决 |
| Play-v1 | 官方 play，严格单步 TF，只训动力学栈 | 最终 `-16.34%`；M1 `5.16mm` | `85.24/118.04/124.55mm`，实质不变或变差 | **科学失败；终局归档** |

V1/V2/V3 的原始终审记录见 `outputs/eval/cube/OFFPOLICY_FINAL_VERDICT.md`；Play-v1 用稳定运行补齐了 V3 留下的基础设施不确定性，但结果表明，仅换成官方 play 数据并遵守安全单步训练，仍不足以学习 CEM 候选所需的五步 off-policy 动力学。

## 七、最终能力边界与归档理由

本轮最重要的正结果是：冻结视觉表征、只训 predictor/action_encoder、使用单步 teacher forcing，确实可以安全保留甚至略微改善 expert 动力学。最重要的负结果是：这种安全训练几乎没有改善 CEM 候选，Blue-v2 与 Yellow-v2 的中位误差还上升。

完整死因链为：

1. V1 证明强 off-policy 监督可以降一部分候选误差，但会破坏专家流形；
2. V2 证明共享多步 rollout 目标会更严重地破坏专家流形；
3. V3 没有产生科学 checkpoint；
4. Play-v1 在稳定基础设施、零 rollout loss、冻结视觉栈和严格止损下保住了专家流形，却没有跨越候选分布鸿沟；
5. 因而现有 LeWM predictor 的能力边界是：专家/近策略动作约 5--6mm，但规划器 off-policy 五步序列仍约 85--125mm，无法靠本项目已验证的数据混合与安全微调方案根治。

**最终决定：归档小脑深造线，不再自动追加训练臂、损失改造或在线重试。** 若未来重新开启，必须是新的研究假设和明确授权，例如能直接覆盖 planner 联合五步序列、同时证明 expert retention 的数据/模型方案；不能把本轮失败解释为“再多训一点”即可解决。

## 八、关键身份与路径

- 数据准备代码：`le-wm/tools/prepare_cube_play_v1.py`，SHA256 `dc3369c84a52ace34bb81589a69d07151a1012435b7f01028c94df811ba56fd1`。
- 数据/loader：`le-wm/cube_play_v1.py`，SHA256 `5445995427292f6d7622029ab3e93c1d543d934a295dc499f77910e60b327b5c`。
- 训练入口：`le-wm/train_cube_play_v1.py`，SHA256 `cafdd1674ae2e522b6af90b638bda8c12d3900839ed21b6cc374ca72180685f2`。
- 离线门禁：`le-wm/tools/evaluate_cube_play_v1.py`，SHA256 `dd1dbf9fec79580f92639e28e8cf6062f65e17d49f87712f02be6f7747192248`。
- 在线入口（未运行）：`le-wm/eval_play_v1.py`，SHA256 `a270439f2870283dfbf656d0abef3ae7b3372d5e9ce491d13a2d99688bb5ce4d`。
- 正式训练：`outputs/train/play_v1/play_v1_dyn_seed3072/`。
- 正式 checkpoint：`checkpoints/lewm-cube-play_v1/play_v1_dyn_seed3072/`。
- 离线门禁：`outputs/eval/cube/play_v1/offline/`。
- 正式日志：`logs/train/play_v1/play_v1_dyn_seed3072.log`、`logs/eval/cube/play_v1/offline_gate.log`。
