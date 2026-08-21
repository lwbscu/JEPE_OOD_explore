# OOD 终局报告：重渲染、全轴基准、物理鲁棒性与组合迁移

## 终局判决

本轮完成了可审计的状态重渲染数据、robust-v2 训练 fail-stop、robust-v1 六轴四档正式基准，以及唯一一次预注册的 robust-v2b 训练修复；终局扩展还完成了质量/摩擦物理轴、执行噪声边际化 CEM、OOD 自检可测性审计与 cube-double 零样本迁移。**robust-v2 在 step 500 触发专家止损，teacher pred loss 从 `0.0030423054` 升至 `0.0045427407`，相对上涨 `49.319%`；robust-v2b 放开 encoder/projector 后同样在 step 500 触发止损，teacher pred loss 从 `0.0030423054` 升至 `0.1441741750`，相对上涨 `4,638.978%`。两者都远超 `10%` 红线，均无 `weights_final.pt`，各自预注册的 11 组在线评估均为 `NOT RUN`。** 两个隔离的 step-500 权重都不是可部署模型。

当前部署模型保持为 `robust-v1 + T2`：Red / Blue-v2 / Yellow-v2 为 `92% / 92% / 86%`，三色宏平均 `90.00%`。在新的 full-state 重渲染控制口径下，其非颜色 tier0 为 `90%`；相机与地板最敏感，tier1 已分别降至 `54%` 与 `38%`。质量倍率 `0.5/2/4` 在 red 与 blue-v2 均保持 `92%`；摩擦倍率 `2/4` 令 red 降至 `88%`，blue-v2 仍为 `92%`。M=8 执行噪声边际化在四组内部配对中 `0/4` 达到 `+5pp`，且 blue-v2 退化 `4/6pp`。cube-double 零样本迁移为 `4/50（8%）`，是正式测得的组合泛化边界。

重渲染单步微调线至此正式关闭，不再开启第三种配置。当前已知能力边界由 robust-v1 六轴与物理轴基准共同给出：相机 `+3.333 deg` 为 `54%`，地板 `alpha=1/3` 为 `38%`，两轴均无非零 70% 安全边界；质量与摩擦在已测倍率内保持 `>=88%`，但 Measurement-2 的四个物理条件仍显示五步 `E_roll` 中位数约 `236.7--236.8 mm`、`>40 mm` 比例 `99.806%`，说明在线成功率没有消除 off-policy 想象误差。

证据：`outputs/train/robust_v2/robust_v2_rerender_seed3072/completed.json`、`outputs/train/robust_v2b/robust_v2b_rerender_seed3072/completed.json`、`outputs/eval/cube/ood_benchmark/benchmark_summary.json`、`outputs/eval/cube/physics_ood/summary.json`、`outputs/eval/cube/noise_marginalized_cem/summary.json`、`outputs/eval/cube/ood_detector/summary.json`、`outputs/eval/cube/cube_double_transfer/formal/results.json`。

## 1. 冻结协议与结果口径

- 模型：robust-v1 checkpoint `checkpoints/lewm-cube-robust_v1/lewm-cube-robust_v1/weights_final.pt`，SHA-256 `cffe41b70ed743c7ecf63610b0ebad2be64d6903572ec31e0379f95800072eed`。
- 规划：T2，CEM `300 samples / 10 iterations / top-30`，五个 action blocks；memory slots `1..10`，memory-noise slots `11..30`，噪声 `0.1`，selector 为 updated elite mean。
- 配对：seed `42` 的冻结 50 rows；每个 env 翻转为 `2pp`。完整 rows 与逐位成功向量保存在 `outputs/eval/cube/ood_benchmark/benchmark_summary.json` 和 `paired_flips.json`。
- 渲染：除颜色外，各视觉轴与尺寸轴都把真实 current state 与对应 future state 的完整 `qpos+qvel` 放回仿真器，再在同一 variation 下渲染 current/goal；尺寸变化同步修正 cube 高度。颜色沿用真实 H5 goal 与受控 cube-mask recolor。
- 动作噪声：T2 返回物理动作后、`env.step` 前注入，共享标准高斯样本并按环境动作范围裁剪；planned/noise/executed/clip mask 均落盘。
- 斜率：endpoint 为 `(T3-T0)/(distance3-distance0)`；OLS 为四点普通最小二乘。颜色距离是预注册的类别顺序，只能按 ordinal tier 解读。
- 边界：部署边界要求从 T0 起连续保持在本模型 T0 `-3pp` 内；能力边界要求从 T0 起连续 `>=70%`。二者都是四个离散档上的经验边界，不是连续空间保证。

协议与身份来源：`outputs/eval/cube/ood_benchmark/robust_v1/benchmark_invocation.json`、各 condition 的 `render_protocol.json`。

## 2. Part 1：重渲染数据与 robust-v2

### 2.1 数据与 QC

| 项目 | 正式结果 | 状态 |
|---|---:|---|
| 数据规模 | 25,000 clips / 100,000 frames | PASS |
| clip 时间点 | `t,t+5,t+10,t+15` | PASS |
| 动作块精确比对 | 75,000 blocks / 1,875,000 values | PASS |
| variation bank | 256 组，seed `20260821`；每组 97--98 clips | PASS |
| 评估 50 episode overlap | 0 | PASS |
| Measurement-1 overlap | 0 | PASS |
| block / EE 最大几何误差 | `0.000 mm / 1.527 mm` | PASS，阈值 `0.02 / 2 mm` |
| H5 大小 | 3,263,498,388 bytes | PASS |
| 实测峰值 | 6,548,081,616 bytes | PASS，低于 30 GiB |

正式 H5 为 `datasets/cube_rerender_v2/cube_rerender_v2.h5`，SHA-256 `6f9275e1b1d3a4ea3d9d3e5cd37a57ddde76f644e99743987737da0dd96e1967`。五张 QC 图为 `datasets/cube_rerender_v2/qc/qc_clip_00.png` 至 `qc_clip_04.png`；机器校验总表为 `datasets/cube_rerender_v2/validation.json`，`valid=true`。数据只做 `set_state` 后重渲染，没有动作 rollout；每个四帧 clip 共享 camera/light/floor/size variation。

### 2.2 训练铁律与 fail-stop

| 项目 | 冻结值 / 实测值 | 状态 |
|---|---:|---|
| 初始权重 | robust-v1 | 冻结 |
| batch | original 64 + rerender 64 = 128 | 500 steps 均精确 |
| 训练数据消费 | 32,000 original + 32,000 rerender examples | 精确 |
| 可训练栈 | predictor + action_encoder，10,947,716 params | PASS |
| 冻结栈 | encoder + projector + pred_proj，7,086,912 params | 前后 SHA 完全一致 |
| 损失 | 3 个单步 teacher-forcing transitions；rollout depth `0` | PASS |
| model rollout calls | 0 | PASS |
| stopline baseline | `0.0030423054` | step 0 |
| stopline step 500 | `0.0045427407`，`+49.319%` | **STOPLINE_FAIL** |
| 执行时长 | `952.112s` | step 500 终止 |
| 最终权重 | `null` | **NOT PRODUCED** |

冻结哈希、loss contract 和 source counts 分别来自 `frozen_integrity.json` 与 `completed.json`。训练严格按预注册规则终止；`checkpoints/lewm-cube-robust_v2/.../weights_stopped_step500.pt` 是隔离的负结果证据，不是 `weights_final.pt`，也未授权任何在线评估。

### 2.3 11 组视觉参考矩阵

robust-v2 因 stopline 未通过，**11 组 robust-v2 评估全部 `NOT RUN`**。为固定新 full-state 协议本身的参考值，同批 50 rows 补跑 robust-v1；此表是 robust-v1 reference，不是 robust-v2 成绩。

| Condition | Axis | robust-v1 reference | robust-v2 | 状态 |
|---|---|---:|---:|---|
| red | regression | 46/50 (92%) | NOT RUN | v2 stopline 未授权 |
| blue_v2 | regression | 46/50 (92%) | NOT RUN | v2 stopline 未授权 |
| yellow_v2 | regression | 43/50 (86%) | NOT RUN | v2 stopline 未授权 |
| floor_red | floor | 26/50 (52%) | NOT RUN | v2 stopline 未授权 |
| floor_green | floor | 23/50 (46%) | NOT RUN | v2 stopline 未授权 |
| light_low | light | 41/50 (82%) | NOT RUN | v2 stopline 未授权 |
| light_high | light | 42/50 (84%) | NOT RUN | v2 stopline 未授权 |
| camera_minus | camera | 23/50 (46%) | NOT RUN | v2 stopline 未授权 |
| camera_plus | camera | 21/50 (42%) | NOT RUN | v2 stopline 未授权 |
| size_small | size | 37/50 (74%) | NOT RUN | v2 stopline 未授权 |
| size_large | size | 37/50 (74%) | NOT RUN | v2 stopline 未授权 |

机器可读入口为 `outputs/eval/cube/ood_benchmark/robust_v1/part1_invocation.json`；11 个 `results.json` 均在其 `part1_reference/<condition>/` 下。回归三色与既有 robust-v1 `92/92/86%` 完全一致。由于不存在 robust-v2 对照，地板、光照、相机和尺寸的“增益”与“回归”都必须写为 `NOT MEASURED`，不能用 smoke 或隔离权重推断。

旧 `outputs/eval/cube/OOD_ROBUSTNESS_REPORT.md` 中 floor/light/camera 是早期 goal 构造口径；本轮 reference 使用完整 future `qpos+qvel` 重渲染，因此不可将两者差值归因于模型改善或退化。

### 2.4 robust-v2b：放开视觉表征后的预注册修复

robust-v2b 只改变一个变量：在保持同一 64/64 数据、单步 teacher-forcing、零 rollout 和 500 步止损的前提下，将 encoder + projector 与 predictor + action_encoder 一并训练，仅冻结 pred_proj。该修复没有缓解遗忘，反而在首个止损点造成更严重的专家回归。

| 项目 | 冻结值 / 实测值 | 状态 |
|---|---:|---|
| batch | original 64 + rerender 64 = 128 | 500 steps 均精确 |
| 数据消费 | 32,000 original + 32,000 rerender examples | 精确 |
| 可训练栈 | encoder + projector + predictor + action_encoder；291 tensors / 17,241,860 params | PASS |
| 冻结栈 | pred_proj；6 tensors / 792,768 params | 前后 SHA 完全一致 |
| 损失 | 3 个单步 teacher-forcing transitions + SIGReg；rollout depth/calls `0/0` | PASS |
| stopline baseline | `0.0030423053814207807` | step 0 |
| stopline step 500 | `0.14417417500825488`，`+4,638.978%` | **STOPLINE_FAIL** |
| 执行时长 | `967.304s` | step 500 终止 |
| 最终权重 | `null` | **NOT PRODUCED** |

训练组件哈希确实发生变化，排除了“优化器没有更新新解冻模块”的解释：encoder `45bd98f8…e553e → 17f693b6…df4b5`，projector `40607e1d…85d9e → 165773a1…761ae`，predictor `6e0df89f…d88f → bca5a527…07769`，action_encoder `67919050…a213e → f56a5b69…3e761`。唯一冻结的 pred_proj 保持 `26545360…a5b6d`，`frozen_integrity.exact_match=true`。因此失败不是冻结边界或 source-count 基础设施失效，而是该单步联合优化配方在首 500 步已破坏专家 teacher prediction。

robust-v2b 的因果评估矩阵预注册为三色回归加六轴基准中 camera/floor/light/size 的 T1 与 T3；因止损未通过，以下 11 组全部 `NOT RUN`，没有逐 env flips。

| Condition | 轴 / 档位 | robust-v1 同条件参考 | robust-v2b | 状态 |
|---|---|---:|---:|---|
| red | regression | 46/50 (92%) | NOT RUN | v2b stopline 未授权 |
| blue_v2 | regression | 46/50 (92%) | NOT RUN | v2b stopline 未授权 |
| yellow_v2 | regression | 43/50 (86%) | NOT RUN | v2b stopline 未授权 |
| camera_t1 | +3.333 deg | 27/50 (54%) | NOT RUN | v2b stopline 未授权 |
| camera_t3 | +10 deg | 22/50 (44%) | NOT RUN | v2b stopline 未授权 |
| floor_t1 | alpha=1/3 | 19/50 (38%) | NOT RUN | v2b stopline 未授权 |
| floor_t3 | alpha=1 | 23/50 (46%) | NOT RUN | v2b stopline 未授权 |
| light_t1 | intensity=0.5 | 43/50 (86%) | NOT RUN | v2b stopline 未授权 |
| light_t3 | intensity=0.1 | 35/50 (70%) | NOT RUN | v2b stopline 未授权 |
| size_t1 | half-extent=16.667 mm | 36/50 (72%) | NOT RUN | v2b stopline 未授权 |
| size_t3 | half-extent=10 mm | 37/50 (74%) | NOT RUN | v2b stopline 未授权 |

参考成功向量来自 `outputs/eval/cube/ood_benchmark/robust_v1/` 的冻结 `results.json`；v2b 训练终态、止损曲线和冻结哈希分别在 `outputs/train/robust_v2b/robust_v2b_rerender_seed3072/completed.json`、`stopline_history.json` 与 `frozen_integrity.json`。`checkpoints/lewm-cube-robust_v2b/.../weights_stopped_step500.pt` 只保留为隔离负证据，不进入部署或上传清单。

## 3. Part 2：六轴四档 OOD 基准

### 3.1 成功率、衰减斜率与边界

| Axis | T0 / T1 / T2 / T3 | Endpoint slope | OLS slope | 部署边界 | 能力边界 |
|---|---:|---:|---:|---:|---:|
| Color | 92 / 86 / 92 / 86% | -2.000 pp/tier | -1.200 pp/tier | T0 red | T3 green |
| Camera | 90 / 54 / 42 / 44% | -4.600 pp/degree | -4.500 pp/degree | T0, 0 deg | T0, 0 deg |
| Light | 90 / 86 / 82 / 70% | -33.333 pp/intensity | -32.000 pp/intensity | T0, delta 0 | T3, delta 0.6 |
| Floor | 90 / 38 / 46 / 46% | -44.000 pp/alpha | -37.200 pp/alpha | T0, alpha 0 | T0, alpha 0 |
| Size | 90 / 72 / 74 / 74% | -1.600 pp/mm half-extent | -1.380 pp/mm | T0, 0 mm | T3, 10 mm |
| Action noise | 90 / 86 / 74 / 58% | -106.667 pp/sigma | -108.000 pp/sigma | T0, sigma 0 | T2, sigma 0.2 |

逐档条件：Color 为 red/yellow/blue/green；Camera 为 `0/3.333/6.667/10 deg`；Light 为相对默认强度距离 `0/0.2/0.4/0.6`；Floor 为默认到绿色 palette 的 `alpha=0/1/3/2/3/1`；Size 为 half-extent 改变量 `0/3.333/6.667/10 mm`；Action noise 为 `sigma=0/0.1/0.2/0.3`。

### 3.2 逐轴读法

- **颜色轴已鲁棒。** 四档保持 `86--92%`，全部跨过 70% 能力线；由于 yellow 已较 red 低 `6pp`，严格 `T0-3pp` 部署规则只到 T0。
- **相机轴最弱。** 首档 `+3.333 deg` 即从 `90%` 降到 `54%`，净少 18 局；更远档维持 `42--44%`。训练未安全地产生 robust-v2，因此相机重渲染增广的在线收益没有被测量。
- **地板轴同样没有安全缓冲。** `alpha=1/3` 即降到 `38%`；后两档回到 `46%` 仍远低于 70%。非单调小回升不改变边界结论。
- **光照轴渐进下降。** `90→86→82→70%`；T3 恰好落在能力阈值，部署阈值只通过 T0。
- **尺寸轴受影响但仍过能力线。** 三个 OOD 档为 `72/74/74%`；严格部署线不通过，但预注册范围内能力边界到 T3。
- **动作噪声轴呈单调退化。** `sigma=0.2` 仍为 `74%`，`sigma=0.3` 降至 `58%`；能力边界为 T2。

逐 env 双向翻转、全部 dataset rows、曲线 CSV/PNG 与计算值在 `outputs/eval/cube/ood_benchmark/paired_flips.json`、`benchmark_summary.json` 和六个 `*_curve.csv/png` 中。robust-v1 与 robust-v2 同档翻转为空，因为 robust-v2 未获在线运行授权。

## 4. E1：质量与摩擦物理轴

协议为 robust-v1 + T2、seed `42`、冻结同一 50 rows、goal offset `25`、budget `50`。质量轴同比缩放 `object_0` 的 body mass 与 inertia；摩擦轴同比缩放所有启用接触 geom 的三元摩擦系数。

| 颜色 | 轴 | x0.5 | x2 | x4 | 最低值 |
|---|---|---:|---:|---:|---:|
| red | mass | 92% | 92% | 92% | 92% |
| red | friction | 92% | 88% | 88% | 88% |
| blue_v2 | mass | 92% | 92% | 92% | 92% |
| blue_v2 | friction | 92% | 92% | 92% | 92% |

red friction x2/x4 相对 92% 默认值各有 `0 F→S / 2 S→F`，其余十个条件与对应默认成功向量完全相同。物理轴在线成功率在已测范围内较稳健，但固定 `12×300` 候选的 Measurement-2 并不支持“动力学已准确”的解释：

| Measurement-2 cell | E_roll 中位数 | >40 mm | final success |
|---|---:|---:|---:|
| red mass x4 | 236.831 mm | 99.806% | 0/3,600 |
| blue_v2 mass x4 | 236.711 mm | 99.806% | 0/3,600 |
| red friction x0.5 | 236.831 mm | 99.806% | 0/3,600 |
| blue_v2 friction x0.5 | 236.695 mm | 99.806% | 0/3,600 |

因此，当前任务成功对质量/摩擦变化的表观鲁棒性与候选五步 rollout 的大误差可以同时成立：前者是在线任务结果，后者是固定 off-policy 候选池上的动力学测量，不能互相替代。权威汇总 `outputs/eval/cube/physics_ood/summary.json` 的 SHA-256 为 `a0c9f8aee7e8440a36564455f38ee73599b0853920e4c3c05417440b7e71713c`。

## 5. E2：执行噪声边际化 CEM

E2 在 red / blue-v2、`sigma=0.2/0.3` 上做内部配对：vanilla 用单次 latent cost，marginalized 对每个候选执行 M=8 个独立物理动作噪声推演并取算术均值；实际执行动作噪声在 planner 返回后、`env.step` 前注入，配对臂共享 stateless seed。

| 颜色 | sigma | Vanilla | M=8 marginalized | 差值 | F→S / S→F |
|---|---:|---:|---:|---:|---:|
| red | 0.2 | 37/50 (74%) | 37/50 (74%) | +0pp | 3 / 3 |
| blue_v2 | 0.2 | 36/50 (72%) | 34/50 (68%) | -4pp | 3 / 5 |
| red | 0.3 | 31/50 (62%) | 31/50 (62%) | +0pp | 1 / 1 |
| blue_v2 | 0.3 | 31/50 (62%) | 28/50 (56%) | -6pp | 1 / 4 |

四个比较中 `0/4` 达到预期的 `+5pp`；M=8 的核心评估耗时为 vanilla 的 `5.41×`，每个 candidate cost 的底层 model cost 调用精确为 `8×`。因此该机制无净收益且性价比为负，不进入部署。E2 red 使用真实 H5，blue-v2 使用冻结 recolor；结论只对各自内部 vanilla/marginalized 配对成立，不能将 E2 的成功向量声明为旧六轴 default-rerender action-noise 曲线的复现。权威汇总：`outputs/eval/cube/noise_marginalized_cem/summary.json`，SHA-256 `458dfbd61647ed6131db773dc58e8b2716c399d19b0cbbd66788a7f42ecebdb5`。

## 6. E3：OOD 自检可测性

用户请求的两个 primary score 均为 **NOT MEASURED**：

| 请求分数 | 可连接 episode 分数 | 状态 | 原因 |
|---|---:|---:|---|
| current-frame control-latent kNN distance | 0 | NOT MEASURED | 现有 memory index 是 9D privileged physical state，不是 control latent；六轴正式 current pixels/latents 未持久化 |
| probe imagination error | 0 | NOT MEASURED | 现有 Measurement-2 是旧 checkpoint、三色、12-env 候选审计，无法与 robust-v1 六轴 50-env 按 condition+row 连接 |

因此不能从冻结产物声称模型能识别自己的 OOD 失败。报告中的 privileged-state kNN 与 first-cycle CEM cost AUC 只是已记录量的诊断 proxy；前者使用特权状态，后者是 goal-conditioned planning cost，均不是请求的自检分数。即使某些 proxy AUC 较高，也不得升级为模型 self-knowledge 结论。权威边界：`outputs/eval/cube/ood_detector/requested_score_availability.json` 与 `OOD_DETECTOR_REPORT.md`。

## 7. Part 3 / E4：cube-double 零样本迁移

E1--E3 完成且数据盘余量超过 `40 GiB` 后，E4 gate 为 `PASS`。OGBench cube-double train/val 发布索引约为 `284 MB / 28 MB`；Berkeley 官方端点在本机 TLS EOF 后，按冻结 fallback 使用 Hugging Face transport revision `0290b1be6721a8750c77334c316aca998ba4aa8b` 的同名文件。该文件符合预期 NPZ schema，评估前没有本地内容变换；由于未取得 Berkeley 端点文件或官方 checksum，与 Berkeley 端点的逐字节一致性未独立验证。

prepared manifest 从互异 source episodes 选出 seed-42 的 50 对，全部 raw goal offset 为 `+125`：`block_0` 位移最小/中位/最大为 `5.762/26.424/50.906 cm`，`block_1` 位移最大 `1.381 cm`，满足预注册的 `block_0 >=5 cm / block_1 <=2 cm`。manifest SHA-256 为 `d4033dc843b152bf8dd828943aeebdb743fb02a001f5d50e11bc956914b07b9f`。

正式协议为 robust-v1 + T2、`data_collection` double world、target=`block_0`、budget `50`，**零训练、零微调**；`block_1` 没有独立显式 target coordinate，也不参与物理 success，但其 full-state current/goal pixels 可能影响 latent cost。结果为 **4/50（8%）**，成功 env `[4,17,27,28]`，首次成功步 `[5,46,13,12]`，共 `97` planning cycles / `970` T2 iteration records，evaluator elapsed `34.889s`，视频 `50/50`。这表明存在非零组合迁移，但离稳定泛化相差显著。由于该任务是 play-derived、block-1 近静止、`+125` raw-step、block-0-only，且没有同源 single-cube paired control，不能将 `8%` 归因于某一个独立因素。

权威结果：`outputs/eval/cube/cube_double_transfer/formal/results.json`，SHA-256 `c7310757c679323507a520eed18b09cf88f1cfccf13fdf723e03035f8e2b7b73`。完整协议与解读见 `outputs/eval/cube/cube_double_transfer/CUBE_DOUBLE_TRANSFER_REPORT.md`。

## 8. 全部战线最终矩阵

### 8.1 正式三色矩阵

| 配置 | Red | Blue-v2 | Yellow-v2 | Macro | 类型 |
|---|---:|---:|---:|---:|---|
| Pretrained mean | 72% | 64% | 62% | 66.00% | 正式视觉起点 |
| Pretrained top-1 | 64% | 60% | 64% | 62.67% | selector 对照 |
| Memory Seed | 88% | 68% | 66% | 74.00% | 搜索支持 |
| Global hue | 80% | 64% | 78% | 74.00% | 像素增广 |
| MaskedAug mean | 74% | 76% | 74% | 74.67% | 像素增广 |
| Memory × global hue | 92% | 68% | 80% | 80.00% | 组合 |
| Memory × MaskedAug | 84% | 86% | 84% | 84.67% | 组合 |
| MaskedAug + T2 | 88% | 88% | 86% | 87.33% | 正式视觉 |
| **robust-v1 + T2** | **92%** | **92%** | **86%** | **90.00%** | **当前部署** |
| No-aug step 12,732 + T2 | 86% | 74% | 74% | 78.00% | 训练对照 |
| No-aug step 16,732 + T2 | 88% | 72% | 72% | 77.33% | 训练对照 |
| robust-v2 + T2 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | step-500 fail-stop |
| robust-v2b + T2 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | step-500 fail-stop |

机器可读版：`outputs/eval/cube/final_matrix.csv`；其中 `task_pct` 专用于 cube-double 等非三色任务，不能与 `macro_pct` 混算。三色历史数字逐项取自 `outputs/eval/cube/PLANNING_PROBLEM_ANALYSIS.md`；robust-v2 / v2b 状态分别取自各自的 `completed.json`。Privileged probe direct Red `94%` 单列为诊断上限，不可并入三色视觉 macro。

### 8.2 四条机制战线与本轮终局

| 战线 | 最强正结果 / 关键负结果 | 最终状态 |
|---|---|---|
| 视觉与搜索 | robust-v1 + T2 `92/92/86%`，macro `90.00%` | 部署基线保留 |
| Goal / selector | probe 直射 OOD `50/58/52%`；red offset25 `94%`；waypoint in-box `16%` vs direct `50%` | 坐标诊断有效，航点链归档 |
| 长程大脑 | offset100 baseline `72%`；B2 rule/LLM 均 `70%`；连续性 smoke 子目标 `0/6` | 归档 |
| Off-policy 动力学 | Play-v1 stopline `-16.34%`，但 Red 候选 `85.29→85.24 mm`，近乎零改善 | 四轮归档 |
| 重渲染 robust-v2 | 专家 stopline `+49.319%` @ step 500 | fail-stop，在线 NOT RUN |
| 重渲染 robust-v2b | 联合训练四个核心组件，专家 stopline `+4,638.978%` @ step 500 | fail-stop，整线正式关闭 |
| 物理轴 | mass 全部 `92%`；red friction x2/x4 `88%`、blue-v2 全部 `92%`；四个 Measurement-2 cell 的 E_roll 中位数约 `236.7--236.8 mm` | 在线任务稳健，不代表 off-policy dynamics 准确 |
| 噪声边际化 | M=8：red `74→74/62→62%`，blue-v2 `72→68/62→56%`；核心成本 `5.41×` | 归档 |
| OOD 自检 | 请求的 latent-kNN / probe-error 均无可连接 episode 分数 | NOT MEASURED；proxy 仅诊断 |
| 组合迁移 | robust-v1 + T2 cube-double 零样本 `4/50（8%）` | 正式测得，能力弱 |

来源：`waypoint_probe/WAYPOINT_REPORT.md`、`longhorizon/BRAIN_LINE_VERDICT.md`、`PLAY_LINE_VERDICT.md` 与本轮训练/门禁产物。

## 9. 能力边界声明

1. **已证实能力：** robust-v1 + T2 在冻结三色 50-env 协议上达到 macro `90.00%`；新颜色四档为 `86--92%`。
2. **视觉弱轴：** Camera `+3.333 deg` 与 floor `alpha=1/3` 已跌破 70%，没有可声明的非零安全边界；光照在强度差 `0.6` 时为 70%。
3. **物理与控制：** cube half-extent 变化到 `10 mm` 仍为 74%；质量 `x0.5/x2/x4` 在 red/blue-v2 均为 92%，摩擦最低为 red 的 88%。旧六轴内部 action-noise 在 `sigma=0.2` 为 74%；E2 新内部配对进一步证明 M=8 边际化不能改善噪声鲁棒性。这些都只是所测环境族与离散档位的经验能力线。
4. **训练边界：** 状态重渲染数据本身通过身份与几何 QC，但 dynamics-only robust-v2 在 500 steps 回归 `+49.319%`；进一步放开 encoder/projector 的 robust-v2b 回归扩大到 `+4,638.978%`。数据有效不等于单步联合优化安全，且该重渲染微调线不再增加第三种配置。
5. **自检边界：** 请求的 control-latent kNN 与 probe imagination error 均为 `NOT MEASURED`；现有 proxy AUC 只能诊断相关性，不能证明模型能在决策前识别失败。
6. **组合边界：** cube-double 的正式零样本结果为 `8%`。单方块模型有非零迁移，但无法稳定处理第二个物体；该结果只针对 block-0 单目标 success，不覆盖双目标控制。
7. **未测能力：** robust-v2 / v2b 在线表现仍为 `NOT RUN / NOT MEASURED`，不能用 robust-v1 结果替代。
8. **规划边界：** 五步 CEM 候选的 off-policy 想象误差仍是全线共同瓶颈；E1 的四个物理 Measurement-2 cell 均约 `236.7--236.8 mm` 且 `99.806% >40 mm`，换 selector、goal cost、航点链、play 单步微调、rerender 单步微调与 M=8 边际化均未解决。

## 10. 未来工作

优先级按因果证据排序：

1. **多步训练目标。** 直接对齐规划所需的 action-conditioned 多步推演，先在冻结审计池证明候选五步误差下降，同时维持专家 teacher loss；这是首要项。
2. **解耦视觉适配与动力学保持。** robust-v2 与 v2b 已表明 1:1 重渲染的单步微调无法同时保持专家预测；若未来重开，应作为全新研究问题比较冻结 dynamics 的视觉 adapter、低秩视觉侧适配或 feature consistency，而不是本线继续追加第三种配置。
3. **相机与地板定向基准。** 以 camera tier1 `54%`、floor tier1 `38%` 为最小离线/在线回归集，报告 paired flips，避免三色回归掩盖弱轴。
4. **真实 OOD 自检仪表。** 持久化 decision-time robust-v1 control latent 与 executed-plan terminal XYZ/physical XYZ，冻结独立 calibration split 后再测 latent-kNN 与 probe-error；不再用 privileged-state 或 CEM cost proxy 替代。
5. **事前防呆式慢环。** 慢环只在可验证的短程可达性与物理连续性成立时介入；不再用 LLM、静态 waypoint 或 M=8 cost 平均替代错误的多步动力学。
6. **cube-double 组合泛化。** 保留本 50 对为只读 zero-shot 基准；未来训练必须另建 train/calibration split，维持 block-0 target 与 block-1 非目标语义，并报告是否因第二物体的视觉干扰、接触动力学或 single-cube memory support 而失败。

## 11. 产物索引

- 重渲染数据报告：`datasets/cube_rerender_v2/RERENDER_DATA_REPORT.md`
- 数据 manifest / validation：`datasets/cube_rerender_v2/manifest.json`、`validation.json`
- robust-v2 训练计划与终态：`outputs/train/robust_v2/robust_v2_rerender_seed3072/run_plan.json`、`completed.json`
- stopline：同目录 `stopline_event.json`、`stopline_history.json`、`frozen_integrity.json`
- robust-v2b 负结果：`outputs/train/robust_v2b/robust_v2b_rerender_seed3072/run_plan.json`、`completed.json`、`stopline_event.json`、`stopline_history.json`、`frozen_integrity.json`
- 六轴总表：`outputs/eval/cube/ood_benchmark/benchmark_summary.json`
- 六轴报告与曲线：`outputs/eval/cube/ood_benchmark/OOD_BENCHMARK_REPORT.md`、六组 `*_curve.csv/png`
- 双向逐 env 翻转：`outputs/eval/cube/ood_benchmark/paired_flips.json`
- 物理轴报告与汇总：`outputs/eval/cube/physics_ood/PHYSICS_OOD_REPORT.md`、`summary.json`、`success_matrix.csv`
- 执行噪声边际化：`outputs/eval/cube/noise_marginalized_cem/REPORT.md`、`summary.json`、`matrix.csv`、`paired_flips.json`
- OOD 自检可测性：`outputs/eval/cube/ood_detector/OOD_DETECTOR_REPORT.md`、`requested_score_availability.json`、`diagnostic_proxy_axis_auc.csv`
- cube-double gate / manifest / 正式结果：`outputs/eval/cube/cube_double_transfer/part3_gate.json`、`prepared/pair_manifest.json`、`formal/results.json`
- cube-double 专项报告：`outputs/eval/cube/cube_double_transfer/CUBE_DOUBLE_TRANSFER_REPORT.md`
- 双仓建议清单：`outputs/eval/cube/OOD_FINAL_UPLOAD_LIST.md`
