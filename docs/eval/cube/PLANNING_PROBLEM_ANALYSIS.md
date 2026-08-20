# Cube 规划问题分析（canonical v2）

状态：定版分析文档  
更新日期：2026-08-21  
范围：Cube 单任务中，从 goal 协议、候选生成、选择器、训练增广、规划器约束到 off-policy 动力学的完整证据链。  
主部署配置：`robust-v1 + T2`，Red / Blue-v2 / Yellow-v2 为 `92% / 92% / 86%`，三色宏平均 `90.00%`。

## 0. 执行摘要

本项目最终确认了四类相互独立的问题：goal 图是否表达正确目标、CEM 是否生成有用候选、cost/selector 是否按正确标准排序、predictor 是否能准确推演候选动作。四者不能用同一个成功率变化相互替代。

主要结论如下：

1. Goal 必须使用真实 HDF5 帧；目标身份只允许受控 recolor；动态子目标还必须满足 EE 物理连续性。单独满足 EE 距离 `<=10cm` 仍不足以保证短程可达。
2. 将 CEM 最终 elite mean 改成 top-1 没有改善结果；Memory Seed、MaskedAug 与 T2 的有效增益来自候选支持和视觉稳健性的组合，而不是 selector 的简单替换。
3. 视觉三色正式结果从原始 `66.00%` 提升至 `90.00%`。其中 robust-v1 的颜色回归良好，但地板、光照、相机和跨 episode 目标仍明显敏感。
4. Probe 坐标 cost 证明 goal 图连续性会造成显著损失，但 privileged XYZ 直射和 waypoint 均未解决 predictor 的多步导航误差；Red probe direct `94%` 是诊断结果，不是三色视觉部署成绩。
5. 四轮 off-policy 微调最终给出明确边界：保留专家流形与改善 CEM 五步候选误差没有同时实现。Play-v1 保住专家流形，却没有改善候选分布。

**JEPA 单步训练目标与规划所需多步推演能力存在结构性错配**

这句话是当前证据支持的核心机制判断：expert/near-policy 动作上的五步误差约 `5--6mm`，而真实 CEM off-policy 五步候选的三色中位误差仍约 `85--125mm`。该差异不是由单一颜色、goal cost 或训练步数能够解释。

## 1. 范围、前置证据与因果边界

### 1.1 前置证据

| 项目 | 已验证结果 | 证据路径 | 读法 |
|---|---:|---|---|
| 原始 Cube，CEM10 Red | `72%` | `outputs/eval/cube/pretrained/evidence/ogb_cube_results.txt` | 原始权重与原始红色任务的起点结果。 |
| PushT pretrained | `92%` | `outputs/eval/pusht_pretrained/evidence/pusht_results.txt` | 证明相同整体范式并非在所有任务上都低成功；不用于 Cube 三色矩阵。 |
| Cube 正式三色起点 | `72/64/62%`，macro `66.00%` | `outputs/eval/cube/ood/COLOR_OOD_REPORT.md` | 采用真实 H5 future frame 与受控 recolor 的可比较基线。 |

PushT 数据约 `56G` 是用户提供的运维记录，当前仓内没有删除前体积或删除命令的可审计产物，因此本文不把该数字作为仓内核验事实。现存证据只确认 PushT 评估已经运行。

### 1.2 术语与统计口径

| 术语 | 本文定义 |
|---|---|
| Goal construction | 将目标状态表达为规划器可消费的图像或坐标；它不等于候选生成。 |
| Candidate generation | CEM 在五步联合动作空间中采样、更新和保留候选的过程。 |
| Selector / cost | 对已生成候选排序并输出动作的规则；改变它不等于改善 predictor。 |
| Dynamics | predictor 与 action encoder 对动作条件未来隐状态的推演能力。 |
| `E_roll` | 五步候选末端的物理 rollout 误差，单位 mm；比较想象末端与缓存物理执行末端。 |
| expert/near-policy manifold | 专家或接近专家动作分布；固定专家动作 Measurement-1 用于测量该区域。 |
| planner off-policy sequence | CEM 产生的五步联合动作序列，可能明显偏离专家分布。 |
| mean | 最后一轮 CEM 更新得到的 elite 动作均值。 |
| top-1 | 最后一轮采样候选中 cost 最低的单个样本。 |
| candidate0 | 第九轮更新后插入第十轮候选池的 updated mean；它不是最终 elite mean。 |
| pp | 百分点差，例如 72% 到 74% 是 `+2pp`，不是相对增长 2%。 |
| Macro | Red、Blue-v2、Yellow-v2 三个 50-env 成功率的不加权平均。 |
| 分辨率 | 每个颜色 50 env，因此单个 env 翻转对应 `2pp`。 |
| PASS / FAIL | 已运行且按预注册判据通过或失败。 |
| NOT MEASURED | 没有产生足够科学产物，不能判成失败或通过。 |
| NOT RUN | 因 fail-stop 条件未满足而未执行；不能写成 0%。 |
| Probe result | 使用 privileged 目标 XYZ 的诊断结果，不是纯视觉部署结果。 |
| `+5cm` / fallback `5.57cm` | 前者是真实盒外约 5cm 档；nominal `+10/+20cm` 因数据不足均回退到中位 `5.57cm`、最大 `7.02cm` 的同一 population。 |

### 1.3 瓶颈地图（问题分层表）

| 层 | 主要问题 | 关键实验 | 已解决程度 | 当前结论 |
|---|---|---|---|---|
| Goal 图 | 合成构图、颜色身份和场景连续性 | A、长程监督器诊断 | 静态目标协议已解决；动态子目标未解决 | 真实帧与受控 recolor 必须同时满足。 |
| Selector / cost | mean、top-1、candidate0 含义混淆，文本选择是否能补足 latent cost | B、H、I | 已排除所测替代方案 | 读数规则、盲化 LLM、probe 与 waypoint 的四段证伪链均未形成稳定部署增益。 |
| 候选支持 | 自由 CEM 候选包含大量失真序列 | C、E | 部分缓解 | Memory Seed 与 T2 提升在线成功，但保留较大误差尾。 |
| 视觉稳健性 | 颜色、背景、光照、相机变化 | D、F、G | 颜色显著改善，其他轴仍敏感 | 针对性增广有效，纯训练时长不能替代。 |
| Goal cost | 跨 episode 图像目标的编码连续性损失 | H | 诊断时可不使用图像 goal，不能部署替代 | Probe cost 显著提升 OOD，但 in-box 仅 50%。 |
| 长程导航 | 五步 horizon 对远目标不足 | I | 未解决 | waypoint 分解反而降低成功。 |
| Off-policy dynamics | 五步候选想象误差远大于专家动作 | J、四轮时间线 | 未解决，实验已归档 | 当前数据混合与安全微调不能适配该分布差异。 |

## A. Goal 构图协议

### 方法

在同一批 Cube 评估中对比三类 goal：当前机械臂合成构图、保持红色 goal 的颜色不匹配协议、真实 HDF5 future frame 加受控 recolor。CEM10 与 RS1 都被保留，以区分 goal 表达问题和搜索预算问题。

### 结果

| Planner / goal | Red | Blue | Yellow | Macro |
|---|---:|---:|---:|---:|
| CEM10，synthetic matched | 72% | 46% | 44% | 54.00% |
| CEM10，raw-red-goal mismatched | 72% | 74% | 70% | 72.00% |
| CEM10，真实 H5 recolor，matched-v2 | 72% | 64% | 62% | 66.00% |
| RS1，synthetic matched | 42% | 42% | 44% | 42.67% |
| RS1，raw-red-goal mismatched | 42% | 38% | 38% | 39.33% |
| RS1，matched-v2 | 42% | 38% | 38% | 39.33% |

### 逐列读法

- `Planner / goal` 同时标识搜索预算与 goal 来源；不能只按宏平均选择协议。
- Red、Blue、Yellow 各是 50 env。Blue/Yellow 只有 matched-v2 才表达正确颜色身份。
- `raw-red-goal mismatched` 的 `72/74/70%` 是诊断值，不是可部署颜色协议，因为蓝黄环境仍接收红色 goal。
- `synthetic matched` 的低值说明像素级合成破坏了机械臂、遮挡和场景结构。

### 结论

正式起点采用真实 H5 recolor 的 `72/64/62%`，macro `66.00%`。Goal 最终规则为：

1. 使用真实 HDF5 帧，禁止合成场景构图。
2. 只做受控 recolor，保持几何、遮挡与场景结构，并绑定源帧身份。
3. 动态子目标必须满足 EE 物理连续性；`<=10cm` 只是必要筛选，未来还需验证抓持关系、姿态与短程连通性。

证据：`outputs/eval/cube/ood/COLOR_OOD_REPORT.md`、`outputs/eval/cube/longhorizon/BRAIN_LINE_VERDICT.md`。

## B. 选择层对照：读数规则与 blind LLM reranker

### 方法

第一项对照固定相同 CEM 候选与评估清单，只比较最后输出最后一轮 elite mean 和最后一轮 top-1 候选。第二项对照在固定 12 个首周期、三种颜色共 36 个盲化单元上，将候选只以文本统计特征提供给 GPT-5.5 与 DeepSeek；同时保留 cost top-1、Top-30 均匀随机期望和 hindsight oracle。两项对照都不改变候选生成与 predictor。

### 结果

| Selector | Red | Blue-v2 | Yellow-v2 | Macro | 总成功 |
|---|---:|---:|---:|---:|---:|
| Elite mean | 72% | 64% | 62% | 66.00% | 99/150 |
| Top-1 | 64% | 60% | 64% | 62.67% | 94/150 |
| Top-1 相对 mean | -8pp | -4pp | +2pp | -3.33pp | -5 |

| 36 个 blind reranker 单元 | Ever success | Final success |
|---|---:|---:|
| GPT-5.5 | 17/36 | 15/36 |
| DeepSeek | 18/36 | 13/36 |
| Cost top-1 | 16/36 | 13/36 |
| Top-30 均匀随机期望 | 14.733/36 | 13.333/36 |
| Hindsight oracle | 22/36 | 22/36 |

### 逐列读法

- `Selector` 只说明最终执行选择，不改变候选池。
- 三个颜色的 top-1 变化方向不一致，且总成功下降。
- `candidate0` 是第九轮 updated mean 被插入第十轮池，不等于表中的最终 elite mean。
- Blind reranker 的 36 个单元来自同一 12 个 env 跨三种颜色重复，不能当作 36 个独立总体样本，也不能等价成 50-env 闭环成功率。
- GPT-5.5 的 ever/final 为 `17/15`，DeepSeek 为 `18/13`；它们相对 cost top-1 的 `16/13` 没有一致的 final 优势。DeepSeek 的三个严格协议失败均按失败计入，没有重问。
- Oracle 的 `22/22` 说明候选集合仍有未被所测 selector 稳定识别的可行项，但这不证明文本 LLM 能识别这些项。

### 结论

简单改用 top-1 不能解释或修复规划失败，blind LLM reranker 也不支持稳定选择优势，正式协议保留 elite mean。选择层的完整证伪链为：`读数规则 → 盲化 LLM → probe → waypoint`。其中 probe 和 waypoint 分别在 H、I 节进一步测试 cost 表达与分段执行；四项都没有形成稳定、可部署的选择层替代方案。后续增益必须归因到候选支持、模型或视觉协议，而不是把 candidate0、top-1 与最终 mean 混为一项。

证据：`outputs/eval/cube/ood_select/CEM_SELECTOR_REPORT.md`、`outputs/rerank_pilot/RERANK_PILOT_REPORT.md`。

## C. Memory Seed

### 方法

从约 `1.76m` 个专家锚点中检索当前状态近邻；排除当前评估 episode，并从 10 个不同源 episode 各取一个动作序列加入候选支持。该方法保留原 planner，只修改初始候选来源。

### 结果

| 配置 | Red | Blue-v2 | Yellow-v2 | Macro |
|---|---:|---:|---:|---:|
| Mean baseline | 72% | 64% | 62% | 66.00% |
| Memory Seed | 88% | 68% | 66% | 74.00% |
| 变化 | +16pp | +4pp | +4pp | +8.00pp |

| 固定 12-env 审计项 | Red | Blue-v2 | Yellow-v2 |
|---|---:|---:|---:|
| 注入后可行候选 ever / final | 12/12 / 11/12 | 12/12 / 11/12 | 11/12 / 11/12 |
| Memory slots ever / final | 10/12 / 9/12 | 10/12 / 9/12 | 10/12 / 9/12 |
| CEM mean ever / final | 9/12 / 9/12 | 8/12 / 8/12 | 7/12 / 7/12 |

### 逐列读法

- 第一表的三色列是正式 50-env 在线结果。
- 第二表的 `ever / final` 区分整个搜索过程曾出现可行解与最终是否仍保留可行解。
- Memory slot 的高命中不意味着最终 elite mean 一定可行；Yellow 的 CEM mean 最弱，显示候选传播仍会损失支持。

### 结论

Memory Seed 把宏平均从 `66.00%` 提高到 `74.00%`，证明专家近邻候选能直接改善搜索支持。但它没有完全解决 Blue/Yellow，且最终 mean 仍会离开可行候选区域。

证据：`outputs/eval/cube/memory_seed/MEMORY_SEED_REPORT.md`。

## D. 增广与组合

### 方法

分别比较整帧 global hue、只对方块身份区域换色的 MaskedAug，以及它们与 Memory Seed 的组合。所有组合在相同三色 50-env 协议下比较。

### 结果

| 配置 | Red | Blue-v2 | Yellow-v2 | Macro |
|---|---:|---:|---:|---:|
| Pretrained mean | 72% | 64% | 62% | 66.00% |
| Memory Seed | 88% | 68% | 66% | 74.00% |
| Global hue | 80% | 64% | 78% | 74.00% |
| MaskedAug mean | 74% | 76% | 74% | 74.67% |
| Memory × global hue | 92% | 68% | 80% | 80.00% |
| Memory × MaskedAug | 84% | 86% | 84% | 84.67% |

| 交互项 | Red | Blue-v2 | Yellow-v2 | Macro |
|---|---:|---:|---:|---:|
| Memory × global，相对加性预期 | -4pp | 0pp | -2pp | -2.00pp |
| Memory × Masked，相对加性预期 | -6pp | +6pp | +6pp | +2.00pp |

### 逐列读法

- MaskedAug 相对 pretrained 为 `+2/+12/+12pp`，主要改善颜色 OOD，而不是只提高 Red。
- Memory × Masked 相对 MaskedAug 为 `+10/+10/+10pp`，显示搜索支持与视觉身份增广可组合。
- 交互项不是配置的绝对增益，而是相对两个单因素增益简单相加后的偏差。

### 结论

针对性像素增广的归因成立。Global hue 可以提高部分颜色，但结构性副作用更大；MaskedAug 对 Blue/Yellow 更稳定。Memory × MaskedAug 的 `84.67%` 成为进入 T2 前的冻结基线。

证据：`outputs/eval/cube/MASKEDAUG_COMBO_REPORT.md`。

## E. T2 规划器

### 方法

T1 将单个最近专家序列作为 mean，并把初始 std 收窄到 `0.2`。T2 保留 legacy zero/unit CEM，同时在 300 个候选中固定放入 `10` 条 exact memory seed、`20` 条扰动 seed 和 `270` 条自由候选。两者都使用五步 horizon、10 轮 CEM。

### 结果

| 在线配置 | Red | Blue-v2 | Yellow-v2 | Macro |
|---|---:|---:|---:|---:|
| Seed × MaskedAug baseline | 84% | 86% | 84% | 84.67% |
| T1 | 72% | 70% | 70% | 70.67% |
| T2 | 88% | 88% | 86% | 87.33% |
| T2 相对 baseline | +4pp | +2pp | +2pp | +2.67pp |

| 离线协议 | Red median / `>40mm` | Blue median / `>40mm` | Yellow median / `>40mm` |
|---|---:|---:|---:|
| Old unseeded | 85.720mm / N/A | 112.476mm / N/A | 123.356mm / N/A |
| T1 | 10.763mm / 3.56% | 11.910mm / 2.17% | 10.692mm / 3.61% |
| T2 | 21.115mm / 32.78% | 22.672mm / 33.94% | 23.143mm / 37.94% |

### 逐列读法

- 在线表比较任务成功率；离线表比较同一物理缓存上的五步末端误差，二者不能互换。
- T1 的误差最低但在线最差，排除了“误差越低，成功率必然越高”的单变量解释。
- T2 保留约 `33--38%` 的大误差尾，却实现在线正增益，说明目标相关性与候选覆盖同样重要。

### 结论

T2 的 `88/88/86%`、macro `87.33%` 是 MaskedAug 阶段的定版规划协议。它验证了“保留自由搜索并注入专家邻域支持”优于把全部搜索限制在单一近邻，但没有消除 off-policy 误差尾。

证据：`outputs/eval/cube/trust_region/TRUST_REGION_REPORT.md`。

## F. robust-v1 全视觉轴增广

### 方法

从 MaskedAug 权重继续训练，组合方块 hue shift、背景/地板低饱和区域 hue shift 与 gamma `0.7--1.4` 的亮度扰动。训练 4,000 steps，保留 T2，随后按颜色回归、地板、光照、相机和目标位置分轴评估。

### 结果

| 评估轴 / 条件 | 成功率 |
|---|---:|
| Red / Blue-v2 / Yellow-v2 | 92% / 92% / 86% |
| 三色 Macro | 90.00% |
| Floor red / green | 48% / 46% |
| Light low / high | 60% / 68% |
| Camera minus / plus | 44% / 42% |
| Goal in-box | 26% |
| Goal true `+5cm` | 22% |
| Goal nominal `+10/+20cm`，实际 fallback | 12% / 12% |

### 逐列读法

- 三色回归是正式部署矩阵；其余是单轴 OOD 测试，不能与三色 macro 混算。
- nominal `+10/+20cm` 没有足够真实帧，实际都落在中位 `5.57cm`、最大 `7.02cm` 的 fallback population，不能解释为真实 10cm 或 20cm 曲线。
- 地板、光照和相机成功率明显低于颜色回归，说明训练覆盖没有自动迁移到所有视觉轴。

### 结论

robust-v1 将正式三色结果提高到 `92/92/86%`、macro `90.00%`，是最终视觉部署配置。它对颜色稳健，但对地板、光照、相机和跨 episode goal 仍敏感。训练 4,000 steps，耗时 `10,113s = 2h48m33s`；专家止损相对基座为 `-7.42%`，通过。

证据：`outputs/eval/cube/OOD_ROBUSTNESS_REPORT.md`。

## G. No-augmentation control

### 方法

从原始权重出发，用原始专家数据、零增广训练。step `12,732` 对齐 MaskedAug 训练算力；再以 fresh optimizer 训练 4,000 steps 至累计 `16,732`，对齐 robust-v1 的累计步数。该设计专门检验“增广收益只是原模型欠拟合”的替代解释。

### 结果

| Checkpoint | Red | Blue-v2 | Yellow-v2 | Macro |
|---|---:|---:|---:|---:|
| No-aug step 12,732 | 86% | 74% | 74% | 78.00% |
| No-aug step 16,732 | 88% | 72% | 72% | 77.33% |
| Robust-v1 + T2 | 92% | 92% | 86% | 90.00% |

### 逐列读法

- Red 随无增广训练变长从 86% 到 88%，但 Blue/Yellow 从 74/74% 变为 72/72%。
- 两阶段使用 fresh optimizer，因此 `16,732` 是累计训练预算，不是一条连续 optimizer trajectory。
- 与 robust-v1 的差距集中在 Blue/Yellow，正是增广预期改善的 OOD 轴。

### 结论

对照排除了纯欠拟合解释：增加训练步数主要提高 Red，Blue/Yellow 没有追平增广臂。Control phase A 耗时 `5,449.934492s`，phase B 耗时 `1,823.654701s`，合计 `7,273.589193s = 2h01m13.589s`。

证据：`outputs/eval/cube/CONTROL_AND_PROBEGOAL_REPORT.md`、`outputs/train/control_noaugment/`。

## H. Probe 坐标目标 cost

### 方法

为 robust-v1 训练 episode 隔离的 XYZ probe，测试中位误差 `3.3206mm`，通过 `<15mm` 质量门。规划 cost 不再编码 goal 图，而是比较 probe 解码的想象末端 XYZ 与 privileged 目标方块坐标。

### 结果

| Tier | Masked latent | Robust latent | Robust probe | Probe 相对 Robust |
|---|---:|---:|---:|---:|
| In-box | 26% | 32% | 50% | +18pp |
| True `+5cm` | 22% | 18% | 58% | +40pp |
| Fallback `5.57cm` | 12% | 12% | 52% | +40pp |

### 逐列读法

- `Masked latent` 与 `Robust latent` 都使用图像 goal latent cost，只是权重不同。
- `Robust probe` 不使用 goal 图，而使用特权 XYZ，因此只能诊断 cost 表达损失。
- In-box probe 只有 50%，未达到预注册 `>=70%`，说明 goal 图不是唯一瓶颈。

### 结论

Probe cost 显著恢复跨 episode 目标成功率，证明 goal 图连续性带来 `18--40pp` 的损失；但它没有解决剩余约一半失败，且不是纯视觉可部署接口。Red standard probe direct `94%` 属于后续单色 privileged-coordinate 诊断记录，不进入正式三色 macro。

证据：`outputs/eval/cube/CONTROL_AND_PROBEGOAL_REPORT.md`。

## I. 几何 waypoint chain

### 方法

把当前方块到目标 XYZ 的直线按主间距 `4cm` 分段，用 probe cost 逐段追踪；到达阈值 `2cm`，每段最多 25 步，失败后回退为直射最终目标。In-box 另比较 `2.5cm` 与 `6cm`。

### 结果

| 场景 | Probe direct | Waypoint d=4cm | 其他间距 |
|---|---:|---:|---:|
| OOD in-box | 50% | 16% | d=2.5cm: 14%；d=6cm: 22% |
| OOD true `+5cm` | 58% | 12% | N/A |
| OOD fallback `5.57cm` | 52% | 18% | N/A |
| offset100 | 72% | 68% | N/A |
| Red offset25 | 94% | 70% | N/A |

### 逐列读法

- 每一行的 direct 与 waypoint 使用配对评估清单；所有主比较均为负增益。
- 首段到达率约 `30--60%`，但不能传播成整条链的更高最终成功率。
- 五步想象 XYZ 误差约 `49.56--69.07mm`，已经大于 `4cm` waypoint 间距和 `2cm` 到达阈值，段切换依据不可靠。

### 结论

几何分段没有改善跨 episode 或长程任务。失败不是简单的“目标太远”，而是每段仍依赖同一五步 predictor；误差尺度超过分段尺度后，增加切换反而损失控制一致性。

证据：`outputs/eval/cube/waypoint_probe/WAYPOINT_REPORT.md`。

## J. Play-v1 动力学微调

### 方法

混合 `62.5% expert + 37.5% official play`，从 robust-v1 warm start，只训练 predictor 与 action encoder；encoder、projector 和 pred_proj 冻结。只使用单步 teacher forcing pred loss 与冻结 target SIGReg，不调用 rollout loss。每 500 step 在固定 4,352 个专家 clips 上检查止损，训练共 5,000 steps。

### 结果

| 门 | 要求 | Play-v1 | 状态 |
|---|---:|---:|---|
| 专家 stopline | 相对 step 0 `<=+10%` | `-16.34%` | PASS |
| Red 候选 | median `<40mm` 且尾率减半 | 85.24mm / 63.00% | FAIL |
| Blue-v2 候选 | 同上 | 118.04mm / 77.61% | FAIL |
| Yellow-v2 候选 | 同上 | 124.55mm / 77.31% | FAIL |
| Measurement-1 expert depth-5 | median `<=8mm` | 5.16mm | PASS |
| 在线 T2 / probe | 全部门通过才运行 | NOT RUN | N/A |

| 候选对照 | Red | Blue-v2 | Yellow-v2 |
|---|---:|---:|---:|
| Robust reference median | 85.29mm | 112.08mm | 121.82mm |
| Play-v1 median | 85.24mm | 118.04mm | 124.55mm |
| Play 相对 Robust | -0.05mm | +5.96mm | +2.72mm |

### 逐列读法

- Stopline 与 Measurement-1 证明专家流形被保留，不能把候选失败归因于灾难性遗忘。
- 三色候选中位都远高于 `40mm`；Blue/Yellow 相对 robust 还变差。
- 在线列为 `NOT RUN`，因为 fail-stop 正确阻止了无资格 checkpoint 的在线评估；不能填 0%。

### 结论

Play-v1 在安全单步训练下学到了 expert/near-policy 动力学，却没有迁移到 CEM 五步联合候选。训练 5,000 steps、耗时 `6,534.750052s = 1h48m54.750s`，结果构成 off-policy 动力学实验的最终科学失败。

证据：`outputs/eval/cube/PLAY_LINE_VERDICT.md`、`outputs/eval/cube/play_v1/offline/`。

## 2. 长程慢环监督器：独立负证据

该组实验不进入 A--J 三色主矩阵，但用于验证“在执行层外增加规则或 LLM 决策是否能补偿规划失败”。

| 阶段 | Baseline | Rule | LLM | 子目标到达 | 结论 |
|---|---:|---:|---:|---:|---|
| B1 offset100 | 72% | N/A | 72% | N/A | 70/70 决策为 CONTINUE；14/14 失败局均触发。 |
| B2 offset100 | 72% | 70% | 70% | Rule 0/36；LLM 0/38 | Rule 与 LLM 成功向量逐位相同。 |
| EE continuity smoke | 子集 baseline 0/2 | 0/2 | NOT RUN | STALLED 0/6；全部 0/7 | 前置 smoke 失败，formal 未运行。 |

大脑死因链（失败因果序列）：

1. B1 的触发和 API 调用流程正常，但 70/70 决策均返回 CONTINUE。
2. B2 修复 prompt 后实际生成干预，规则与 LLM 都得到 70%，但没有一次子目标到达。
3. 离线诊断显示原规则 36/36 都是 planner cost 下降而方块不移动；goal 帧 EE 跳变中位 `18.1cm`，`33/36` 超过 `10cm`。
4. 加入 EE `<=10cm` 连续性筛选后，smoke 仍为 STALLED `0/6`、全部干预 `0/7` 到达，最终 `0/2` 成功。
5. 因此单点 EE 距离是必要约束但不是局部可达性的充分条件；规则 formal 与条件 LLM 臂按协议均未运行，监督器实验停止并归档。

证据：`outputs/eval/cube/longhorizon/BRAIN_B1_REPORT.md`、`outputs/eval/cube/longhorizon/BRAIN_B2_REPORT.md`、`outputs/eval/cube/longhorizon/SUBGOAL_DIAGNOSIS.md`、`outputs/eval/cube/longhorizon/BRAIN_LINE_VERDICT.md`。

## 3. Off-policy 动力学四轮时间线

| 轮次 | 数据与损失 | 专家流形 | 五步候选结果 | 科学状态 |
|---|---|---:|---|---|
| V1 | 合成 Gaussian/T2/AR1，单步损失 | pred loss `+71.1%` | Red `85.72→43.15mm`；Blue `112.48→94.88mm`；Yellow `123.36→78.30mm` | 候选改善但专家遗忘，门失败，在线未运行。 |
| V2 main | 真实规划分布，expert 与 V2 共用 rollout loss | `+282.0%` | 止损后未测 | 科学失败。 |
| V2 retry | 提高 expert 比例，仍共用 rollout loss | `+1213.0%` | 止损后未测 | 更差，停止模型重试。 |
| V3 | rollout loss 仅用于 V2 | 两次均未到 step 500 | 无有效 checkpoint | 基础设施中断，`NOT MEASURED`，不能记作科学失败。 |
| Play-v1 | Official play，单步 TF，只训动力学栈 | stopline `-16.34%`；M1 `5.16mm` | `85.24/118.04/124.55mm` | 保住专家但候选无改善，科学失败并归档。 |

四轮共同结论不是“所有 off-policy 训练都不可能”，而是当前已验证方案无法同时满足专家保持和 CEM 五步候选改善。V3 没有科学 checkpoint，必须保留为基础设施未测；Play-v1 的完整 5,000-step 运行排除了该不确定性。

证据：`outputs/eval/cube/OFFPOLICY_FINAL_VERDICT.md`、`outputs/eval/cube/offpolicy_v1/OFFPOLICY_V1_REPORT.md`、`outputs/eval/cube/PLAY_LINE_VERDICT.md`。

## 4. 正式三色总矩阵

下表只保留 Red / Blue-v2 / Yellow-v2 各 50 env、协议可比较的正式结果。Synthetic、颜色 mismatched、长程、目标 tier、privileged probe 和未运行 checkpoint 均不进入 Macro。

| 配置 | Red | Blue-v2 | Yellow-v2 | Macro | 结果类型 |
|---|---:|---:|---:|---:|---|
| Pretrained mean，真实帧 recolor | 72% | 64% | 62% | 66.00% | 正式视觉 |
| Pretrained top-1 | 64% | 60% | 64% | 62.67% | 正式 selector 对照 |
| Memory Seed | 88% | 68% | 66% | 74.00% | 正式视觉 |
| Global hue | 80% | 64% | 78% | 74.00% | 正式视觉 |
| MaskedAug mean | 74% | 76% | 74% | 74.67% | 正式视觉 |
| Memory × global hue | 92% | 68% | 80% | 80.00% | 正式视觉 |
| Memory × MaskedAug | 84% | 86% | 84% | 84.67% | 正式视觉 |
| MaskedAug + T2 | 88% | 88% | 86% | 87.33% | 正式视觉 |
| robust-v1 + T2 | **92%** | **92%** | **86%** | **90.00%** | 最终部署配置 |
| No-aug step 12,732 + T2 | 86% | 74% | 74% | 78.00% | 正式训练对照 |
| No-aug step 16,732 + T2 | 88% | 72% | 72% | 77.33% | 正式训练对照 |

### 总矩阵逐列读法

- Red 是原始红色环境；Blue-v2/Yellow-v2 同时改变环境目标身份，并使用真实 H5 future goal 的受控 recolor。
- 每列 50 env，单个翻转为 2pp；Macro 是三列不加权平均。
- `94%` probe direct Red 不在表中，因为它使用 privileged XYZ 且只有一个颜色。
- robust-v1 的 `90.00%` 是当前可部署的三色单配置最高宏平均。

## 5. 主成果轨迹

最佳可部署单配置的实验记录轨迹为：

`66% → 74% → 74% → 84.7% → 87.33% → 90.00%`

| 节点 | 配置 | 冻结轨迹显示值 |
|---:|---|---:|
| 1 | Pretrained mean | 66% |
| 2 | Memory Seed | 74% |
| 3 | Route2 global hue | 74% |
| 4 | Memory × MaskedAug | 84.7% |
| 5 | MaskedAug + T2 | 87.33% |
| 6 | robust-v1 + T2 | 90.00% |

这是冻结的六节点成果轨迹，不表示所有节点属于同一个 checkpoint 的连续训练 lineage。显示精度按冻结口径保留；总矩阵中的源值仍保留两位小数，其中 `84.7%` 对应精确宏平均 `84.67%`。MaskedAug mean `74.67%` 与 Memory × global hue `80.00%` 仍保留在总矩阵，但不插入该冻结轨迹。

## 6. 定版总结

### 6.1 已定版

- 模型：`checkpoints/lewm-cube-robust_v1/` 下的 robust-v1 权重。
- Planner：T2，即 10 exact memory seeds、20 perturbed seeds、270 free candidates、legacy zero/unit CEM、最终 elite mean。
- Goal：真实 HDF5 future frame；身份变化只做受控 recolor；禁止 synthetic composition。
- 正式成绩：Red / Blue-v2 / Yellow-v2 `92/92/86%`，macro `90.00%`。
- 诊断记录：privileged probe direct Red `94%`，只作为 cost 表达的单色上限证据。

### 6.2 未解决能力边界

- 地板 `48/46%`、光照 `60/68%`、相机 `44/42%` 表明未覆盖视觉轴仍敏感。
- Goal OOD 的视觉 latent cost 只有 `12--32%`；probe cost 提高到 `50--58%`，但仍未达到 in-box `70%` 门。
- Waypoint 在所有主比较中为负增益。
- 规则与 LLM 慢环监督器没有形成可达子目标。
- Expert/near-policy 五步误差约 `5--6mm`，而 planner off-policy 候选约 `85--125mm`；这是当前主要未解决问题。

### 6.3 最终研究判断

当前最可靠的增益来自三项组合：正确的真实帧 goal 协议、针对性视觉增广、保留自由搜索的专家邻域候选注入。选择层四段证伪链 `读数规则 → 盲化 LLM → probe → waypoint` 没有形成稳定部署增益；规则/LLM 慢环干预和四轮动力学微调也没有单独解决剩余失败。

若未来重新研究 off-policy dynamics，前置条件必须是新的可验证假设：数据需要覆盖 planner 的联合五步序列，同时训练方案要保留 expert manifold。不能把 V3 的基础设施中断或 Play-v1 的负结果解释为“继续增加训练步数”即可解决。

## 7. 成本与资源附录

### 7.1 训练成本

| 训练 | Steps | Wall time | 说明 |
|---|---:|---:|---|
| robust-v1 | 4,000 | 10,113s = 2h48m33s | 从 MaskedAug 继续训练。 |
| Control phase A | 12,732 | 5,449.934492s | 零增广，fresh optimizer。 |
| Control phase B | 4,000 | 1,823.654701s | 第二个 fresh optimizer phase。 |
| Control 合计 | 16,732 | 7,273.589193s = 2h01m13.589s | 累计预算，不是连续 optimizer trajectory。 |
| Play-v1 | 5,000 | 6,534.750052s = 1h48m54.750s | 仅训练 dynamics stack。 |

### 7.2 正式评估成本

| 范围 | Groups | Env | Evaluator elapsed |
|---|---:|---:|---:|
| F robust 视觉矩阵 | 9 | 450 | 208.248543s |
| F goal-OOD curve | 4 | 200 | 129.768700s |
| F 合计 | 13 | 650 | 338.017243s |
| G control | 6 | 300 | 130.111620s |
| H probe | 6 | 300 | 202.843740s |
| I waypoint | 9 | 450 | 448.752599s |
| G--I 合计 | 21 | 1,050 | 781.707958s |
| J Play-v1 online | 0 | 0 | NOT RUN |
| J candidate-pool offline | 6 model-color cells | N/A | 14.705492s |

Evaluator elapsed 不含 smoke、进程启动、报告生成与人工审计。Play Measurement-1 没有持久化 elapsed 字段，本文不估算。

### 7.3 LLM 成本

| 项目 | Accounted tokens | 计入口径 |
|---|---:|---|
| 早期 blind reranker | 约 3,500,000 | 近似历史总量，单列。 |
| B1 online | 25,831 | prompt 25,341 + completion 490。 |
| B2 online | 87,552 | 权威 accounted；其中 provider reported 42,408，unknown upper 45,144。 |
| B1 + B2 online | 113,383 | 精确在线 accounted 合计。 |
| B2 offline prompt iteration | 78,131 | 离线开发成本，不计入 online 合计。 |

不得用 B2 provider reported `42,408` 替代在线预算权威值 `87,552`，也不得把离线 prompt iteration `78,131` 混入在线调用成本。

### 7.4 存储与清理

| 项目 | 大小 / 数量 | 审计状态 |
|---|---:|---|
| 已清理历史 MP4 | 20,566 files，486,391,688 bytes，约 0.45GiB | 仓内报告已核。 |
| PushT 数据 | 约 56G | 用户提供运维记录；仓内未审计删除前体积与删除事件。 |
| Play raw NPZ | 约 269MB | Play 数据体检已核。 |
| Play rendered H5 | 约 5.68GB | 正式 train/val H5 已核。 |

视频清理仅删除已验收 MP4；报告、JSON、CSV、cost NPZ 和文档引用图片保留。证据：`outputs/eval/cube/OOD_DISK_CLEANUP_REPORT.md`、`outputs/eval/cube/PLAY_LINE_VERDICT.md`。

## 8. 主要证据索引

| 主题 | 权威产物 |
|---|---|
| Goal 构图 | `outputs/eval/cube/ood/COLOR_OOD_REPORT.md` |
| Selector | `outputs/eval/cube/ood_select/CEM_SELECTOR_REPORT.md` |
| Blind LLM reranker | `outputs/rerank_pilot/RERANK_PILOT_REPORT.md` |
| Memory Seed | `outputs/eval/cube/memory_seed/MEMORY_SEED_REPORT.md` |
| 增广组合 | `outputs/eval/cube/MASKEDAUG_COMBO_REPORT.md` |
| T2 | `outputs/eval/cube/trust_region/TRUST_REGION_REPORT.md` |
| Robust OOD | `outputs/eval/cube/OOD_ROBUSTNESS_REPORT.md` |
| Control + probe | `outputs/eval/cube/CONTROL_AND_PROBEGOAL_REPORT.md` |
| Waypoint | `outputs/eval/cube/waypoint_probe/WAYPOINT_REPORT.md` |
| 长程监督器 | `outputs/eval/cube/longhorizon/BRAIN_LINE_VERDICT.md` |
| Off-policy V1--V3 | `outputs/eval/cube/OFFPOLICY_FINAL_VERDICT.md` |
| Play-v1 终局 | `outputs/eval/cube/PLAY_LINE_VERDICT.md` |
