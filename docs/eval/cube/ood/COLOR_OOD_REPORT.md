# LeWM Cube 颜色 OOD 评估报告

## 1. 评估口径

- 固定同一批 50 个数据集初态，seed=42，goal offset=25，控制预算=50 步。
- CEM10：10 轮、每轮 300 个候选、top-30 elite mean；RS1：同一求解器仅运行 1 轮，并非执行单个 argmin 候选。
- `matched`：蓝/黄当前图与目标图使用同色 live render；目标图保持当前机械臂，只把方块瞬移到 H5 目标位姿。
- `mismatched`：当前图为蓝/黄 live render，目标图保留原始红色 H5 future frame。
- `matched-v2`：当前图仍为蓝/黄 live render；goal 是原始 H5 future frame，仅将可见的红色方块像素 recolor，机械臂和其余构图逐像素保留。
- 成功判定始终是方块中心距目标不超过 0.04 m，与颜色无关。

## 2. 50-env 成功率矩阵

| 当前颜色 | Goal 图 | CEM10 | RS1 | CEM10 相对 RS1 |
|---|---|---:|---:|---:|
| Red | matched（原 H5 协议） | **36/50 = 72%** | 21/50 = 42% | +30 pp |
| Blue | matched | 23/50 = 46% | 21/50 = 42% | +4 pp |
| Blue | mismatched（红 H5 goal） | **37/50 = 74%** | 19/50 = 38% | +36 pp |
| Blue | matched-v2（真实 H5 构图 recolor） | **32/50 = 64%** | 19/50 = 38% | +26 pp |
| Yellow | matched | 22/50 = 44% | 22/50 = 44% | 0 pp |
| Yellow | mismatched（红 H5 goal） | **35/50 = 70%** | 19/50 = 38% | +32 pp |
| Yellow | matched-v2（真实 H5 构图 recolor） | **31/50 = 62%** | 19/50 = 38% | +24 pp |

`env_9` 的 H5 goal 中方块不可见，recolor 对该行是 no-op；它在四个 v2 组中都失败。排除该行的敏感性结果为：Blue CEM10/RS1 `32/49 = 65.31%` / `19/49 = 38.78%`，Yellow 为 `31/49 = 63.27%` / `19/49 = 38.78%`，不改变结论。

## 3. 12-env 机制审计

表中每格依次为：`候选池有解？ / ever-success 候选数 / 成功候选中的最佳 latent 名次`。名次从 1 开始；`ever-success` 表示 25 步执行过程中至少一次进入 4 cm 成功范围。

| env | Red | Blue matched | Yellow matched |
|---:|:---:|:---:|:---:|
| 0 | 否 / 0 / — | 否 / 0 / — | 否 / 0 / — |
| 1 | 否 / 0 / — | 否 / 0 / — | 否 / 0 / — |
| 2 | 是 / 300 / 1 | 是 / 300 / 1 | 是 / 300 / 1 |
| 6 | 否 / 0 / — | 否 / 0 / — | 否 / 0 / — |
| 7 | 是 / 196 / 1 | 是 / 57 / 1 | 是 / 61 / 4 |
| 11 | 是 / 3 / 58 | 否 / 0 / — | 是 / 1 / 140 |
| 12 | 是 / 247 / 1 | 是 / 75 / 2 | 是 / 110 / 1 |
| 23 | 是 / 71 / 1 | 否 / 0 / — | 否 / 0 / — |
| 26 | 是 / 13 / 25 | 否 / 0 / — | 否 / 0 / — |
| 37 | 是 / 263 / 1 | 是 / 179 / 4 | 是 / 280 / 1 |
| 38 | 否 / 0 / — | 否 / 0 / — | 否 / 0 / — |
| 49 | 是 / 300 / 1 | 是 / 300 / 1 | 是 / 300 / 1 |

汇总：

| 颜色 | 候选池有解 | CEM mean 成功 | ρ(latent cost, min distance) | ρ(latent cost, final distance) |
|---|---:|---:|---:|---:|
| Red | 8/12 | 6/12 | 0.084802 | 0.099407 |
| Blue matched | 5/12 | 2/12 | 0.012607 | 0.060215 |
| Yellow matched | 6/12 | 3/12 | 0.053596 | 0.048234 |

Spearman 使用 SciPy 的 tie-aware ranks。12-env 是按红色正式结果定向选取的 6 个失败例和 6 个成功例，只用于机制诊断，不能与 50-env 成功率直接比较或外推总体性能。

## 4. 结论

1. **颜色 OOD 影响不能单独归因为 dynamics 失效。** Blue/Yellow matched CEM10 为 46%/44%，低于红色 72%；但同样是蓝/黄当前图，使用红色 H5 goal 时仍达到 74%/70%。同时 matched 的首周期最终 sampled best-cost 中位数更低（Blue 4.29、Yellow 4.15），而 mismatched 为 5.57、7.06。性能下降没有伴随 cost 地板抬高，更符合 goal 表示、合成构图或 latent cost 校准失配。

2. **Goal 协议高度敏感。** CEM10 下，mismatched 比 matched 高 28 pp（Blue）和 26 pp（Yellow）。但这里不仅改变目标颜色，也同时改变目标图的渲染来源和机械臂姿态：matched 是 current-arm 的合成图，mismatched 是 H5 future frame。因此可确认“goal 输入协议强烈影响性能”，不能把差异写成纯颜色一致性因果效应。

3. **CEM 精修是否有价值取决于 goal 表示。** 相对 RS1，CEM10 在 Red matched、Blue/Yellow mismatched 上分别提升 30、36、32 pp；在 Blue/Yellow matched 上仅提升 4、0 pp。也就是说，当前 latent goal 具有可用排序结构时，多轮 CEM 很有价值；matched synthetic goal 下，继续压低 latent cost 并未转化为物理成功。12-env 中 Blue/Yellow 分别存在 3 个“池内有解但 CEM mean 失败”的 case，说明后续 reranker 有研究空间；但 best-of-300 是 hindsight oracle，尚不能证明 LLM 能实现该提升。

后续 reranker 盲测优先样本：Red `env 11/26`，Blue `env 7/12/37`，Yellow `env 7/11/12`。

## 5. matched-v2 判定

matched-v2 得到 Blue 64%、Yellow 62%，既不是预设的 `70%+` 完全恢复，也不是 `40%多` 的颜色编码崩溃，而是**构图修复后的显著、但不完全恢复**：

- 相对 matched-v1，Blue 从46%升到64%（+18 pp），Yellow从44%升到62%（+18 pp）。这直接支持 matched-v1 的主要损失来自 current-arm teleport 合成构图，而不是颜色本身。
- 相对原红H5 72%，v2仍低8 pp（Blue）和10 pp（Yellow）；相对同一蓝/黄current、红H5 goal的74%/70%，仍低10/8 pp。因此goal颜色变化仍有次要影响，不能写成“颜色完全无伤encoder”。
- CEM10相对RS1在v2下恢复到+26/+24 pp，接近红色+30 pp和mismatched +32~36 pp，远高于matched-v1的0~4 pp。说明保留真实H5构图后，latent cost地图重新具备明显的精修价值。

因此，后续LLM子目标不应采用“只瞬移方块、机械臂保持当前态”的matched-v1构图。可行优先级是：**检索真实帧 > 在真实帧/构图上做受控recolor或编辑 > 任意姿态合成图**。若要彻底隔离剩余8~10 pp差异是否来自goal颜色，还需保持同一完整goal帧并做更多planner seeds；本轮单seed结果不足以宣称颜色完全无影响。
