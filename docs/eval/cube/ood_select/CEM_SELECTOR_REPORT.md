# Cube CEM 最终提取规则：elite mean vs top-1

## 实验口径

- 固定同一组 50 个 dataset rows、seed 42、CEM 10 轮、每轮 300 candidates、top-30 distribution update。
- `mean`：执行第 10 轮 top-30 更新后的 elite mean（既有结果）。
- `top1`：执行第 10 轮 300 candidates 中 latent cost 最小的完整动作序列。
- 同一视觉协议的 cycle 0 候选、cost、mean、variance 与 RNG 严格配对；执行动作分叉后，cycle 25 的物理状态和活跃 env 集合会自然分叉，因此第二次规划不再声称逐位同池。

## 成功率

| 颜色/目标协议 | elite mean | top-1 | top-1 − mean |
|---|---:|---:|---:|
| Red / H5 matched | **36/50（72%）** | 32/50（64%） | **−8 pp** |
| Blue-v2 / recolor | **32/50（64%）** | 30/50（60%） | **−4 pp** |
| Yellow-v2 / recolor | 31/50（62%） | **32/50（64%）** | **+2 pp** |

整体上，top-1 没有稳定优于 elite mean：三色合计为 mean 99/150，top-1 94/150。Yellow 有小幅净提升，但不足以支持全局替换默认规则。

## 逐 env 配对翻盘

| 条件 | mean 失败 → top-1 成功 | mean 成功 → top-1 失败 |
|---|---|---|
| Red | 无 | env 5、9、32、46 |
| Blue-v2 | env 42 | env 5、10、41 |
| Yellow-v2 | env 10、36、37 | env 23、25 |

Yellow env 37 与此前 12-env 审计一致：末轮 latent top-1 候选能成功，而 elite mean 失败。但它是局部机制案例；全 50 局中 Yellow 同时有两个反向损失，因此净收益只有 +1 局。

## Candidate 0 与 top-1 的 cost 差

诊断定义：`末轮 candidate 0 cost − 末轮 top-1 cost`。Candidate 0 是第 9 轮更新后的 mean，被放进第 10 轮候选池并实际评分；它不是 legacy 规则最终执行的第 10 轮更新后 elite mean。因此这个差值不能解释为“mean cost − top1 cost”。

| 条件 | cycle 0：中位数 / 均值 / 最大值 | 所有 cycle：中位数 / 均值 / 最大值 |
|---|---:|---:|
| Red | 1.26 / 6.68 / 105.79 | 1.97 / 6.55 / 105.79 |
| Blue-v2 | 3.67 / 7.71 / 38.99 | 1.28 / 6.46 / 38.99 |
| Yellow-v2 | 5.11 / 7.06 / 35.84 | 4.89 / 7.21 / 35.84 |

差值与物理成功并不单调：

- Red 的四个反向失败，cycle-0 差值从 0.86 到 105.79 都有；latent cost 明显更低也未带来更高总体成功率。
- Blue 的唯一正向翻盘 env 42 差值为 2.88；三个反向失败为 0.008、1.28、3.57。
- Yellow 的三个正向翻盘为 0.84、2.13、19.28；两个反向失败为 0.04、1.45。

所以“top-1 比 candidate 0 低多少”不能作为可靠的 selector 置信度。它只说明末轮采样找到了更低 latent cost，不能证明该动作在真实物理空间更好。

## 结论

1. **ID Red：聚合损失不是主导因素。** Top-1 从 72% 降到 64%，没有任何新救回，反而丢失 4 局；elite mean 的 hedging 在分布内总体有效。
2. **OOD Blue/Yellow：没有一致放大的 top-1 优势。** Blue 下降 4 pp，Yellow 上升 2 pp。颜色 OOD 会制造个别“均值毁掉好解”案例，但不足以让 hard top-1 成为更可靠的全局规则。
3. **下一候选方案应是 top-k 小均值或自适应 selector，而不是直接 top-1。** 建议后续测试 `k∈{3,5,10,30}`，并保留 mean/top1 作为两端基线；若接 LLM/VLM，应让它在 top-K 完整轨迹之间选择，不能使用分支执行后的物理真值。

## 产物

- Red top-1：`/root/autodl-tmp/ailab/outputs/eval/cube/ood_select/red_top1/`
- Blue-v2 top-1：`/root/autodl-tmp/ailab/outputs/eval/cube/ood_select/blue_v2_top1/`
- Yellow-v2 top-1：`/root/autodl-tmp/ailab/outputs/eval/cube/ood_select/yellow_v2_top1/`
- 控制台日志：`/root/autodl-tmp/ailab/logs/eval/cube/ood_select/`

每组包含 50 个三联视频、`results.json/txt`、50 组 cost JSON/NPZ。每个 cycle 的 NPZ 保存完整末轮 `(300,5,25)` 候选、top-1 index/cost/action、candidate 0 cost/action，以及 solver 实际返回动作，可独立验证提取规则。

