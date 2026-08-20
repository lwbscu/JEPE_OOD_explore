# Cube 物理读数头（Route 1）报告

## 结论

物理读数头能从 LeWM latent 中准确读出方块位置，也能改善物理最优候选的总体排名；但在 Memory-Seed 最终 300 候选池上，Probe cost Top-1 的成功数在红、蓝、黄三色中都比对应 CEM elite-mean 少 1 个。因此离线门禁失败，按协议停止本路线，不运行 50-env 在线 probe-cost 评估。

## 探针质量

训练、验证、测试按 episode 隔离划分，并从所有划分中排除了固定 50 个评估 episode。正式排序使用 xyz，yaw 仅作辅助监督且权重为 0。

| Probe | 测试集位置中位误差 | R² x | R² y | R² z | R² yaw |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear | 12.00 mm | 0.9921 | 0.9939 | 0.9912 | -0.1332 |
| MLP | 6.47 mm | 0.9958 | 0.9988 | 0.9983 | -0.0764 |

位置可线性/非线性高质量读出；yaw 在按 episode 留出集上不可可靠泛化，但不参与本轮正式成功判据或主 cost。

## Memory-Seed 300 池离线重排

每色为固定 12-env、首个规划周期、25-step open-loop 机制审计。`Top5-any` 是 Top-5 中至少存在一条成功候选的 oracle coverage；`Top5 均匀期望` 是从 Top-5 均匀抽一条的理论期望，均不是新策略的在线成功率。

| 颜色 | Probe Top-1 ever/final | CEM mean ever/final | JEPA Top-1 ever/final | Probe Top5-any ever/final | Top5 均匀期望 ever/final | 门禁 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Red | 8/12 / 8/12 | 9/12 / 9/12 | 8/12 / 7/12 | 11/12 / 10/12 | 7.4/12 / 6.8/12 | Fail（8 < 9） |
| Blue-v2 | 7/12 / 5/12 | 8/12 / 8/12 | 9/12 / 8/12 | 10/12 / 9/12 | 7.2/12 / 5.4/12 | Fail（7 < 8） |
| Yellow-v2 | 6/12 / 6/12 | 7/12 / 7/12 | 7/12 / 7/12 | 9/12 / 9/12 | 6.0/12 / 6.0/12 | Fail（6 < 7） |

## 排名诊断

物理最优候选在 Probe cost 下的中位名次优于原 JEPA cost，但仍离 Top-1 很远：

| 颜色 | 按最小距离：Probe vs JEPA | 按最终距离：Probe vs JEPA |
| --- | ---: | ---: |
| Red | 35.5 vs 62 | 37 vs 62 |
| Blue-v2 | 106.5 vs 116 | 38 vs 73 |
| Yellow-v2 | 66.5 vs 105.5 | 66.5 vs 117.5 |

三色合并后，物理最优候选的平均名次从 JEPA 的 130.61 改善到 Probe 的 108.03（最小距离口径），从 104.81 改善到 75.22（最终距离口径）。这说明读数头包含有用的物理排序信号，但单看预测终点位置仍不足以恢复可用的 Top-1 选择器，可能还缺少接触、夹持稳定性与完整轨迹动力学信息。

## 门禁决定

- `online_evaluation_authorized = false`
- 未启动 `eval_probe_cost.py` 的三色 50-env 在线评估。
- Route 1 到此作为离线负结果结束；不与 Route 2 直接组合。

原始证据：`offline_v1/summary.json`、`offline_v1/REPORT.md` 与 `models/probes/cube_block4d_v1/REPORT.md`。
