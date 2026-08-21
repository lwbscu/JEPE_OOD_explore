# Cube 全轴分级 OOD 基准

生成时间：`2026-08-20T18:54:57.955385+00:00`。

## 协议与读法

- 每轴 4 档、每档同一批 seed=42 的 50 env，规划器固定为完整 T2。
- Color tier0 使用真实 HDF5 `color_t0_red`；相机、光照、地板、尺寸、动作噪声共享 full-state `default_rerender_control`。`all` 模式各执行一次；额外技术副本必须逐 episode 匹配 canonical。
- endpoint 斜率为 `(tier3 成功率 - tier0 成功率) / (tier3 距离 - tier0 距离)`；OLS 对四档做普通最小二乘。负值表示随 OOD 距离增加而下降。
- 部署边界：从 tier0 起连续满足 `成功率 >= 本模型共享 tier0 - 3pp` 的最远档。能力边界：连续满足 `成功率 >= 70%` 的最远档。
- `benchmark_summary.json` 对每个 tier 与其 canonical tier0、以及 robust_v1 与 robust_v2 同档结果保存双向逐 env flips；曲线 CSV 同时给出四类方向的翻转计数。

## 分轴结果

| Model | Axis | T0/T1/T2/T3 | Endpoint slope | OLS slope | Deployment boundary | Capability boundary |
|---|---|---:|---:|---:|---:|---:|
| robust_v1 | color | 92%/86%/92%/86% | -2.000 pp/unit | -1.200 pp/unit | T0 | T3 |
| robust_v1 | camera | 90%/54%/42%/44% | -4.600 pp/unit | -4.500 pp/unit | T0 | T0 |
| robust_v1 | light | 90%/86%/82%/70% | -33.333 pp/unit | -32.000 pp/unit | T0 | T3 |
| robust_v1 | floor | 90%/38%/46%/46% | -44.000 pp/unit | -37.200 pp/unit | T0 | T0 |
| robust_v1 | size | 90%/72%/74%/74% | -1.600 pp/unit | -1.380 pp/unit | T0 | T3 |
| robust_v1 | action_noise | 90%/86%/74%/58% | -106.667 pp/unit | -108.000 pp/unit | T0 | T2 |

## 安全边界声明

边界是本次四档离散采样上的经验边界，不外推为连续空间保证；只声明从 tier0 开始连续通过的档位。动作噪声在 T2 返回物理动作之后、`env.step` 之前注入，并按环境动作空间裁剪。非颜色视觉轴与尺寸轴使用同一 variation 下的完整当前/未来 `qpos+qvel` 重渲染。

## 产物

- `color_curve.csv`、`color_curve.png`：逐档数值与曲线。
- `camera_curve.csv`、`camera_curve.png`：逐档数值与曲线。
- `light_curve.csv`、`light_curve.png`：逐档数值与曲线。
- `floor_curve.csv`、`floor_curve.png`：逐档数值与曲线。
- `size_curve.csv`、`size_curve.png`：逐档数值与曲线。
- `action_noise_curve.csv`、`action_noise_curve.png`：逐档数值与曲线。
- `benchmark_summary.json`：逐 episode 身份核验、双向 flips、斜率与边界的机器可读记录。
- `paired_flips.json`：每 tier 对 canonical tier0、robust_v1 对 robust_v2 的双向逐 env 翻转。
