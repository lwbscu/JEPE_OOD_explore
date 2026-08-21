# Cube 物理轴 OOD 基准

协议：robust_v1 + T2，seed=42，同一 50 个 episode，goal offset=25，budget=50。
质量轴同比缩放 object_0 的 body_mass 与 body_inertia；摩擦轴同比缩放所有启用接触 geom 的三元摩擦系数。

| 颜色 | 轴 | 倍率 | 成功率 | 默认基线 | 差值 | 翻转 F→S / S→F |
|---|---:|---:|---:|---:|---:|---:|
| blue_v2 | friction | ×0.5 | 46/50 (92.00%) | 92.00% | +0.00 pp | 0 / 0 |
| blue_v2 | friction | ×2 | 46/50 (92.00%) | 92.00% | +0.00 pp | 0 / 0 |
| blue_v2 | friction | ×4 | 46/50 (92.00%) | 92.00% | +0.00 pp | 0 / 0 |
| blue_v2 | mass | ×0.5 | 46/50 (92.00%) | 92.00% | +0.00 pp | 0 / 0 |
| blue_v2 | mass | ×2 | 46/50 (92.00%) | 92.00% | +0.00 pp | 0 / 0 |
| blue_v2 | mass | ×4 | 46/50 (92.00%) | 92.00% | +0.00 pp | 0 / 0 |
| red | friction | ×0.5 | 46/50 (92.00%) | 92.00% | +0.00 pp | 0 / 0 |
| red | friction | ×2 | 44/50 (88.00%) | 92.00% | -4.00 pp | 0 / 2 |
| red | friction | ×4 | 44/50 (88.00%) | 92.00% | -4.00 pp | 0 / 2 |
| red | mass | ×0.5 | 46/50 (92.00%) | 92.00% | +0.00 pp | 0 / 0 |
| red | mass | ×2 | 46/50 (92.00%) | 92.00% | +0.00 pp | 0 / 0 |
| red | mass | ×4 | 46/50 (92.00%) | 92.00% | +0.00 pp | 0 / 0 |

## 分轴斜率

- red/mass: +0.000 pp / log2倍率；最低 92.00%。
- red/friction: -1.600 pp / log2倍率；最低 88.00%。
- blue_v2/mass: +0.000 pp / log2倍率；最低 92.00%。
- blue_v2/friction: +0.000 pp / log2倍率；最低 92.00%。

## Measurement-2（固定 12×300）

- red_mass_x4: E_roll 中位数 236.831 mm；>40 mm 比例 99.81%。
- blue_v2_mass_x4: E_roll 中位数 236.711 mm；>40 mm 比例 99.81%。
- red_friction_x0p5: E_roll 中位数 236.831 mm；>40 mm 比例 99.81%。
- blue_v2_friction_x0p5: E_roll 中位数 236.695 mm；>40 mm 比例 99.81%。

## 参数审计

每个 condition 的 `physics_parameters.json` 记录 baseline、期望值、MuJoCo readback、关键接触对有效摩擦及逐步持久性断言。
仅缩放方块 geom 的方案被拒绝：×0.5 会被地板/夹爪接触对的 priority/max 组合抵消。
