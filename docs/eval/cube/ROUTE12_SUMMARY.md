# Cube Route 1 + Route 2 总结

## 最终判断

- **Route 1（物理读数头 cost）失败并按门禁停止。** Probe 能准确读出 xyz，也改善了物理最优候选的总体名次，但 Probe Top-1 在 Red/Blue/Yellow 三色中均比 Memory-Seed CEM mean 少 1 个成功，未运行在线 50-env。
- **Route 2（颜色增广微调）通过 Red 回归门禁。** Red 提升 8pp，Yellow-v2 提升 16pp，Blue-v2 无净变化。
- **组合组未运行。** Route 2 自身满足 promotion 条件，但 Route 1 未通过离线 fail-stop；不能把两条路线的数字相加或声称已验证叠加。

## Route 1

### 探针质量

| Probe | Test xyz 中位误差 | R² x/y/z |
| --- | ---: | --- |
| Linear | 12.00 mm | 0.9921 / 0.9939 / 0.9912 |
| MLP | 6.47 mm | 0.9958 / 0.9988 / 0.9983 |

### Memory-Seed 300 池离线门禁

| 条件 | Probe Top-1 ever | CEM mean ever | 结果 |
| --- | ---: | ---: | --- |
| Red | 8/12 | 9/12 | Fail |
| Blue-v2 | 7/12 | 8/12 | Fail |
| Yellow-v2 | 6/12 | 7/12 | Fail |

Probe 把物理最优候选的合并平均名次从 JEPA 的 130.61 改善到 108.03（min-distance），从 104.81 改善到 75.22（final-distance），但不足以把成功候选稳定推到 Top-1。

## Route 2

| 条件 | 原模型 | ColorAug | 变化 |
| --- | ---: | ---: | ---: |
| Red | 72% | 80% | +8pp |
| Blue-v2 | 64% | 64% | 0pp |
| Yellow-v2 | 62% | 78% | +16pp |

ColorAug 之后相对同模型 Red 的差距：Blue -16pp，Yellow -2pp。Yellow 的颜色 OOD 基本修复，Blue 仍是明显残余问题。

## 与 Memory Seed 比较

| 条件 | Memory Seed | ColorAug |
| --- | ---: | ---: |
| Red | 88% | 80% |
| Blue-v2 | 68% | 64% |
| Yellow-v2 | 66% | 78% |

Memory Seed 更偏向修复 Red/Blue 的生成层覆盖，ColorAug 对 Yellow 的表示更有效。下一步最有依据的组合是 **Memory Seed × ColorAug encoder**，而不是继续使用本轮未过门禁的 probe cost。

详细报告：`probe_cost/PROBE_COST_REPORT.md` 与 `coloraug/COLORAUG_REPORT.md`。
