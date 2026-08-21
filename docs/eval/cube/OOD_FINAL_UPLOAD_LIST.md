# OOD 终局双仓补充清单（仅建议，不执行上传）

## 决策

本文件只列下一次人工确认后的同步范围；本轮不执行 GitHub 或 Hugging Face 上传。robust-v2 与 robust-v2b 都没有 `weights_final.pt`，因此不得伪造或发布 final 权重；两个 step-500 隔离权重也不作为部署资产上传。robust-v2b 只同步代码与负结果元数据。E1--E4 只建议同步代码、报告和经白名单选择的小型证据；不上传第三方 cube-double 原始 NPZ。

## GitHub：代码、文档与小证据

建议补充：

| 目标目录 | 本地来源 | 说明 |
|---|---|---|
| `code/` | `le-wm/tools/prepare_cube_rerender_v2.py` | 状态重渲染数据管线 |
| `code/` | `le-wm/cube_robust_v2.py`、`le-wm/train_cube_robust_v2.py` | fail-stop 训练实现 |
| `code/` | `le-wm/cube_robust_v2b.py`、`le-wm/train_cube_robust_v2b.py` | 联合解冻的预注册负结果实现 |
| `code/` | `le-wm/eval_cube_robust_v2b.py` | 训练 PASS 后才授权的 11 组评估器；本轮未运行 |
| `code/tools/` | `le-wm/tools/summarize_cube_robust_v2b.py` | v2b 结果汇总器；本轮无结果可汇总 |
| `code/` | `le-wm/eval_cube_ood_final.py` | 11 组与六轴 evaluator |
| `code/tools/` | `le-wm/tools/summarize_cube_ood_final.py` | 曲线、斜率、边界与 flips 汇总 |
| `code/` | `le-wm/eval_cube_physics_ood.py` | E1 质量/摩擦正式 evaluator |
| `code/tools/` | `le-wm/tools/summarize_cube_physics_ood.py` | E1 汇总、斜率与 paired flips |
| `code/` | `le-wm/eval_noise_marginalized_cem.py` | E2 M=8 执行噪声边际化 evaluator |
| `code/tools/` | `le-wm/tools/summarize_noise_marginalized_cem.py` | E2 成功率与 flips 汇总 |
| `code/tools/` | `le-wm/tools/analyze_cube_ood_detector.py` | E3 冻结产物可测性与 proxy 分析 |
| `code/` | `le-wm/eval_cube_double_transfer.py` | E4 cube-double 零样本 evaluator |
| `code/tools/` | `le-wm/tools/prepare_cube_double_eval.py` | E4 固定 50 对准备工具 |
| `docs/` | `outputs/eval/cube/OOD_FINAL_REPORT.md` | 本轮总报告 |
| `docs/` | `outputs/eval/cube/ood_benchmark/OOD_BENCHMARK_REPORT.md` | 六轴基准报告 |
| `docs/` | `outputs/eval/cube/physics_ood/PHYSICS_OOD_REPORT.md` | E1 物理轴报告 |
| `docs/` | `outputs/eval/cube/noise_marginalized_cem/REPORT.md` | E2 边际化负结果 |
| `docs/` | `outputs/eval/cube/ood_detector/OOD_DETECTOR_REPORT.md` | E3 自检可测性结论 |
| `docs/` | `outputs/eval/cube/cube_double_transfer/CUBE_DOUBLE_TRANSFER_REPORT.md` | E4 零样本迁移报告 |
| `docs/data/` | `datasets/cube_rerender_v2/RERENDER_DATA_REPORT.md` | 数据与 QC 报告 |
| `docs/` | `outputs/eval/cube/OOD_FINAL_UPLOAD_LIST.md` | 本清单 |
| `evidence/ood_final/` | `outputs/eval/cube/ood_benchmark/*_curve.png` | 六轴曲线 6 张 |
| `evidence/ood_final/` | `datasets/cube_rerender_v2/qc/qc_clip_00.png` ... `qc_clip_04.png` | 重渲染 contact sheets 5 张 |
| `evidence/ood_final/` | `outputs/eval/cube/ood_benchmark/*_curve.csv`、`robust_v1/part1_invocation.json`、`outputs/eval/cube/final_matrix.csv` | 小型机器可读证据 |
| `evidence/ood_final/e1/` | `outputs/eval/cube/physics_ood/summary.json`、`success_matrix.csv` | 物理轴机器可读证据 |
| `evidence/ood_final/e2/` | `outputs/eval/cube/noise_marginalized_cem/summary.json`、`matrix.csv`、`paired_flips.json` | M=8 内部配对证据 |
| `evidence/ood_final/e3/` | `outputs/eval/cube/ood_detector/requested_score_availability.json`、`diagnostic_proxy_axis_auc.csv`、`diagnostic_proxy_calibration.png` | NOT MEASURED 边界与 proxy 诊断 |
| `evidence/ood_final/e4/` | `outputs/eval/cube/cube_double_transfer/part3_gate.json`、`prepared/pair_manifest.json`、`formal/results.json` | cube-double gate、50 对身份与正式向量 |
| `evidence/ood_final/e4/videos/` | `formal/videos/env_4.mp4`、`env_17.mp4`、`env_27.mp4`、`env_28.mp4` | 四个成功 episode 的最小视频证据 |

README 建议新增：robust-v2 `STOPLINE_FAIL (+49.319% @ step 500)`；robust-v2b `STOPLINE_FAIL (+4,638.978% @ step 500)` 且重渲染单步微调线正式关闭；robust-v1 六轴与质量/摩擦边界；M=8 `0/4` 达到 `+5pp` 且核心成本 `5.41×`；请求的 OOD 自检分数均 `NOT MEASURED`；cube-double 零样本 `4/50（8%）`；明确 robust-v1 仍是最终模型。

## Hugging Face Dataset：数据、元数据与报告

建议补充：

| 目标目录 | 本地来源 | 大小 / 用途 |
|---|---|---|
| `datasets/cube_rerender_v2/` | `datasets/cube_rerender_v2/cube_rerender_v2.h5` | 3,263,498,388 bytes，100k 重渲染帧 |
| 同上 | `manifest.json`、`validation.json`、`variation_bank.npz` | 身份、QC 与 variation bank |
| 同上 `qc/` | `qc_clip_00.png` ... `qc_clip_04.png`、`qc_qc_report.json` | 可视 QC |
| `reports/` | `OOD_FINAL_REPORT.md`、`OOD_BENCHMARK_REPORT.md`、`RERENDER_DATA_REPORT.md` | 结论与数据卡补充 |
| `evidence/ood_final/` | 六组 `*_curve.csv/png`、`benchmark_summary.json`、`paired_flips.json`、`robust_v1/part1_invocation.json` | 分级基准、11 组 reference 和逐 env 证据 |
| `training/robust_v2_negative/` | `run_plan.json`、`completed.json`、`stopline_event.json`、`stopline_history.json`、`frozen_integrity.json` | 负结果元数据，不含权重 |
| `training/robust_v2b_negative/` | `outputs/train/robust_v2b/robust_v2b_rerender_seed3072/` 下的 `run_plan.json`、`completed.json`、`stopline_event.json`、`stopline_history.json`、`frozen_integrity.json` | 联合解冻负结果元数据，不含任何 checkpoint/权重 |
| `reports/` | `PHYSICS_OOD_REPORT.md`、`noise_marginalized_cem/REPORT.md`、`OOD_DETECTOR_REPORT.md`、`CUBE_DOUBLE_TRANSFER_REPORT.md` | E1--E4 正式结论 |
| `evidence/ood_final/e1_e2_e3/` | E1 `summary.json/success_matrix.csv`、E2 `summary.json/matrix.csv/paired_flips.json`、E3 `requested_score_availability.json/diagnostic_proxy_axis_auc.csv` | 小型机器可读证据 |
| `evidence/ood_final/e4/` | cube-double `part3_gate.json`、`pair_manifest.json`、`formal/results.json` 与四个成功视频 | 零样本协议、结果与最小视觉证据 |

H5 上传前应再次核验 SHA-256：`6f9275e1b1d3a4ea3d9d3e5cd37a57ddde76f644e99743987737da0dd96e1967`，并复核上游数据许可是否允许发布派生重渲染数据。README/Dataset Card 应明确源 expert H5 不随仓发布、重渲染数据由本地 expert 状态派生，且 50 个评估 episode 与 Measurement-1 均零重叠。

## 明确不上传

- `/root/autodl-tmp/ailab/llm_api.md`、任何 token/key/credential 文件或日志片段。
- `datasets/ogbench/cube_single_expert.h5` 原始第三方专家数据。
- `datasets/cube_rerender_v2/work/` worker shards、`provenance_invalid_initial/`、smoke H5、临时/恢复文件。
- `checkpoints/lewm-cube-robust_v2/.../weights_stopped_step500.pt`、`last.ckpt`、`stopline_step500.ckpt`：均是隔离或恢复资产，不是最终权重。
- `checkpoints/lewm-cube-robust_v2b/.../weights_stopped_step500.pt`、`last.ckpt`、`stopline_step500.ckpt` 与 `config.json`：不得上传；v2b 没有 `weights_final.pt`。
- `datasets/ogbench_cube_double/cube-double-play-v0*.npz`：官方第三方原始数据，不上传；Dataset Card 给出官方源与冻结 transport revision。
- `outputs/eval/cube/cube_double_transfer/prepared/cube_double_eval_pairs.npz`：含从第三方状态派生的完整 prepared pairs；许可复核前不上传。可上传不含密钥的 `pair_manifest.json` 与正式结果元数据。
- cube-double 的 46 个失败视频与大体积 trust/cost NPZ 默认不上传；最小证据只建议四个成功视频，若未来需要完整审计再单独确认。
- 任何现有旧仓未在待传 staging 中通过密钥模式扫描的文件。

## 上传前检查（下一次执行时）

1. 重新组装 staging，禁止从项目根递归复制；只按白名单复制本清单文件。
2. 对 staging 全量扫描 `sk-`、`hf_`、`ghp_`、Bearer、AWS key 等模式，并保留扫描报告；不输出密钥原文。
3. 计算文件数、总字节数和 SHA-256 清单；先 GitHub 小文件，再 HF 大 H5 断点续传。
4. 远端校验 Git commit、HF 文件数，并对 H5、manifest、validation 与至少两张 PNG 抽样比对 SHA-256。
5. Dataset Card 只声明实际完成项：robust-v2 / v2b online 为 `NOT RUN / NOT MEASURED`；cube-double 是 robust-v1 + T2 的正式零样本 `4/50（8%）`，不得写成 robust-v2 结果。
