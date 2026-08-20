# OGBench Cube Play 数据体检

格式：`cube_play_v1_dataset_v1`。源为官方 `cube-single-play-v0` train/val state NPZ；本报告统计完整原始 1.101M 帧，而非降采样后的渲染 H5。
官方一手源：`https://rail.eecs.berkeley.edu/datasets/ogbench/`。因本机直连该站点时 SSL 失败，字节文件仅经传输镜像 `https://huggingface.co/datasets/ryanhoangt/ogbench_data` 的固定 revision `0290b1be6721a8750c77334c316aca998ba4aa8b` 获取；文件名与 SHA256 仍作为内容身份。

## 规模与字段

- Play：1100 episodes，1,101,100 帧，每局 1001 帧。
- Expert 对照：10000 episodes，2,010,000 帧。
- Play 成功帧占比：**N/A**；原因：official state NPZ has no success or reward field。

## 动作与状态覆盖

- Play action min/max：`[-1.0000, -1.0000, -1.0000, -1.0000, -0.6189]` / `[1.0000, 1.0000, 1.0000, 1.0000, 0.8338]`。
- Expert action min/max（逐维过滤终止占位 NaN）：`[-1.0000, -1.0000, -1.0000, -1.0000, -0.6371]` / `[1.0000, 1.0000, 1.0000, 1.0000, 0.9076]`。
- 动作直方图重叠系数（5 维均值）：**0.9594**；JS divergence：**0.0018 bits**。
- Play block 范围：`[0.2451, -0.3572, 0.0126]` → `[0.6062, 0.3500, 0.3471]` m。
- Expert block 范围：`[0.2444, -0.3556, 0.0130]` → `[0.6108, 0.3550, 0.3480]` m。
- Play EE 范围：`[0.2443, -0.3500, 0.0132]` → `[0.6049, 0.3501, 0.3500]` m。
- Expert EE 范围：`[0.2410, -0.3534, 0.0138]` → `[0.6063, 0.3517, 0.3458]` m。

## 轨迹排除核验

- 全 expert 完整 episode hash 交集：**0**。
- 固定 50 episode 完整 hash 交集：**0**。
- Measurement-1 留出 episode 完整 hash 交集：**0**。
- 全 expert / 固定 50 / Measurement-1 的 1e-3 量化头中尾签名交集：**0 / 0 / 0**。
- 独立采集是官方数据变体的来源声明，不作为数值门；门由上述完整与近重复交集为零给出。

## 转换纪律

正式 H5 保留官方 1000/100 train/val split；每局只存 phase-0 的 raw step `0,5,…,1000`。这是一套 1/5 时间相位样本，训练必须用 `frameskip=1`。图像只由 `set_state(qpos,qvel)` 后渲染，禁止动作 rollout。

- Train H5：`cube_single_play_train_phase0.h5`，201,000 帧、198,000 窗口，SHA256 `2f26e4231f9442c74147becfcdce07c9cf27e72d7106d88fc7eb951e5b8976a0`。
- Val H5：`cube_single_play_val_phase0.h5`，20,100 帧、19,800 窗口，SHA256 `d19f919e5a837ad866e792fae5f698ee2cfedc076b6dae9af3de314802537eee`。
- `manifest.json` SHA256 `629c1ab15153b3000a7fa9091204692f068f70109c0ee2e4220184aab9cf0519`；`validation.json` SHA256 `b02c0b5c039ad2ac5f421c1e70c90cbff057eb6cddfb5a3bd6db8f53f6129e10`，`valid=true`。
- 跨 EGL context 像素复验为 report-only：20 帧中 17 帧 byte-exact，其余最大通道差 1、每帧最多 3 个像素变化；状态、动作、qpos/qvel、observation 与几何量仍全量精确通过。

机器可读全量直方图、逐维矩与交集定义见 `health_report.json`。
