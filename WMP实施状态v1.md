# WMP 实施状态 v1

日期：2026-06-24

## 目标

在 `src/mjlab/tasks/WMP` 中按可验证切片实现 WMP/Go1 复现框架。新任务保持
独立实现，复用 mjlab 基础环境、Go1 资产、传感器、地形生成和任务注册机制。

## 当前里程碑

- M0 状态账本：完成。
- M1 任务包骨架与注册：完成，任务 ID 为
  `Mjlab-WMP-Rough-Unitree-Go1`。
- M2 Go1 WMP 环境与观测组：完成第一版，包含 `actor`、`critic`、
  `wm_prop`、`wm_depth`、`wm_forward_height_map`、`amp`。
- M3 WMP 地形与课程：完成第一版，包含 slope、stairs、gap、pit/climb、
  tilt、crawl、rough flat 与 terrain level curriculum。
- M4 WMP RL runner/model/AMP/world-model scaffold：完成第一版，包含 PPO
  rollout、WMP actor-critic、世界模型、深度预测器、AMP 判别器与 checkpoint。
- M5 编译、注册和 smoke 验证：完成。
- M6 深度缓存、部分真相机渲染、AMP 原仓库 mocap loader、pit/climb 地形补强：
  完成。
- M7 运行说明与来源 NOTICE：完成。
- M8 轻量 pytest：完成，覆盖注册、配置序列化、AMP loader、WMP 模块形状、
  fake vec-env runner learn/save/load。
- M9a AMP normalizer：完成，AMP policy/expert transition 在判别器前使用
  running mean/std 标准化，normalizer state 纳入 checkpoint。
- M9b RSSM/Dreamer 风格世界模型训练目标：完成。
- M10 64-env/5-iteration 真实工程回归：完成。

## M9b 变更摘要

- `SimpleWorldModel` 从一步 MLP 预测器升级为轻量 RSSM-style learner：
  - prop/depth encoder。
  - deterministic GRU state。
  - Gaussian stochastic state。
  - action-conditioned prior。
  - observation-conditioned posterior。
  - Dreamer 风格 representation KL 与 dynamics KL。
  - KL free-nats、`dyn_scale`、`rep_scale`。
  - prop reconstruction、next-prop prediction、reward、continuation、depth decoder。
  - prior/posterior entropy metrics。
- actor/critic 侧 world-model feature 维度保持为 `world_model.feature_dim`，
  PPO 接口不变。
- runner 的 world-model update 改为直接使用 `[T, B, D]` rollout 序列训练，
  更接近原 WMP 对短序列 world-model batch 的训练方式。
- 配置新增 `stoch_dim`、`min_std`、`kl_free`、`dyn_scale`、`rep_scale`、
  `prop_loss_scale`、`recon_loss_scale`。

## 验证记录

- `uv --cache-dir .uv-cache run pytest tests\test_wmp_task.py -q`：通过，
  `5 passed`。
- `uv --cache-dir .uv-cache run python -m py_compile ...WMP...`：通过。
- `TrainConfig.from_task("Mjlab-WMP-Rough-Unitree-Go1")` + `asdict`：通过，
  可读取 `stoch_dim=32`、`kl_free=1.0`、`dyn_scale=0.5`、`rep_scale=0.1`。
- fake runner world-model update：通过，输出有限的
  `kl`、`dyn_loss`、`rep_loss`、`prior_ent`、`post_ent`、
  `recon_prop_loss`、`prop_loss`、`reward_loss`、`continue_loss`、`depth_loss`。
- M10 preflight：2 env、1 iteration 真实 env 通过，相机 Warp kernel、深度观测、
  AMP 原仓库 mocap loader 均接通。
- M10 target：64 env、5 iterations 真实 env 通过，保存
  `logs/rsl_rl_m10/go1_wmp/2026-06-24_17-35-16_m10_64env_5iter_fixed/model_5.pt`。
- M10 TensorBoard NaN 审计：通过，PPO、AMP、DepthPredictor、WorldModel 标量均为
  finite。
  - `PPO/policy_loss`: `-0.00963198859244585`
  - `PPO/value_loss`: `7.904122829437256`
  - `AMP/loss`: `0.4446718394756317`
  - `DepthPredictor/loss`: `0.09572990238666534`
  - `WorldModel/loss`: `1.9841079711914062`
  - `WorldModel/kl`: `0.14684335887432098`
  - `Depth/real_fraction`: `0.1666666716337204`
- M10 checkpoint finite 审计：通过，`model_5.pt` 中
  actor/world/depth/AMP/normalizer state 共 12,103,731 个 tensor value 全部 finite。
- 历史环境 smoke 已通过：
  - `actor`: `[1, 45]`
  - `critic`: `[1, 255]`
  - `wm_prop`: `[1, 45]`
  - `wm_depth.depth`: `[1, 64, 64, 1]`
  - `wm_forward_height_map`: `[1, 525]`
  - `amp`: `[1, 30]`

## 变更文件

- `.gitignore`
- `src/mjlab/tasks/WMP/**`
- `tests/test_wmp_task.py`
- `WMP实施状态v1.md`

## 已知限制

- 当前 RSSM 是 MjLab 轻量版，已经具备 Dreamer 风格 prior/posterior/KL 目标，
  但还没有完整迁移原仓库的 Conv image encoder/decoder、symlog distribution、
  discrete stochastic latent 和离线 wm dataset sampler。
- world model 目前使用 rollout 短序列在线训练，尚未建立原 WMP 那种跨 episode
  wm dataset buffer。
- `feet_edge` 仍是占位 penalty，后续需要结合 gap/pit/edge terrain mask 做真实
  边缘惩罚。
- 长训验收尚未完成，目前完成工程 smoke 与轻量回归测试。
- M10 使用 CPU 完成，耗时较长；后续大规模短训建议用单卡或多卡 GPU。
- 相机和 raycast geom groups 不同会触发 sensor_context union warning，当前不影响回归。

## 下一步

- M11 建立原 WMP 风格 world-model replay/dataset buffer，提高 RSSM batch length。
- M12 对齐 depth predictor 预训练流程与 terrain mask/feet edge penalty。
