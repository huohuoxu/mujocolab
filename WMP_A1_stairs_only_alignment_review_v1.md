# Mjlab-WMP-Stairs-Only-Unitree-A1 对齐审查 v1

日期：2026-06-26

审查对象：`Mjlab-WMP-Stairs-Only-Unitree-A1`

对照对象：本地原 WMP 仓库 `E:\开源项目\WMP-master\WMP-master`，重点文件：

- `legged_gym/envs/a1/a1_amp_config.py`
- `legged_gym/envs/base/legged_robot.py`
- `rsl_rl/runners/wmp_runner.py`
- `rsl_rl/modules/actor_critic_wmp.py`
- `rsl_rl/algorithms/amp_ppo.py`
- `rsl_rl/algorithms/amp_discriminator.py`
- `rsl_rl/modules/depth_predictor.py`
- `rsl_rl/datasets/motion_loader.py`

## 结论

当前 `Mjlab-WMP-Stairs-Only-Unitree-A1` 不是“除地图外均与原项目一致”。它已经完成了 A1 资产、A1 默认姿态、A1 PD、`action_scale=0.25`、AMP motion 语义和主要 PPO 超参的第一层对齐，但训练闭环仍存在多处会明显影响学习行为的差异。

## 已对齐项

1. A1 默认站姿基本对齐。
   - 当前：`src/mjlab/asset_zoo/robots/unitree_a1/a1_constants.py`
   - 原项目：`a1_amp_config.py` 的 `init_state.default_joint_angles`
   - hip/thigh/calf 默认角均按 A1 AMP 配置迁移。

2. 控制周期和动作尺度对齐。
   - 原项目：`dt=0.005`、`decimation=4`、`action_scale=0.25`
   - 当前 mjlab：仿真步长 `0.005`、env step `0.02`、A1 action scale `0.25`

3. A1 AMP motion 不再做 Go1 hip 符号翻转。
   - 当前 A1 cfg：`expert_joint_pos_scale=None`
   - AMP observation 语义保持 `[joint_pos(12), base_lin_vel(3), base_ang_vel(3), joint_vel(12)]`

4. 主要 task reward 项已基本对齐。
   - `tracking_lin_vel`
   - `tracking_ang_vel`
   - `lin_vel_z`
   - `torques`
   - `dof_acc`
   - `action_rate`
   - `dof_error`
   - `feet_air_time`
   - `collision`
   - `feet_stumble`
   - `feet_edge`
   - `cheat`
   - `stuck`
   - `only_positive_rewards`

5. 相机核心参数基本对齐。
   - 位置 `(0.27, 0, 0.03)`
   - 64x64 depth
   - fov 58
   - near/far `[0, 2]`
   - update interval 5

## 高优先级不一致

### Critical 1：world model 不是原项目 Dreamer world model

原项目 `rsl_rl/runners/wmp_runner.py`：

- 构造 `dreamer.models.WorldModel`
- 使用 Dreamer config
- `wm_feature_dim = self.wm_config.dyn_deter`
- 每 5 个 env step 聚合一次 action/reward
- 维护 `wm_dataset` 和 `wm_buffer`
- episode reset 时把 buffer 写入 dataset
- 数据量超过 `train_start_steps` 后训练 world model
- 按序列 batch 训练 `batch_length`

当前项目：

- 使用 `SimpleWorldModel`
- feature dim 默认 128
- 直接用当前 rollout 训练
- 没有原项目的 episode-level replay dataset
- 没有原 Dreamer 的完整训练流程和配置体系

影响：这不是地图差异，而是核心 WMP 表征学习机制差异。即使环境和奖励完全一致，actor 得到的 `h_t` 也不是原项目语义。

### Critical 2：AMP/PPO 联合优化方式不完全一致

原项目 `AMPPPO`：

- 一个 optimizer 同时管理 `actor_critic`、`discriminator.trunk`、`discriminator head`
- PPO loss 中同时包含 `amp_loss + grad_pen_loss`
- AMP policy samples 来自 replay buffer
- discriminator reward 通过 `predict_amp_reward()` 加到 task reward

当前项目：

- `actor_critic` optimizer 和 `amp_optimizer` 分开
- PPO policy update 只用 combined reward 的 advantage，不在同一个 loss 中反传 discriminator loss
- AMP policy transitions 直接来自当前 rollout，不是原项目 replay buffer

影响：AMP 仍然影响 reward，但训练动力学不是原 AMPPPO。

### High 1：ActorCriticWMP 网络结构不一致

原项目 `ActorCriticWMP`：

- history encoder hidden dims `[256, 128]`
- actor hidden dims `[256, 128, 64]`
- critic hidden dims `[512, 256, 128]`
- latent dim `32 + 3`
- world model latent dim `32`
- actor 输入是 `[history_latent, command, wm_latent]`
- critic 输入是 `[critic_obs, wm_latent]`

当前项目：

- 统一 `hidden_dims=(512, 256, 128)` 用于 history encoder、actor、critic
- history latent 默认 128
- world model latent 默认 64
- actor/critic 拓扑语义相似，但尺寸和编码器结构不一致

影响：不是致命 bug，但会显著改变训练稳定性和容量分配。

### High 2：domain randomization 不完整

原项目 A1 AMP 配置启用：

- friction
- restitution
- base mass
- link mass
- COM
- PD gains
- motor strength
- action latency
- push

当前项目已覆盖：

- foot friction
- base COM
- push

当前缺少或未等价：

- restitution randomization
- base mass randomization
- link mass randomization
- PD gain randomization
- motor strength randomization
- action latency
- critic 中对应 privileged randomization 信息也不完整

影响：短训看起来可能更容易，但与原项目鲁棒训练分布不一致。

### High 3：critic privileged observation 不完整

原项目 `privileged_dim = 24 + 26 + 3`，critic obs 包含：

- base linear velocity
- proprioception
- friction/restitution/mass/COM/PD gains 等随机化信息
- contact force / contact flag
- height scan

当前 critic obs 包含：

- base linear velocity
- actor terms
- height scan
- foot height
- foot contact
- foot contact forces

当前缺少多项 domain randomization privileged 信息。

影响：value function 的 teacher 信息不等价，尤其在开启完整 randomization 后会更明显。

### High 4：depth predictor 训练方式不一致

原项目：

- `DepthPredictor` 是 MLP encoder + ConvTranspose decoder
- `training_interval=10`
- `training_iters=1000`
- `batch_size=1024`
- 从 world-model dataset 采样真实 depth / forward height map / prop
- 对 tilt/crawl 等特殊地形有 camera env 策略

当前项目：

- depth predictor 结构较轻
- 直接从当前 rollout 的 real camera mask 样本训练
- 没有 dataset 采样训练 1000 iter 的机制
- 没有原项目的 `depth_index_without_crawl_tilt` 等 camera 分配逻辑

影响：depth predictor 的学习节奏和数据分布不一致。

### High 5：reset 随机化不一致

原项目普通 reset：

- joint pos = default * uniform(0.5, 1.5)
- root velocity uniform(-0.5, 0.5)
- base xy 在 terrain origin 附近随机

当前项目：

- joint position offset range `(0.0, 0.0)`
- joint velocity `(0.0, 0.0)`
- root pose 有随机化，但 velocity_range 为空

影响：当前初始状态分布更窄，不能视为原项目一致。

## 中优先级不一致

1. `num_envs` 默认不一致。
   - 原 A1 AMP：4096
   - 当前注册默认：1，依赖 CLI 覆盖

2. `max_iterations` 不一致。
   - 原 A1 AMP：20000
   - 当前：10000

3. `save_interval` 不一致。
   - 原 A1 AMP：1000
   - 当前：200

4. checkpoint 内容不一致。
   - 原项目保存 world model、wm optimizer、depth predictor、actor optimizer，但 discriminator/amp normalizer 被注释掉未保存。
   - 当前项目保存 policy/world/depth/AMP/discriminator/normalizer/curriculum state，更完整，但不等价。

5. 任务名是 stairs-only，但原项目没有一个严格等价的 stairs-only 配置。
   - 因此“仅地图不同”的可比对象实际是 `A1AMPCfg` rough 任务在地形集合上裁剪后的假设版本。

## 地图差异之外的当前对齐判断

| 子系统 | 当前状态 | 是否与原项目一致 |
|---|---|---|
| A1 默认姿态 | 已迁移 | 基本一致 |
| A1 PD/action scale | 已迁移 | 基本一致 |
| AMP motion 数据语义 | A1 下无 hip 翻转 | 基本一致 |
| task reward 项 | 主要项已补齐 | 大体一致 |
| reward curriculum | feet_edge schedule 已做 | 大体一致 |
| only_positive rewards | correction term 实现 | 语义接近 |
| PPO 超参 | 主要超参已对齐 | 大体一致 |
| ActorCritic 网络结构 | 拓扑类似，尺寸不同 | 不一致 |
| AMP/PPO 联合优化 | 分离 optimizer/replay 行为不同 | 不一致 |
| World model | 简化 RSSM，无原 Dreamer replay | 不一致 |
| Depth predictor | 结构和训练节奏不同 | 不一致 |
| Domain randomization | 只覆盖一部分 | 不一致 |
| Critic privileged obs | 缺少 randomization privileged 信息 | 不一致 |
| Reset 分布 | 当前更窄 | 不一致 |

## 建议修复顺序

1. 先决定目标：是否要把 A1 stairs-only 做成“原 WMP rough 的地形子集”，还是仅作为调试任务。
2. 若目标是“除地图外一致”，优先补齐：
   - ActorCriticWMP 网络尺寸
   - AMP replay buffer + AMPPPO 单 optimizer 训练结构
   - reset 随机化
   - domain randomization 和 critic privileged obs
3. 再补齐：
   - 原 Dreamer WorldModel replay dataset 训练
   - ConvTranspose depth predictor 和原训练 schedule
4. 最后再比较 stairs-only 训练曲线；否则训练失败/不稳定不能简单归因到地图。

