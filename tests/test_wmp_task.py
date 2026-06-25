from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
from mjlab.scripts.train import TrainConfig
from mjlab.sensor import (
  CameraSensor,
  ContactData,
  ContactMatch,
  ContactSensor,
  ContactSensorCfg,
)
from mjlab.tasks.registry import list_tasks, load_runner_cls
from mjlab.tasks.WMP import mdp
from mjlab.tasks.WMP.mdp.commands import WmpVelocityCommand, WmpVelocityCommandCfg
from mjlab.tasks.WMP.rl.amp import MotionLoader
from mjlab.tasks.WMP.rl.modules import (
  ActorCriticWMP,
  DepthPredictor,
  SimpleWorldModel,
)
from mjlab.tasks.WMP.rl.runner import WMPRunner

TASK_ID = "Mjlab-WMP-Rough-Unitree-Go1"


def test_wmp_task_registered_and_cfg_serializable():
  assert TASK_ID in list_tasks()
  assert load_runner_cls(TASK_ID) is WMPRunner

  cfg = TrainConfig.from_task(TASK_ID)
  data = asdict(cfg)

  assert data["agent"]["class_name"] == "WMPRunner"
  assert data["agent"]["obs_groups"]["wm_depth"] == ("wm_depth",)
  assert data["agent"]["depth_predictor"]["camera_update_interval"] == 5
  assert data["agent"]["amp"]["expert_joint_pos_scale"] == (
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
  )
  assert "wm_forward_height_map" in data["agent"]["obs_groups"]
  assert "front_depth_camera" in {
    sensor.name for sensor in cfg.env.scene.sensors if hasattr(sensor, "name")
  }
  terrain_cfg = cfg.env.scene.terrain.terrain_generator
  assert terrain_cfg is not None
  terrain_names = tuple(terrain_cfg.sub_terrains)
  assert terrain_cfg.curriculum
  assert terrain_cfg.num_rows == 10
  assert terrain_cfg.num_cols == 20
  assert len(terrain_names) == 20
  assert sum(name.startswith("gap_") for name in terrain_names) == 5
  assert sum(name.startswith("pit_climb_") for name in terrain_names) == 5
  assert sum(name.startswith("stairs_up_") for name in terrain_names) == 3
  assert sum(name.startswith("stairs_down_") for name in terrain_names) == 3
  assert terrain_names[-1].startswith("rough_flat_")
  twist_cfg = cfg.env.commands["twist"]
  assert isinstance(twist_cfg, WmpVelocityCommandCfg)
  assert twist_cfg.resampling_time_range == (10.0, 10.0)
  assert twist_cfg.heading_command
  assert twist_cfg.ranges.lin_vel_x == (0.0, 0.8)
  assert twist_cfg.ranges.lin_vel_y == (0.0, 0.0)
  assert twist_cfg.ranges.heading == (0.0, 0.0)
  assert twist_cfg.flat_ranges.ang_vel_z == (-1.0, 1.0)
  assert "illegal_thigh_contact" not in cfg.env.terminations
  reward_names = tuple(cfg.env.rewards)
  assert reward_names == (
    "tracking_lin_vel",
    "tracking_ang_vel",
    "lin_vel_z",
    "torques",
    "dof_acc",
    "action_rate",
    "dof_error",
    "feet_air_time",
    "collision",
    "feet_stumble",
    "feet_edge",
    "cheat",
    "stuck",
    "only_positive_clip",
  )
  assert cfg.env.rewards["collision"].params["sensor_names"] == (
    "thigh_ground_touch",
    "shank_ground_touch",
  )
  assert "upright" not in cfg.env.rewards
  assert "feet_slip" not in cfg.env.rewards
  assert "thigh_collision" not in cfg.env.rewards
  assert "trunk_collision" not in cfg.env.rewards


def _quat_to_matrix(quat: tuple[float, float, float, float]) -> torch.Tensor:
  q = torch.tensor(quat, dtype=torch.float64)
  w, x, y, z = q
  return torch.tensor(
    [
      [
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
      ],
      [
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
      ],
      [
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
      ],
    ],
    dtype=torch.float64,
  )


def test_wmp_camera_mount_and_depth_normalization_are_forward_facing():
  cfg = TrainConfig.from_task(TASK_ID).env
  camera_cfg = next(
    sensor for sensor in cfg.scene.sensors if sensor.name == "front_depth_camera"
  )

  assert camera_cfg.parent_body == "robot/trunk"
  assert camera_cfg.pos == (0.27, 0.0, 0.03)
  assert camera_cfg.width == 64
  assert camera_cfg.height == 64
  assert camera_cfg.fovy == 58.0
  assert camera_cfg.data_types == ("depth",)

  camera_mat = _quat_to_matrix(camera_cfg.quat)
  optical_axis_b = camera_mat @ torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)
  right_axis_b = camera_mat @ torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

  assert optical_axis_b[0] > 0.99
  assert abs(float(optical_axis_b[1])) < 1.0e-6
  assert optical_axis_b[2] < -0.05
  assert torch.allclose(
    right_axis_b,
    torch.tensor([0.0, -1.0, 0.0], dtype=torch.float64),
    atol=1.0e-6,
  )

  raw_depth = torch.tensor(
    [[[[0.0], [1.0], [2.0], [3.0], [float("nan")], [float("inf")]]]],
    dtype=torch.float32,
  )
  fake_sensor = CameraSensor(camera_cfg)
  fake_sensor._cached_data = type("CameraData", (), {"depth": raw_depth})()
  fake_sensor._cache_valid = True
  env = type("FakeEnv", (), {"scene": {"front_depth_camera": fake_sensor}})()

  depth = mdp.depth_image(env, "front_depth_camera", near_clip=0.0, far_clip=2.0)

  assert torch.allclose(
    depth.flatten(),
    torch.tensor([0.5, 0.0, 0.5, 0.5, 0.5, 0.5]),
  )
  assert float(depth.min()) >= -0.5
  assert float(depth.max()) <= 0.5


class _FakeRobotData:
  def __init__(self, num_envs: int) -> None:
    self.root_link_lin_vel_b = torch.zeros(num_envs, 3)
    self.root_link_ang_vel_b = torch.zeros(num_envs, 3)
    self.heading_w = torch.zeros(num_envs)
    self.site_pos_w = torch.zeros(num_envs, 4, 3)


class _FakeRobot:
  def __init__(self, num_envs: int) -> None:
    self.data = _FakeRobotData(num_envs)


class _FakeTerrainCfg:
  def __init__(self, terrain_generator) -> None:
    self.terrain_generator = terrain_generator


class _FakeTerrain:
  def __init__(
    self,
    terrain_generator,
    terrain_types: torch.Tensor,
    terrain_levels: torch.Tensor | None = None,
    metadata: dict[str, torch.Tensor] | None = None,
  ) -> None:
    rows = (
      max(1, int(terrain_levels.max().item()) + 1) if terrain_levels is not None else 1
    )
    cols = len(terrain_generator.sub_terrains)
    self.terrain_origins = torch.zeros(rows, cols, 3)
    size_x, size_y = terrain_generator.size
    for row in range(rows):
      for col in range(cols):
        self.terrain_origins[row, col] = torch.tensor([size_x / 2, size_y / 2, 0.0])
    self.terrain_types = terrain_types
    self.terrain_levels = (
      terrain_levels if terrain_levels is not None else torch.zeros_like(terrain_types)
    )
    self.metadata = metadata or {}
    self.cfg = _FakeTerrainCfg(terrain_generator)


class _FakeScene:
  def __init__(self, terrain, robot) -> None:
    self.terrain = terrain
    self._robot = robot

  def __getitem__(self, name: str):
    assert name == "robot"
    return self._robot


class _FakeCommandEnv:
  def __init__(self) -> None:
    from mjlab.terrains import TerrainGeneratorCfg
    from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg

    self.num_envs = 4
    self.device = "cpu"
    terrain_generator = TerrainGeneratorCfg(
      size=(8.0, 8.0),
      curriculum=True,
      num_rows=1,
      num_cols=2,
      sub_terrains={
        "gap_0": BoxFlatTerrainCfg(proportion=1.0),
        "rough_flat_0": BoxFlatTerrainCfg(proportion=1.0),
      },
    )
    terrain = _FakeTerrain(
      terrain_generator,
      terrain_types=torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )
    self.scene = _FakeScene(terrain, _FakeRobot(self.num_envs))
    self.step_dt = 0.02


def test_wmp_command_matches_original_obstacle_and_flat_semantics():
  cfg = WmpVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(10.0, 10.0),
    heading_command=True,
    heading_control_stiffness=0.5,
    rel_heading_envs=1.0,
    rel_forward_envs=1.0,
    ranges=WmpVelocityCommandCfg.Ranges(
      lin_vel_x=(0.4, 0.8),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(-1.0, 1.0),
      heading=(0.0, 0.0),
    ),
    flat_ranges=WmpVelocityCommandCfg.Ranges(
      lin_vel_x=(0.4, 0.8),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(0.7, 0.7),
      heading=(-1.0, 1.0),
    ),
  )
  command = WmpVelocityCommand(cfg, _FakeCommandEnv())

  env_ids = torch.arange(4)
  command._resample_command(env_ids)
  command._update_command()

  assert torch.all(command.vel_command_b[:2, 0] >= 0.4)
  assert torch.all(command.vel_command_b[:2, 1:] == 0.0)
  assert torch.all(command.vel_command_b[2:, 0] >= 0.4)
  assert torch.all(command.vel_command_b[2:, 1] == 0.0)
  assert torch.allclose(command.vel_command_b[2:, 2], torch.full((2,), 0.7))


def _contact_sensor(found: torch.Tensor, force: torch.Tensor | None = None):
  sensor = ContactSensor(
    ContactSensorCfg(
      name="fake_contact",
      primary=ContactMatch(mode="geom", pattern="fake"),
    )
  )
  sensor._cached_data = ContactData(found=found, force=force)
  sensor._cache_valid = True
  return sensor


class _FakeRewardScene:
  def __init__(self, terrain, robot, sensors: dict[str, ContactSensor]) -> None:
    self.terrain = terrain
    self._robot = robot
    self._sensors = sensors

  def __getitem__(self, name: str):
    if name == "robot":
      return self._robot
    return self._sensors[name]


class _FakeCommandManager:
  def __init__(self, command: torch.Tensor) -> None:
    self._command = command

  def get_command(self, name: str) -> torch.Tensor:
    assert name == "twist"
    return self._command


class _FakeRewardManager:
  def __init__(self, reward_buf: torch.Tensor) -> None:
    self._reward_buf = reward_buf
    self._scale_by_dt = True


class _FakeRewardEnv:
  def __init__(self) -> None:
    from mjlab.terrains import TerrainGeneratorCfg
    from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg

    self.num_envs = 4
    self.device = "cpu"
    generator = TerrainGeneratorCfg(
      size=(8.0, 8.0),
      curriculum=True,
      num_rows=5,
      num_cols=3,
      sub_terrains={
        "gap_0": BoxFlatTerrainCfg(proportion=1.0),
        "pit_climb_0": BoxFlatTerrainCfg(proportion=1.0),
        "rough_flat_0": BoxFlatTerrainCfg(proportion=1.0),
      },
    )
    edge_mask = torch.zeros(5, 3, 80, 80, dtype=torch.bool)
    edge_mask[:, 0, 40, 40] = True
    edge_mask[:, 1, 20, 20] = True
    terrain = _FakeTerrain(
      generator,
      terrain_types=torch.tensor([0, 0, 1, 2], dtype=torch.long),
      terrain_levels=torch.tensor([4, 2, 4, 4], dtype=torch.long),
      metadata={"wmp_x_edge_mask": edge_mask},
    )
    robot = _FakeRobot(self.num_envs)
    robot.data.site_pos_w[:, :, :2] = torch.tensor(
      [
        [[4.0, 4.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        [[4.0, 4.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        [[2.0, 2.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        [[4.0, 4.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
      ]
    )
    foot_found = torch.tensor(
      [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
      ],
      dtype=torch.float32,
    )
    foot_force = torch.tensor(
      [
        [[5.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [[5.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [[5.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [[5.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
      ],
      dtype=torch.float32,
    )
    self.scene = _FakeRewardScene(
      terrain,
      robot,
      {"feet_ground_contact": _contact_sensor(foot_found, foot_force)},
    )
    self.command_manager = _FakeCommandManager(
      torch.tensor(
        [
          [0.5, 0.0, 0.0],
          [0.5, 0.0, 0.0],
          [0.5, 0.0, 0.0],
          [0.5, 0.0, 0.0],
        ]
      )
    )
    self.step_dt = 0.02
    self._wmp_iteration = 7000


def test_wmp_feet_edge_stumble_curriculum_and_clip_rewards():
  env = _FakeRewardEnv()
  asset_cfg = type("Cfg", (), {"name": "robot", "site_ids": [0, 1, 2, 3]})()

  edge = mdp.feet_edge(env, "feet_ground_contact", asset_cfg)
  stumble = mdp.feet_stumble(env, "feet_ground_contact")

  assert torch.allclose(edge, torch.tensor([0.55, 0.0, 0.55, 0.0]))
  assert torch.allclose(stumble, torch.tensor([1.0, 0.0, 1.0, 0.0]))
  assert mdp.wmp_feet_edge_coef(0) == 0.1
  assert mdp.wmp_feet_edge_coef(4000) == 0.1
  assert abs(mdp.wmp_feet_edge_coef(7000) - 0.55) < 1.0e-6
  assert mdp.wmp_feet_edge_coef(10000) == 1.0

  env.reward_manager = _FakeRewardManager(torch.tensor([-0.2, 0.3, -1.0, 0.0]))
  correction = mdp.only_positive_reward_clip(env)
  assert torch.allclose(correction * env.step_dt, torch.tensor([0.2, 0.0, 1.0, 0.0]))


def test_wmp_motion_loader_reads_original_json_layout(tmp_path: Path):
  frames = torch.arange(3 * 61, dtype=torch.float32).reshape(3, 61).tolist()
  motion_path = tmp_path / "motion.txt"
  motion_path.write_text(json.dumps({"Frames": frames}), encoding="utf-8")

  loader = MotionLoader((str(motion_path),), amp_obs_dim=30, device=torch.device("cpu"))

  assert loader.has_data
  assert loader.transitions.shape == (2, 60)
  frame_tensor = torch.tensor(frames)
  expected_obs_0 = torch.cat(
    (
      frame_tensor[0, 7:19],
      frame_tensor[0, 31:37],
      frame_tensor[0, 37:49],
    )
  )
  expected_obs_1 = torch.cat(
    (
      frame_tensor[1, 7:19],
      frame_tensor[1, 31:37],
      frame_tensor[1, 37:49],
    )
  )
  assert torch.equal(loader.transitions[0], torch.cat((expected_obs_0, expected_obs_1)))


def test_wmp_motion_loader_joint_stats_and_bias_scale(tmp_path: Path):
  frames = torch.zeros(3, 61, dtype=torch.float32)
  frames[:, 7:19] = torch.arange(36, dtype=torch.float32).reshape(3, 12)
  frames[:, 31:37] = 1.0
  frames[:, 37:49] = 2.0
  motion_path = tmp_path / "motion.txt"
  motion_path.write_text(json.dumps({"Frames": frames.tolist()}), encoding="utf-8")

  loader = MotionLoader(
    (str(motion_path),),
    amp_obs_dim=30,
    device=torch.device("cpu"),
    joint_pos_scale=(2.0,) * 12,
    joint_pos_bias=(1.0,) * 12,
  )

  expected_first = frames[0, 7:19] * 2.0 + 1.0
  expected_second = frames[1, 7:19] * 2.0 + 1.0
  assert torch.allclose(loader.transitions[0, :12], expected_first)
  assert torch.allclose(loader.transitions[0, 30:42], expected_second)

  stats = loader.joint_position_stats()
  assert torch.allclose(stats["min"], expected_first)
  assert torch.allclose(stats["max"], expected_second)
  assert torch.allclose(stats["mean"], (expected_first + expected_second) * 0.5)


def test_wmp_modules_shapes_and_losses_are_finite():
  batch = 4
  actor_dim = 45
  critic_dim = 255
  action_dim = 12
  prop_dim = 45
  depth_dim = 64
  height_dim = 32
  feature_dim = 16
  hidden_dims = (32,)

  actor_critic = ActorCriticWMP(
    actor_dim,
    critic_dim,
    action_dim,
    history_length=3,
    hidden_dims=hidden_dims,
    activation="elu",
    history_latent_dim=16,
    wm_feature_dim=feature_dim,
    wm_latent_dim=8,
    command_dim=3,
    init_std=0.5,
  )
  world_model = SimpleWorldModel(
    prop_dim,
    action_dim,
    depth_dim,
    feature_dim=feature_dim,
    hidden_dims=hidden_dims,
  )
  depth_predictor = DepthPredictor(
    height_dim,
    prop_dim,
    depth_dim,
    hidden_dims=hidden_dims,
  )

  actor = torch.randn(batch, actor_dim)
  critic = torch.randn(batch, critic_dim)
  history = torch.randn(batch, 3, actor_dim)
  prop = torch.randn(batch, prop_dim)
  next_prop = torch.randn(batch, prop_dim)
  actions = torch.randn(batch, action_dim)
  rewards = torch.randn(batch)
  dones = torch.zeros(batch)
  depth = torch.randn(batch, depth_dim)
  height = torch.randn(batch, height_dim)
  wm_feature = world_model.features(prop)

  sampled_action, log_prob, value = actor_critic.act(
    actor,
    critic,
    history,
    wm_feature,
  )
  assert sampled_action.shape == (batch, action_dim)
  assert log_prob.shape == (batch,)
  assert value.shape == (batch,)

  loss, losses = world_model.loss(
    prop,
    actions,
    next_prop,
    rewards,
    dones,
    depth,
    reward_loss_scale=1.0,
    continue_loss_scale=1.0,
    depth_loss_scale=0.1,
    latent_l2_scale=1.0e-4,
  )
  assert torch.isfinite(loss)
  assert all(torch.isfinite(value) for value in losses.values())
  for key in ("kl", "dyn_loss", "rep_loss", "prior_ent", "post_ent"):
    assert key in losses
  assert depth_predictor(height, prop).shape == (batch, depth_dim)


def test_wmp_world_model_loss_handles_sequence_resets():
  batch = 3
  time_steps = 4
  prop_dim = 45
  action_dim = 12
  depth_dim = 16
  world_model = SimpleWorldModel(
    prop_dim,
    action_dim,
    depth_dim,
    feature_dim=16,
    hidden_dims=(32,),
    stoch_dim=8,
  )
  prop = torch.randn(time_steps, batch, prop_dim)
  next_prop = torch.randn(time_steps, batch, prop_dim)
  action = torch.randn(time_steps, batch, action_dim)
  reward = torch.randn(time_steps, batch)
  done = torch.zeros(time_steps, batch)
  done[1, 0] = 1.0
  done[2, 2] = 1.0
  depth = torch.randn(time_steps, batch, depth_dim)

  loss, losses = world_model.loss(
    prop,
    action,
    next_prop,
    reward,
    done,
    depth,
    kl_free=1.0,
    dyn_scale=0.5,
    rep_scale=0.1,
    prop_loss_scale=1.0,
    recon_loss_scale=1.0,
    reward_loss_scale=1.0,
    continue_loss_scale=1.0,
    depth_loss_scale=0.1,
    latent_l2_scale=1.0e-4,
  )

  assert torch.isfinite(loss)
  assert all(torch.isfinite(value) for value in losses.values())


class _FakeWmpEnv:
  num_envs = 2
  num_actions = 12
  max_episode_length = 20

  def __init__(self) -> None:
    self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
    self.common_step_counter = 0

  @property
  def unwrapped(self):
    return self

  def get_observations(self):
    return self._obs()

  def step(self, actions: torch.Tensor):
    self.common_step_counter += 1
    self.episode_length_buf += 1
    obs = self._obs()
    rewards = -0.01 * actions.square().mean(dim=-1)
    dones = torch.zeros(self.num_envs, dtype=torch.bool)
    return obs, rewards, dones, {}

  def _obs(self):
    value = float(self.common_step_counter)
    return {
      "actor": torch.full((self.num_envs, 45), value),
      "critic": torch.full((self.num_envs, 255), value),
      "wm_prop": torch.full((self.num_envs, 45), value),
      "wm_depth": {
        "depth": torch.full((self.num_envs, 64, 64, 1), value / 10.0),
      },
      "wm_forward_height_map": {
        "height": torch.full((self.num_envs, 525), value / 20.0),
      },
      "amp": torch.full((self.num_envs, 30), value),
    }


def _tiny_runner_cfg(expert_motion_files: tuple[str, ...] = ()) -> dict:
  return {
    "num_steps_per_env": 2,
    "save_interval": 100,
    "policy": {
      "hidden_dims": (16,),
      "activation": "elu",
      "history_length": 2,
      "history_latent_dim": 8,
      "wm_feature_dim": 16,
      "wm_latent_dim": 8,
      "command_dim": 3,
      "init_std": 0.5,
    },
    "algorithm": {
      "num_learning_epochs": 1,
      "num_mini_batches": 1,
      "learning_rate": 1.0e-3,
      "gamma": 0.99,
      "lam": 0.95,
      "clip_param": 0.2,
      "entropy_coef": 0.0,
      "value_loss_coef": 1.0,
      "max_grad_norm": 1.0,
      "amp_task_reward_lerp": 0.3,
    },
    "world_model": {
      "hidden_dims": (16,),
      "feature_dim": 16,
      "learning_rate": 1.0e-3,
      "update_interval": 1,
      "reward_loss_scale": 1.0,
      "continue_loss_scale": 1.0,
      "depth_loss_scale": 0.01,
      "latent_l2_scale": 1.0e-4,
    },
    "depth_predictor": {
      "hidden_dims": (16,),
      "learning_rate": 1.0e-3,
      "update_fraction": 1.0,
      "camera_update_interval": 2,
      "camera_num_envs": 1,
      "predict_non_camera_envs": True,
    },
    "amp": {
      "hidden_dims": (16,),
      "learning_rate": 1.0e-3,
      "updates_per_iteration": 1,
      "reward_scale": 1.0,
      "normalize_input": True,
      "expert_motion_files": expert_motion_files,
    },
  }


def test_wmp_runner_learn_save_load_with_fake_env(tmp_path: Path):
  expert_path = tmp_path / "amp_expert.txt"
  expert = torch.arange(8 * 60, dtype=torch.float32).reshape(8, 60)
  expert_path.write_text(
    "\n".join(" ".join(str(float(v)) for v in row) for row in expert),
    encoding="utf-8",
  )
  env = _FakeWmpEnv()
  runner = WMPRunner(
    env,
    _tiny_runner_cfg((str(expert_path),)),
    device="cpu",
  )

  runner.learn(num_learning_iterations=1)

  assert runner.current_learning_iteration == 1
  assert runner._depth_step == 2
  assert runner.amp_normalizer.count > 1.0
  assert runner.world_model.stoch_dim == 32

  checkpoint = tmp_path / "model.pt"
  runner.save(str(checkpoint))
  assert checkpoint.exists()

  loaded = WMPRunner(
    _FakeWmpEnv(),
    _tiny_runner_cfg((str(expert_path),)),
    device="cpu",
  )
  loaded.load(str(checkpoint))
  policy = loaded.get_inference_policy(device="cpu")
  action = policy(loaded.env.get_observations())

  assert loaded.current_learning_iteration == 1
  assert torch.equal(loaded.amp_normalizer.count, runner.amp_normalizer.count)
  assert action.shape == (loaded.env.num_envs, loaded.env.num_actions)
