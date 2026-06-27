from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
from mjlab.actuator import BuiltinPositionActuator
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scripts.train import TrainConfig
from mjlab.sensor import (
  CameraSensor,
  ContactData,
  ContactMatch,
  ContactSensor,
  ContactSensorCfg,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_runner_cls
from mjlab.tasks.WMP import mdp
from mjlab.tasks.WMP.mdp.commands import WmpVelocityCommand, WmpVelocityCommandCfg
from mjlab.tasks.WMP.rl.amp import MotionLoader
from mjlab.tasks.WMP.rl.amp import AMPDiscriminator
from mjlab.tasks.WMP.rl.modules import (
  ActorCriticWMP,
  DepthPredictor,
  DreamerWorldModelAdapter,
  SimpleWorldModel,
)
from mjlab.tasks.WMP.rl.runner import WMPRunner

TASK_ID = "Mjlab-WMP-Rough-Unitree-Go1"
STAIRS_ONLY_TASK_ID = "Mjlab-WMP-Stairs-Only-Unitree-Go1"
A1_TASK_ID = "Mjlab-WMP-Rough-Unitree-A1"
A1_STAIRS_ONLY_TASK_ID = "Mjlab-WMP-Stairs-Only-Unitree-A1"


def test_wmp_task_registered_and_cfg_serializable():
  assert TASK_ID in list_tasks()
  assert load_runner_cls(TASK_ID) is WMPRunner

  cfg = TrainConfig.from_task(TASK_ID)
  data = asdict(cfg)

  assert data["agent"]["class_name"] == "WMPRunner"
  assert data["agent"]["obs_groups"]["wm_depth"] == ("wm_depth",)
  assert data["agent"]["depth_predictor"]["camera_update_interval"] == 5
  assert data["agent"]["amp"]["reward_scale"] == 0.01
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
  reward_weights = {name: term.weight for name, term in cfg.env.rewards.items()}
  assert reward_weights == {
    "tracking_lin_vel": 1.5,
    "tracking_ang_vel": 0.5,
    "lin_vel_z": -1.0,
    "torques": -1.0e-4,
    "dof_acc": -2.5e-7,
    "action_rate": -0.03,
    "dof_error": -0.04,
    "feet_air_time": 0.5,
    "collision": -1.0,
    "feet_stumble": -0.1,
    "feet_edge": -1.0,
    "cheat": -1.0,
    "stuck": -1.0,
    "only_positive_clip": 1.0,
  }
  assert "upright" not in cfg.env.rewards
  assert "feet_slip" not in cfg.env.rewards
  assert "thigh_collision" not in cfg.env.rewards
  assert "trunk_collision" not in cfg.env.rewards
  assert tuple(cfg.env.observations["critic"].terms) == (
    "foot_contact",
    "foot_contact_forces",
    "d_gains",
    "p_gains",
    "base_com",
    "base_mass",
    "restitution",
    "friction",
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "command",
    "joint_pos",
    "joint_vel",
    "actions",
    "height_scan",
  )
  assert cfg.env.observations["critic"].terms["foot_contact_forces"].params == {
    "sensor_name": "feet_ground_contact",
    "scale": 0.005,
  }
  assert cfg.env.observations["critic"].terms["d_gains"].params == {
    "key": "d_gains",
    "dim": 12,
    "scale": 5.0,
  }
  assert cfg.env.observations["critic"].terms["p_gains"].params == {
    "key": "p_gains",
    "dim": 12,
    "scale": 5.0,
  }
  assert cfg.env.observations["critic"].terms["base_com"].params == {
    "key": "base_com",
    "dim": 3,
    "scale": 20.0,
  }
  assert cfg.env.events["reset_robot_joints"].func is mdp.reset_joints_by_scale
  assert cfg.env.events["reset_robot_joints"].params["position_scale_range"] == (
    0.5,
    1.5,
  )
  assert cfg.env.events["cache_privileged_randomization"].func is (
    mdp.cache_privileged_randomization
  )


def test_wmp_play_randomizes_terrain_origin_before_base_reset():
  cfg = load_env_cfg(TASK_ID, play=True)
  reset_event_names = tuple(cfg.events)

  assert "push_robot" not in cfg.events
  assert cfg.scene.num_envs == len(cfg.scene.terrain.terrain_generator.sub_terrains)
  assert "randomize_play_terrain" in cfg.events
  assert reset_event_names.index("randomize_play_terrain") < reset_event_names.index(
    "reset_base"
  )
  assert cfg.events["randomize_play_terrain"].func is mdp.randomize_play_terrain
  assert cfg.curriculum == {}


def test_wmp_stairs_only_task_registered_without_amp():
  assert STAIRS_ONLY_TASK_ID in list_tasks()
  assert load_runner_cls(STAIRS_ONLY_TASK_ID) is WMPRunner

  cfg = TrainConfig.from_task(STAIRS_ONLY_TASK_ID)

  assert cfg.agent.class_name == "WMPRunner"
  assert cfg.agent.run_name == "stairs_only_no_amp"
  assert cfg.agent.clip_actions == 6.0
  assert cfg.agent.policy.init_std == 1.0
  assert cfg.agent.policy.min_std is None
  assert cfg.agent.policy.max_std is None
  assert cfg.agent.policy.history_exclude_command
  assert cfg.agent.policy.clip_observations == 100.0
  assert cfg.agent.algorithm.entropy_coef == 0.01
  assert cfg.agent.algorithm.use_clipped_value_loss
  assert cfg.agent.algorithm.schedule == "adaptive"
  assert cfg.agent.algorithm.desired_kl == 0.01
  assert cfg.agent.algorithm.vel_predict_coef == 1.0
  assert cfg.agent.amp.expert_motion_files == ()
  assert cfg.agent.amp.reward_scale == 0.0
  assert cfg.agent.amp.updates_per_iteration == 0
  assert not cfg.agent.amp.diagnostics_enabled
  assert cfg.env.rewards["only_positive_clip"].params == {}

  terrain_cfg = cfg.env.scene.terrain.terrain_generator
  assert terrain_cfg is not None
  terrain_names = tuple(terrain_cfg.sub_terrains)
  assert terrain_cfg.curriculum
  assert terrain_cfg.num_rows == 10
  assert terrain_cfg.num_cols == 2
  assert terrain_names == (
    "stairs_up_0",
    "stairs_down_0",
  )

  play_cfg = load_env_cfg(STAIRS_ONLY_TASK_ID, play=True)
  play_terrain_cfg = play_cfg.scene.terrain.terrain_generator
  assert play_terrain_cfg is not None
  assert tuple(play_terrain_cfg.sub_terrains) == terrain_names
  assert play_cfg.scene.num_envs == len(terrain_names)
  assert "randomize_play_terrain" in play_cfg.events
  assert play_cfg.curriculum == {}


def test_wmp_a1_tasks_registered_with_a1_action_scale_and_amp():
  assert A1_TASK_ID in list_tasks()
  assert A1_STAIRS_ONLY_TASK_ID in list_tasks()
  assert load_runner_cls(A1_TASK_ID) is WMPRunner
  assert load_runner_cls(A1_STAIRS_ONLY_TASK_ID) is WMPRunner

  cfg = TrainConfig.from_task(A1_TASK_ID)

  assert cfg.agent.experiment_name == "a1_wmp"
  assert cfg.agent.run_name == "rough"
  assert cfg.agent.amp.reward_scale == 0.01
  assert cfg.agent.amp.updates_per_iteration == 1
  assert cfg.agent.amp.expert_motion_files
  assert cfg.agent.amp.expert_joint_pos_scale is None
  assert cfg.agent.amp.expert_joint_pos_bias is None
  assert cfg.env.actions["joint_pos"].scale == 0.25

  robot_cfg = cfg.env.scene.entities["robot"]
  assert robot_cfg.init_state.joint_pos["FR_hip_joint"] == -0.1
  assert robot_cfg.init_state.joint_pos["FL_hip_joint"] == 0.1
  assert robot_cfg.init_state.joint_pos["RR_thigh_joint"] == 1.0
  assert robot_cfg.init_state.joint_pos[".*_calf_joint"] == -1.5

  stairs_cfg = TrainConfig.from_task(A1_STAIRS_ONLY_TASK_ID)
  assert stairs_cfg.agent.experiment_name == "a1_wmp"
  assert stairs_cfg.agent.run_name == "stairs_only_amp"
  assert stairs_cfg.agent.amp.expert_motion_files
  assert stairs_cfg.agent.amp.reward_scale == 0.01
  assert stairs_cfg.agent.amp.updates_per_iteration == 1
  assert stairs_cfg.agent.amp.diagnostics_enabled
  assert stairs_cfg.agent.amp.expert_joint_pos_scale is None
  assert stairs_cfg.agent.amp.expert_joint_pos_bias is None
  assert stairs_cfg.env.actions["joint_pos"].scale == 0.25


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


class _FakePlayEnv:
  def __init__(self, num_envs: int, num_rows: int = 3, num_cols: int = 5) -> None:
    self.num_envs = num_envs
    self.device = "cpu"
    terrain = type("Terrain", (), {})()
    terrain.terrain_origins = torch.zeros(num_rows, num_cols, 3)
    for row in range(num_rows):
      for col in range(num_cols):
        terrain.terrain_origins[row, col] = torch.tensor(
          [float(row), float(col), 0.0]
        )
    terrain.terrain_levels = torch.zeros(num_envs, dtype=torch.long)
    terrain.terrain_types = torch.zeros(num_envs, dtype=torch.long)
    terrain.env_origins = torch.zeros(num_envs, 3)
    self.scene = type("Scene", (), {"terrain": terrain})()


def test_wmp_play_terrain_event_covers_types_and_rotates_single_env():
  env = _FakePlayEnv(num_envs=5, num_rows=3, num_cols=5)

  mdp.randomize_play_terrain(env, None)

  assert set(env.scene.terrain.terrain_types.tolist()) == set(range(5))
  assert torch.equal(
    env.scene.terrain.env_origins,
    env.scene.terrain.terrain_origins[
      env.scene.terrain.terrain_levels, env.scene.terrain.terrain_types
    ],
  )

  single_env = _FakePlayEnv(num_envs=1, num_rows=3, num_cols=5)
  seen = []
  for _ in range(5):
    mdp.randomize_play_terrain(single_env, None)
    seen.append(int(single_env.scene.terrain.terrain_types[0].item()))

  assert len(set(seen)) == 5


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
  correction = mdp.only_positive_reward_clip(env, min_reward=-0.5)
  assert torch.allclose(
    correction * env.step_dt,
    torch.tensor([0.19, 0.0, 0.99, 0.0]),
  )


class _FakeWmpDrModel:
  def __init__(self) -> None:
    self.actuator_gainprm = torch.zeros(2, 3, 10)
    self.actuator_biasprm = torch.zeros(2, 3, 10)
    self.actuator_forcerange = torch.zeros(2, 3, 2)
    self.body_mass = torch.tensor(
      [
        [0.0, 6.5],
        [0.0, 7.25],
      ],
      dtype=torch.float32,
    )
    self.body_ipos = torch.zeros(2, 2, 3)
    self.body_ipos[0, 1] = torch.tensor([0.02, -0.01, 0.03])
    self.body_ipos[1, 1] = torch.tensor([-0.04, 0.05, -0.02])
    self.geom_friction = torch.zeros(2, 4, 3)
    self.geom_friction[0, :, 0] = torch.tensor([0.3, 1.4, 1.6, 0.9])
    self.geom_friction[1, :, 0] = torch.tensor([1.1, 0.7, 1.3, 2.0])


class _FakeWmpDrSim:
  def __init__(self) -> None:
    self.model = _FakeWmpDrModel()
    self._defaults = {
      "actuator_gainprm": torch.zeros(3, 10),
      "actuator_biasprm": torch.zeros(3, 10),
      "actuator_forcerange": torch.zeros(3, 2),
      "body_mass": torch.tensor([0.0, 5.0]),
      "body_ipos": torch.zeros(2, 3),
    }
    self._defaults["actuator_gainprm"][:, 0] = torch.tensor([40.0, 50.0, 60.0])
    self._defaults["actuator_biasprm"][:, 1] = -torch.tensor([40.0, 50.0, 60.0])
    self._defaults["actuator_biasprm"][:, 2] = -torch.tensor([1.0, 2.0, 3.0])
    self._defaults["actuator_forcerange"][:, 1] = torch.tensor([20.0, 30.0, 40.0])
    self.model.actuator_gainprm[:, :, 0] = torch.tensor(
      [
        [44.0, 45.0, 72.0],
        [36.0, 60.0, 54.0],
      ]
    )
    self.model.actuator_biasprm[:, :, 2] = -torch.tensor(
      [
        [1.2, 1.6, 3.3],
        [0.8, 2.4, 2.7],
      ]
    )
    self.model.actuator_forcerange[:, :, 1] = torch.tensor(
      [
        [18.0, 33.0, 44.0],
        [22.0, 27.0, 36.0],
      ]
    )

  def get_default_field(self, field: str) -> torch.Tensor:
    return self._defaults[field]


class _FakeWmpDrIndexing:
  def __init__(self) -> None:
    self.body_ids = torch.tensor([0, 1])
    self.geom_ids = torch.tensor([0, 1, 2, 3])


class _FakeWmpDrRobot:
  def __init__(self) -> None:
    self.num_joints = 3
    self.body_names = ("world", "trunk")
    self.geom_names = (
      "FR_foot_collision",
      "FL_foot_collision",
      "RR_foot_collision",
      "RL_foot_collision",
    )
    self.indexing = _FakeWmpDrIndexing()
    actuator = object.__new__(BuiltinPositionActuator)
    actuator._target_ids = torch.tensor([0, 1, 2])
    actuator._global_ctrl_ids = torch.tensor([0, 1, 2])
    self.actuators = [actuator]


class _FakeWmpDrScene:
  def __init__(self) -> None:
    self._robot = _FakeWmpDrRobot()

  def __getitem__(self, name: str):
    assert name == "robot"
    return self._robot


class _FakeWmpDrEnv:
  def __init__(self) -> None:
    self.num_envs = 2
    self.device = "cpu"
    self.sim = _FakeWmpDrSim()
    self.scene = _FakeWmpDrScene()


def test_wmp_privileged_randomization_cache_reads_actual_dr_fields():
  env = _FakeWmpDrEnv()

  mdp.cache_privileged_randomization(
    env,
    torch.tensor([0, 1]),
    asset_cfg=SceneEntityCfg("robot", joint_ids=[0, 1, 2]),
    foot_geom_names=(
      "FR_foot_collision",
      "FL_foot_collision",
      "RR_foot_collision",
      "RL_foot_collision",
    ),
  )

  cache = env._wmp_privileged_randomization
  assert torch.allclose(
    cache["p_gains"],
    torch.tensor(
      [
        [0.1, -0.1, 0.2],
        [-0.1, 0.2, -0.1],
      ]
    ),
  )
  assert torch.allclose(
    cache["d_gains"],
    torch.tensor(
      [
        [0.2, -0.2, 0.1],
        [-0.2, 0.2, -0.1],
      ]
    ),
  )
  assert torch.allclose(
    cache["motor_strength"],
    torch.tensor(
      [
        [0.9, 1.1, 1.1],
        [1.1, 0.9, 0.9],
      ]
    ),
  )
  assert torch.allclose(cache["base_mass"], torch.tensor([[1.5], [2.25]]))
  assert torch.allclose(
    cache["base_com"],
    torch.tensor([[0.02, -0.01, 0.03], [-0.04, 0.05, -0.02]]),
  )
  assert torch.allclose(cache["friction"], torch.tensor([[1.05], [1.275]]))
  assert torch.allclose(cache["restitution"], torch.zeros(2, 1))


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
  expected_vel_first = frames[0, 37:49] * 2.0
  expected_vel_second = frames[1, 37:49] * 2.0
  assert torch.allclose(loader.transitions[0, :12], expected_first)
  assert torch.allclose(loader.transitions[0, 18:30], expected_vel_first)
  assert torch.allclose(loader.transitions[0, 30:42], expected_second)
  assert torch.allclose(loader.transitions[0, 48:60], expected_vel_second)

  stats = loader.joint_position_stats()
  assert torch.allclose(stats["min"], expected_first)
  assert torch.allclose(stats["max"], expected_second)
  assert torch.allclose(stats["mean"], (expected_first + expected_second) * 0.5)


def test_wmp_amp_discriminator_reward_matches_original_quadratic_formula():
  disc = AMPDiscriminator(amp_obs_dim=30, hidden_dims=(16,))
  transitions = torch.zeros(4, 60)
  amp_obs = transitions[:, :30]
  next_amp_obs = transitions[:, 30:]

  class _ConstantNet(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
      return torch.ones(x.shape[0], 1, device=x.device) * 1.0

  disc.net = _ConstantNet()
  reward = disc.reward(amp_obs, next_amp_obs, reward_scale=0.01)

  assert torch.allclose(reward, torch.full((4,), 0.01))


def test_wmp_actor_critic_defaults_match_original_wmp_dims():
  actor_critic = ActorCriticWMP(
    actor_dim=45,
    critic_dim=286,
    action_dim=12,
    history_dim=42,
    history_length=5,
    hidden_dims=(512, 256, 128),
    encoder_hidden_dims=(256, 128),
    wm_encoder_hidden_dims=(64, 64),
    actor_hidden_dims=(256, 128, 64),
    critic_hidden_dims=(512, 256, 128),
    activation="elu",
    history_latent_dim=35,
    wm_feature_dim=512,
    wm_latent_dim=32,
    command_dim=3,
    init_std=1.0,
  )
  history = torch.zeros(2, 5, 42)
  actor = torch.zeros(2, 45)
  critic = torch.zeros(2, 286)
  wm_feature = torch.zeros(2, 512)

  mean, std = actor_critic.distribution_stats(actor, history, wm_feature)
  value = actor_critic.value(critic, wm_feature)

  assert actor_critic.history_encoder[-1].out_features == 35
  assert actor_critic.wm_feature_encoder[-1].out_features == 32
  assert actor_critic.critic_wm_feature_encoder[-1].out_features == 32
  assert mean.shape == (2, 12)
  assert std.shape == (2, 12)
  assert value.shape == (2,)


def test_wmp_dreamer_adapter_trains_batch_and_reports_original_metrics():
  adapter = DreamerWorldModelAdapter(
    prop_dim=5,
    action_dim=4,
    depth_shape=(8, 8, 1),
    device=torch.device("cpu"),
    use_camera=True,
    config_overrides={
      "dyn_hidden": 16,
      "dyn_deter": 16,
      "dyn_stoch": 4,
      "dyn_discrete": 4,
      "units": 16,
      "encoder": {
        "mlp_keys": ".*",
        "cnn_keys": "image",
        "act": "SiLU",
        "norm": True,
        "cnn_depth": 2,
        "kernel_size": 4,
        "minres": 4,
        "mlp_layers": 1,
        "mlp_units": 16,
        "symlog_inputs": True,
      },
      "decoder": {
        "mlp_keys": ".*",
        "cnn_keys": "image",
        "act": "SiLU",
        "norm": True,
        "cnn_depth": 2,
        "kernel_size": 4,
        "minres": 4,
        "mlp_layers": 1,
        "mlp_units": 16,
        "cnn_sigmoid": False,
        "image_dist": "mse",
        "vector_dist": "symlog_mse",
        "outscale": 1.0,
      },
      "reward_head": {
        "layers": 1,
        "dist": "symlog_disc",
        "loss_scale": 0.0,
        "outscale": 0.0,
      },
      "model_lr": 1.0e-4,
      "grad_clip": 100.0,
    },
  )
  batch = {
    "prop": torch.zeros(2, 3, 5).numpy(),
    "image": torch.zeros(2, 3, 8, 8, 1).numpy(),
    "action": torch.zeros(2, 3, 4).numpy(),
    "reward": torch.zeros(2, 3).numpy(),
    "is_first": torch.zeros(2, 3).numpy(),
  }
  batch["is_first"][:, 0] = 1.0

  feature = adapter.features(
    torch.zeros(2, 5),
    torch.zeros(2, 8, 8, 1),
    torch.zeros(2, 4),
  )
  metrics = adapter.train_world_model(batch, train_steps=1)

  assert feature.shape == (2, 16)
  assert "kl" in metrics
  assert "dyn_loss" in metrics
  assert "rep_loss" in metrics
  assert "image_loss" in metrics
  assert "reward_loss" in metrics
  assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_wmp_dreamer_adapter_ignores_checkpoint_runtime_state_batch():
  adapter = DreamerWorldModelAdapter(
    prop_dim=5,
    action_dim=4,
    depth_shape=(8, 8, 1),
    device=torch.device("cpu"),
    use_camera=True,
    config_overrides={
      "dyn_hidden": 16,
      "dyn_deter": 16,
      "dyn_stoch": 4,
      "dyn_discrete": 4,
      "units": 16,
      "encoder": {
        "mlp_keys": ".*",
        "cnn_keys": "image",
        "act": "SiLU",
        "norm": True,
        "cnn_depth": 2,
        "kernel_size": 4,
        "minres": 4,
        "mlp_layers": 1,
        "mlp_units": 16,
        "symlog_inputs": True,
      },
      "decoder": {
        "mlp_keys": ".*",
        "cnn_keys": "image",
        "act": "SiLU",
        "norm": True,
        "cnn_depth": 2,
        "kernel_size": 4,
        "minres": 4,
        "mlp_layers": 1,
        "mlp_units": 16,
        "cnn_sigmoid": False,
        "image_dist": "mse",
        "vector_dist": "symlog_mse",
        "outscale": 1.0,
      },
      "reward_head": {
        "layers": 1,
        "dist": "symlog_disc",
        "loss_scale": 0.0,
        "outscale": 0.0,
      },
    },
  )
  stale_state = adapter.model.dynamics.initial(2048)
  state = adapter.state_dict()
  state["state"] = stale_state

  loaded = DreamerWorldModelAdapter(
    prop_dim=5,
    action_dim=4,
    depth_shape=(8, 8, 1),
    device=torch.device("cpu"),
    use_camera=True,
    config_overrides={
      "dyn_hidden": 16,
      "dyn_deter": 16,
      "dyn_stoch": 4,
      "dyn_discrete": 4,
      "units": 16,
      "encoder": {
        "mlp_keys": ".*",
        "cnn_keys": "image",
        "act": "SiLU",
        "norm": True,
        "cnn_depth": 2,
        "kernel_size": 4,
        "minres": 4,
        "mlp_layers": 1,
        "mlp_units": 16,
        "symlog_inputs": True,
      },
      "decoder": {
        "mlp_keys": ".*",
        "cnn_keys": "image",
        "act": "SiLU",
        "norm": True,
        "cnn_depth": 2,
        "kernel_size": 4,
        "minres": 4,
        "mlp_layers": 1,
        "mlp_units": 16,
        "cnn_sigmoid": False,
        "image_dist": "mse",
        "vector_dist": "symlog_mse",
        "outscale": 1.0,
      },
      "reward_head": {
        "layers": 1,
        "dist": "symlog_disc",
        "loss_scale": 0.0,
        "outscale": 0.0,
      },
    },
  )
  loaded.load_state_dict(state)
  feature = loaded.features(
    torch.zeros(2, 5),
    torch.zeros(2, 8, 8, 1),
    torch.zeros(2, 4),
  )

  assert feature.shape == (2, 16)


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
    history_dim=42,
    history_length=3,
    hidden_dims=hidden_dims,
    activation="elu",
    history_latent_dim=16,
    wm_feature_dim=feature_dim,
    wm_latent_dim=8,
    command_dim=3,
    init_std=0.5,
    min_std=0.1,
    max_std=0.6,
  )
  with torch.no_grad():
    actor_critic.log_std.fill_(2.0)
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
  history = torch.randn(batch, 3, 42)
  prop = torch.randn(batch, prop_dim)
  next_prop = torch.randn(batch, prop_dim)
  actions = torch.randn(batch, action_dim)
  rewards = torch.randn(batch)
  dones = torch.zeros(batch)
  depth = torch.randn(batch, depth_dim)
  height = torch.randn(batch, height_dim)
  wm_feature = world_model.features(prop)
  assert torch.allclose(
    torch.exp(actor_critic._bounded_log_std()),
    torch.full((action_dim,), 0.6),
  )

  sampled_action, log_prob, value = actor_critic.act(
    actor,
    critic,
    history,
    wm_feature,
  )
  assert sampled_action.shape == (batch, action_dim)
  assert log_prob.shape == (batch,)
  assert value.shape == (batch,)
  assert actor_critic.predict_linear_velocity(history).shape == (batch, 3)

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
      "history_exclude_command": True,
      "clip_observations": 100.0,
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
      "schedule": "adaptive",
      "desired_kl": 0.01,
      "gamma": 0.99,
      "lam": 0.95,
      "clip_param": 0.2,
      "entropy_coef": 0.0,
      "value_loss_coef": 1.0,
      "use_clipped_value_loss": True,
      "vel_predict_coef": 1.0,
      "max_grad_norm": 1.0,
      "amp_task_reward_lerp": 0.3,
    },
    "world_model": {
      "hidden_dims": (16,),
      "feature_dim": 16,
      "dyn_deter": 16,
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
      "reward_scale": 0.01,
      "normalize_input": True,
      "expert_motion_files": expert_motion_files,
    },
  }


class _FakeWriter:
  def __init__(self) -> None:
    self.scalars: dict[str, list[tuple[int, float]]] = {}

  def add_scalar(self, tag: str, value, step: int) -> None:
    self.scalars.setdefault(tag, []).append((step, float(value)))


class _FakeRunnerRewardManager:
  active_terms = ("tracking_lin_vel", "action_rate")

  def __init__(self) -> None:
    self._step_reward = torch.zeros(2, 2)


def test_wmp_runner_logs_action_std_action_stats_and_reward_terms():
  env = _FakeWmpEnv()
  env.reward_manager = _FakeRunnerRewardManager()
  runner = WMPRunner(
    env,
    _tiny_runner_cfg(()),
    device="cpu",
  )
  assert runner.history_dim == 42
  writer = _FakeWriter()
  runner._writer = writer
  actions = torch.arange(2 * env.num_envs * env.num_actions, dtype=torch.float32)
  actions = actions.reshape(2, env.num_envs, env.num_actions)
  reward_terms = torch.tensor(
    [
      [[1.0, -0.5], [2.0, -1.5]],
      [[3.0, -2.5], [4.0, -3.5]],
    ],
    dtype=torch.float32,
  )
  rollout = {
    "task_rewards": torch.ones(2, env.num_envs),
    "rewards": torch.ones(2, env.num_envs) * 1.5,
    "depth_real_fraction": torch.tensor(0.5),
    "actions": actions,
    "reward_terms": reward_terms,
  }

  runner._log(
    iteration=1,
    final_iteration=1,
    rollout=rollout,
    ppo={"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0},
    world={},
    depth={},
    amp={},
    timing={
      "total_steps": 4,
      "collection_time": 0.0,
      "learning_time": 0.0,
      "iteration_time": 1.0,
      "elapsed_time": 1.0,
    },
  )

  assert "PPO/action_std" in writer.scalars
  assert "Action/max_abs_mean" in writer.scalars
  assert "Action/joint_00_mean" in writer.scalars
  assert "Action/joint_00_std" in writer.scalars
  assert "Action/joint_00_abs_mean" in writer.scalars
  assert "RewardTerms/tracking_lin_vel" in writer.scalars
  assert "RewardTerms/action_rate" in writer.scalars
  assert writer.scalars["RewardTerms/tracking_lin_vel"][0] == (1, 2.5)
  assert writer.scalars["RewardTerms/action_rate"][0] == (1, -2.0)


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
  assert runner.world_model.config.dyn_stoch == 32

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
