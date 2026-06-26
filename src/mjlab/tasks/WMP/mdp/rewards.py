from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")
_WMP_EDGE_MASK_KEY = "wmp_x_edge_mask"


def _asset(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> Entity:
  return env.scene[asset_cfg.name]


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.15,
  lin_vel_clip: float = 0.1,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_lin_vel_b[:, :2]
  target = command[:, :2]
  upper = torch.where(
    target < 0.0, torch.full_like(target, 1.0e5), target + lin_vel_clip
  )
  lower = torch.where(
    target > 0.0, torch.full_like(target, -1.0e5), target - lin_vel_clip
  )
  clipped_actual = torch.clip(actual, lower, upper)
  error = torch.sum(torch.square(target - clipped_actual), dim=1)
  return torch.exp(-error / std)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.15,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-error / std)


def lin_vel_z_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  return torch.square(_asset(env, asset_cfg).data.root_link_lin_vel_b[:, 2])


def upright(
  env: ManagerBasedRlEnv,
  std: float = 0.2,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  gravity_b = quat_apply_inverse(asset.data.root_link_quat_w, asset.data.gravity_vec_w)
  return torch.exp(-torch.sum(torch.square(gravity_b[:, :2]), dim=1) / std)


def base_height_l2(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  sensor = env.scene[height_sensor_name]
  assert isinstance(sensor, TerrainHeightSensor)
  ground_height = asset.data.root_link_pos_w[:, 2:3] - sensor.data.heights
  base_height = asset.data.root_link_pos_w[:, 2] - torch.mean(ground_height, dim=1)
  return torch.square(base_height - target_height)


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )


def dof_acc_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  prev_vel = getattr(env, "_wmp_prev_joint_vel", None)
  if prev_vel is None or prev_vel.shape != asset.data.joint_vel.shape:
    prev_vel = torch.zeros_like(asset.data.joint_vel)
  acc = (asset.data.joint_vel - prev_vel) / env.step_dt
  env._wmp_prev_joint_vel = asset.data.joint_vel.detach().clone()
  return torch.sum(torch.square(acc), dim=1)


def dof_error_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  return torch.sum(torch.square(asset.data.joint_pos - default_joint_pos), dim=1)


def torques_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  return torch.sum(
    torch.square(asset.data.qfrc_actuator[:, asset_cfg.joint_ids]), dim=1
  )


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  threshold: float = 0.5,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  air_time = sensor.data.current_air_time
  assert air_time is not None
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  reward = torch.sum((air_time - threshold) * first_contact.float(), dim=1)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  reward *= (torch.norm(command[:, :2], dim=1) > 0.1).float()
  return reward


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  assert sensor.data.found is not None
  command = env.command_manager.get_command(command_name)
  assert command is not None
  in_contact = (sensor.data.found > 0).float()
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  cost = torch.sum(torch.square(torch.norm(foot_vel_xy, dim=-1)) * in_contact, dim=1)
  return cost * (torch.norm(command[:, :2], dim=1) > 0.05).float()


def _terrain_type_ids(
  env: ManagerBasedRlEnv, prefixes: tuple[str, ...]
) -> torch.Tensor:
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_generator is None:
    return torch.empty(0, dtype=torch.long, device=env.device)
  ids = [
    index
    for index, name in enumerate(terrain.cfg.terrain_generator.sub_terrains)
    if name.startswith(prefixes)
  ]
  return torch.tensor(ids, dtype=torch.long, device=env.device)


def _terrain_mask(
  env: ManagerBasedRlEnv,
  *,
  prefixes: tuple[str, ...],
  min_level: int | None = None,
) -> torch.Tensor:
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  type_ids = _terrain_type_ids(env, prefixes)
  if type_ids.numel() == 0:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  mask = torch.isin(terrain.terrain_types, type_ids)
  if min_level is not None:
    mask &= terrain.terrain_levels > min_level
  return mask


def feet_stumble(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_terrain_level: int = 3,
  terrain_prefixes: tuple[str, ...] = ("gap_", "pit_climb_"),
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  assert sensor.data.force is not None
  lateral = torch.norm(sensor.data.force[..., :2], dim=-1)
  vertical = torch.abs(sensor.data.force[..., 2])
  if sensor.data.found is not None:
    in_contact = sensor.data.found > 0
    if in_contact.shape != lateral.shape:
      in_contact = in_contact.view(lateral.shape)
    stumble_mask = (lateral > 4.0 * vertical) & in_contact
  else:
    stumble_mask = lateral > 4.0 * vertical
  reward = torch.any(stumble_mask, dim=1).float()
  terrain_mask = _terrain_mask(
    env, prefixes=terrain_prefixes, min_level=min_terrain_level
  )
  return reward * terrain_mask.float()


def stumble(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  return feet_stumble(env, sensor_name)


def _contact_cost(sensor: ContactSensor, force_threshold: float) -> torch.Tensor:
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    return (force_mag > force_threshold).any(dim=-1).sum(dim=1).float()
  assert data.found is not None
  return (data.found > 0).float().sum(dim=-1)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  return _contact_cost(sensor, force_threshold)


def collision_cost(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
  force_threshold: float = 0.1,
) -> torch.Tensor:
  cost = torch.zeros(env.num_envs, device=env.device)
  for sensor_name in sensor_names:
    sensor = env.scene[sensor_name]
    assert isinstance(sensor, ContactSensor)
    cost += _contact_cost(sensor, force_threshold)
  return cost


def stuck(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  velocity = torch.abs(_asset(env).data.root_link_lin_vel_b[:, 0])
  return ((velocity < 0.1) & (torch.abs(command[:, 0]) > 0.1)).float()


def cheat(
  env: ManagerBasedRlEnv,
  heading_limit: float = 1.0,
  terrain_prefixes: tuple[str, ...] = ("rough_flat",),
) -> torch.Tensor:
  asset = _asset(env)
  non_flat = ~_terrain_mask(env, prefixes=terrain_prefixes)
  return (torch.abs(asset.data.heading_w) > heading_limit).float() * non_flat.float()


def wmp_feet_edge_coef(
  iteration: int,
  *,
  start_iteration: int = 4000,
  end_iteration: int = 10000,
  start_value: float = 0.1,
  end_value: float = 1.0,
) -> float:
  if iteration <= start_iteration:
    return start_value
  if iteration >= end_iteration:
    return end_value
  progress = (iteration - start_iteration) / (end_iteration - start_iteration)
  return start_value + progress * (end_value - start_value)


def feet_edge(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
  min_terrain_level: int = 3,
  terrain_prefixes: tuple[str, ...] = ("gap_", "pit_climb_"),
) -> torch.Tensor:
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return torch.zeros(env.num_envs, device=env.device)
  edge_mask = getattr(terrain, "metadata", {}).get(_WMP_EDGE_MASK_KEY)
  if edge_mask is None:
    return torch.zeros(env.num_envs, device=env.device)

  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  assert sensor.data.found is not None
  asset = _asset(env, asset_cfg)

  in_contact = sensor.data.found > 0
  if in_contact.shape[1] != len(asset_cfg.site_ids):
    in_contact = in_contact.view(in_contact.shape[0], len(asset_cfg.site_ids), -1).any(
      dim=-1
    )
  terrain_mask = _terrain_mask(
    env, prefixes=terrain_prefixes, min_level=min_terrain_level
  )

  foot_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]
  origins = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
  size_x, size_y = terrain.cfg.terrain_generator.size
  local_x = foot_xy[..., 0] - (origins[:, None, 0] - size_x * 0.5)
  local_y = foot_xy[..., 1] - (origins[:, None, 1] - size_y * 0.5)
  nx = edge_mask.shape[-2]
  ny = edge_mask.shape[-1]
  ix = torch.floor(local_x / size_x * nx).to(torch.long)
  iy = torch.floor(local_y / size_y * ny).to(torch.long)
  inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
  ix = ix.clamp(0, nx - 1)
  iy = iy.clamp(0, ny - 1)

  env_edge_mask = edge_mask[terrain.terrain_levels, terrain.terrain_types]
  env_ids = torch.arange(env.num_envs, device=env.device).unsqueeze(1)
  on_edge = env_edge_mask[env_ids, ix, iy].bool() & inside
  reward = (on_edge & in_contact.bool()).float().sum(dim=1)
  iteration = int(getattr(env, "_wmp_iteration", 0))
  coef = wmp_feet_edge_coef(iteration)
  env._wmp_feet_edge_coef = coef
  return reward * terrain_mask.float() * coef


def only_positive_reward_clip(
  env: ManagerBasedRlEnv,
  min_reward: float = 0.0,
) -> torch.Tensor:
  manager = env.reward_manager
  dt = env.step_dt if getattr(manager, "_scale_by_dt", True) else 1.0
  previous = manager._reward_buf
  min_value = float(min_reward) * dt
  correction = torch.clamp(previous, min=min_value) - previous
  return correction / dt


class resettable_dof_acc_l2:
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self._prev = torch.zeros_like(_asset(env).data.joint_vel)

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    asset = _asset(env)
    acc = (asset.data.joint_vel - self._prev) / env.step_dt
    self._prev = asset.data.joint_vel.detach().clone()
    return torch.sum(torch.square(acc), dim=1)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._prev[env_ids] = 0.0
