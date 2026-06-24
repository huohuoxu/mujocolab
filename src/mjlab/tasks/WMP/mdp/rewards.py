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


def _asset(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> Entity:
  return env.scene[asset_cfg.name]


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.15,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_lin_vel_b[:, :2]
  target = command[:, :2]
  error = torch.sum(torch.square(target - actual), dim=1)
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


def stumble(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  assert sensor.data.force is not None
  lateral = torch.norm(sensor.data.force[..., :2], dim=-1)
  vertical = torch.abs(sensor.data.force[..., 2])
  return torch.any(lateral > 4.0 * vertical, dim=1).float()


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    return (force_mag > force_threshold).any(dim=1).sum(dim=-1).float()
  assert data.found is not None
  return data.found.sum(dim=-1).float()


def feet_edge(env: ManagerBasedRlEnv) -> torch.Tensor:
  # Placeholder-compatible penalty. Detailed WMP edge masks are supplied by the
  # terrain milestone; zero keeps the term shape stable until then.
  return torch.zeros(env.num_envs, device=env.device)


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
