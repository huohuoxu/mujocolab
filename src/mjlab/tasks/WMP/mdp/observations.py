from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, CameraSensor, ContactSensor, RayCastSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def _asset(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> Entity:
  return env.scene[asset_cfg.name]


def base_lin_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  return _asset(env, asset_cfg).data.root_link_lin_vel_b


def base_ang_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  return _asset(env, asset_cfg).data.root_link_ang_vel_b


def projected_gravity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  return _asset(env, asset_cfg).data.projected_gravity_b


def joint_pos_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT,
  biased: bool = False,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  return joint_pos[:, asset_cfg.joint_ids] - default_joint_pos[:, asset_cfg.joint_ids]


def joint_vel_rel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  return (
    asset.data.joint_vel[:, asset_cfg.joint_ids]
    - default_joint_vel[:, asset_cfg.joint_ids]
  )


def last_action(env: ManagerBasedRlEnv, action_name: str | None = None) -> torch.Tensor:
  if action_name is None:
    return env.action_manager.action
  return env.action_manager.get_term(action_name).raw_action


def command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  cmd = env.command_manager.get_command(command_name)
  assert cmd is not None
  return cmd[:, :3]


def generated_commands(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return command(env, command_name)


def builtin_sensor(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return sensor.data


def height_scan(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  offset: float = 0.0,
  miss_value: float | None = None,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, RayCastSensor)
  if miss_value is None:
    miss_value = sensor.cfg.max_distance

  data = sensor.data
  frames = sensor.num_frames
  rays_per_frame = sensor.num_rays_per_frame
  batch = data.distances.shape[0]
  frame_z = data.frame_pos_w[:, :, 2:3]
  hit_z = data.hit_pos_w[..., 2].view(batch, frames, rays_per_frame)
  heights = (frame_z - hit_z - offset).view(batch, frames * rays_per_frame)
  miss_mask = data.distances < 0
  return torch.where(miss_mask, torch.full_like(heights, miss_value), heights)


def foot_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, TerrainHeightSensor)
  return sensor.data.heights


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  assert sensor.data.found is not None
  return (sensor.data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  assert sensor.data.force is not None
  forces = sensor.data.force.flatten(start_dim=1)
  return torch.sign(forces) * torch.log1p(torch.abs(forces))


def foot_contact_forces_linear(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  scale: float = 0.005,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  assert sensor.data.force is not None
  return sensor.data.force.flatten(start_dim=1) * scale


def privileged_randomization(
  env: ManagerBasedRlEnv,
  key: str,
  *,
  dim: int,
  scale: float = 1.0,
) -> torch.Tensor:
  cache = getattr(env, "_wmp_privileged_randomization", {})
  value = cache.get(key)
  if value is None:
    return torch.zeros(env.num_envs, dim, device=env.device)
  value = value.to(device=env.device, dtype=torch.float32)
  if value.shape[-1] != dim:
    value = value.reshape(env.num_envs, -1)
    if value.shape[-1] < dim:
      pad = torch.zeros(env.num_envs, dim - value.shape[-1], device=env.device)
      value = torch.cat((value, pad), dim=-1)
    else:
      value = value[:, :dim]
  return value * scale


def depth_image(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  near_clip: float = 0.0,
  far_clip: float = 2.0,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, CameraSensor)
  depth = sensor.data.depth
  if depth is None:
    raise RuntimeError(f"Camera sensor {sensor_name!r} has no depth output.")
  depth = torch.nan_to_num(depth, nan=far_clip, posinf=far_clip, neginf=far_clip)
  depth = torch.where(depth <= near_clip, torch.full_like(depth, far_clip), depth)
  depth = torch.clamp(depth, near_clip, far_clip)
  return (depth - near_clip) / (far_clip - near_clip) - 0.5


def wm_prop(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return torch.cat(
    (
      base_ang_vel(env) * 0.25,
      projected_gravity(env),
      command(env, command_name),
      joint_pos_rel(env),
      joint_vel_rel(env) * 0.05,
      last_action(env),
    ),
    dim=-1,
  )


def forward_height_map(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  base_height: float = 0.3,
  scale: float = 5.0,
) -> torch.Tensor:
  heights = height_scan(env, sensor_name)
  root_z = _asset(env).data.root_link_pos_w[:, 2:3]
  return torch.clip(root_z - base_height - heights, -1.0, 1.0) * scale


def amp_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  asset = _asset(env)
  return torch.cat(
    (
      asset.data.joint_pos,
      asset.data.root_link_lin_vel_b,
      asset.data.root_link_ang_vel_b,
      asset.data.joint_vel,
    ),
    dim=-1,
  )
