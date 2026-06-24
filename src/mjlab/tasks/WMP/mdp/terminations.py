from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.episode_length_buf >= env.max_episode_length


def bad_orientation(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.acos(torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)) > (
    limit_angle
  )


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def out_of_terrain_bounds(
  env: ManagerBasedRlEnv,
  margin: float = 0.3,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None or terrain.terrain_origins is None:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  asset: Entity = env.scene[asset_cfg.name]
  root_xy = asset.data.root_link_pos_w[:, :2]
  num_rows, num_cols = terrain.terrain_origins.shape[:2]
  half_x = 0.5 * num_rows * terrain_generator.size[0] + terrain_generator.border_width
  half_y = 0.5 * num_cols * terrain_generator.size[1] + terrain_generator.border_width
  return (root_xy[:, 0].abs() > half_x - margin) | (
    root_xy[:, 1].abs() > half_y - margin
  )
