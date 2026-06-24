from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.WMP.mdp.commands import WmpVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


class VelocityStage(TypedDict, total=False):
  step: int
  lin_vel_x: tuple[float, float]
  lin_vel_y: tuple[float, float]
  ang_vel_z: tuple[float, float]


def terrain_levels_wmp(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> dict[str, torch.Tensor]:
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    z = torch.tensor(0.0, device=env.device)
    return {"mean": z, "max": z}
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )
  move_up = distance > terrain_generator.size[0] / 2
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down &= ~move_up
  terrain.update_env_origins(env_ids, move_up, move_down)
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }
  names = list(terrain_generator.sub_terrains.keys())
  if terrain.terrain_origins.shape[1] == len(names):
    for idx, name in enumerate(names):
      mask = terrain.terrain_types == idx
      if mask.any():
        result[name] = torch.mean(levels[mask])
  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids
  command_term = env.command_manager.get_term(command_name)
  cfg = cast(WmpVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter < stage["step"]:
      continue
    if "lin_vel_x" in stage:
      cfg.ranges.lin_vel_x = stage["lin_vel_x"]
    if "lin_vel_y" in stage:
      cfg.ranges.lin_vel_y = stage["lin_vel_y"]
    if "ang_vel_z" in stage:
      cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0], device=env.device),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1], device=env.device),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0], device=env.device),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1], device=env.device),
  }
