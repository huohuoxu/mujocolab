from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.actuator import BuiltinPositionActuator, IdealPdActuator
from mjlab.actuator.xml_actuator import XmlActuator
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def _as_env_id_tensor(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None
) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if isinstance(env_ids, slice):
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
  return env_ids.to(device=env.device, dtype=torch.long)


def _priv_cache(env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
  cache = getattr(env, "_wmp_privileged_randomization", None)
  if cache is None:
    cache = {}
    setattr(env, "_wmp_privileged_randomization", cache)
  return cache


def _selected_joint_ids(asset: Entity, asset_cfg: SceneEntityCfg) -> list[int]:
  if isinstance(asset_cfg.joint_ids, slice):
    return list(range(asset.num_joints))[asset_cfg.joint_ids]
  if torch.is_tensor(asset_cfg.joint_ids):
    return [int(idx) for idx in asset_cfg.joint_ids.detach().cpu().tolist()]
  return [int(idx) for idx in asset_cfg.joint_ids]


def _safe_ratio(
  current: torch.Tensor,
  default: torch.Tensor,
  fallback: float,
) -> torch.Tensor:
  default = default.to(device=current.device, dtype=current.dtype)
  valid = torch.isfinite(current) & torch.isfinite(default) & (default.abs() > 1.0e-8)
  return torch.where(valid, current / default, torch.full_like(current, fallback))


def _read_pd_gain_deltas(
  env: ManagerBasedRlEnv,
  ids: torch.Tensor,
  asset: Entity,
  joint_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
  """Read actual PD gain deltas in original WMP critic-observation semantics."""
  joint_to_col = {joint_id: col for col, joint_id in enumerate(joint_ids)}
  p_deltas = torch.zeros(len(ids), len(joint_ids), device=env.device)
  d_deltas = torch.zeros_like(p_deltas)
  default_gainprm = env.sim.get_default_field("actuator_gainprm")
  default_biasprm = env.sim.get_default_field("actuator_biasprm")

  for actuator in asset.actuators:
    target_ids = [int(idx) for idx in actuator.target_ids.detach().cpu().tolist()]
    selected = [
      (local_id, joint_to_col[joint_id])
      for local_id, joint_id in enumerate(target_ids)
      if joint_id in joint_to_col
    ]
    if not selected:
      continue
    local_ids = torch.tensor(
      [pair[0] for pair in selected],
      device=env.device,
      dtype=torch.long,
    )
    out_cols = torch.tensor(
      [pair[1] for pair in selected],
      device=env.device,
      dtype=torch.long,
    )

    if isinstance(actuator, IdealPdActuator):
      assert actuator.stiffness is not None
      assert actuator.default_stiffness is not None
      assert actuator.damping is not None
      assert actuator.default_damping is not None
      current_p = actuator.stiffness[ids[:, None], local_ids]
      default_p = actuator.default_stiffness[ids[:, None], local_ids]
      current_d = actuator.damping[ids[:, None], local_ids]
      default_d = actuator.default_damping[ids[:, None], local_ids]
    elif isinstance(actuator, (BuiltinPositionActuator, XmlActuator)):
      if actuator.command_field != "position":
        continue
      ctrl_ids = actuator.global_ctrl_ids[local_ids].to(
        device=env.device,
        dtype=torch.long,
      )
      current_p = env.sim.model.actuator_gainprm[ids[:, None], ctrl_ids, 0]
      default_p = default_gainprm[ctrl_ids, 0].unsqueeze(0)
      current_d = -env.sim.model.actuator_biasprm[ids[:, None], ctrl_ids, 2]
      default_d = -default_biasprm[ctrl_ids, 2].unsqueeze(0)
    else:
      continue

    p_deltas[:, out_cols] = _safe_ratio(current_p, default_p, 1.0) - 1.0
    d_deltas[:, out_cols] = _safe_ratio(current_d, default_d, 1.0) - 1.0

  return p_deltas, d_deltas


def _read_motor_strength(
  env: ManagerBasedRlEnv,
  ids: torch.Tensor,
  asset: Entity,
  joint_ids: list[int],
) -> torch.Tensor:
  """Read actual actuator effort-limit scale in selected joint order."""
  joint_to_col = {joint_id: col for col, joint_id in enumerate(joint_ids)}
  strength = torch.ones(len(ids), len(joint_ids), device=env.device)
  default_forcerange = env.sim.get_default_field("actuator_forcerange")

  for actuator in asset.actuators:
    target_ids = [int(idx) for idx in actuator.target_ids.detach().cpu().tolist()]
    selected = [
      (local_id, joint_to_col[joint_id])
      for local_id, joint_id in enumerate(target_ids)
      if joint_id in joint_to_col
    ]
    if not selected:
      continue
    local_ids = torch.tensor(
      [pair[0] for pair in selected],
      device=env.device,
      dtype=torch.long,
    )
    out_cols = torch.tensor(
      [pair[1] for pair in selected],
      device=env.device,
      dtype=torch.long,
    )

    if isinstance(actuator, IdealPdActuator):
      assert actuator.force_limit is not None
      assert actuator.default_force_limit is not None
      current = actuator.force_limit[ids[:, None], local_ids].abs()
      default = actuator.default_force_limit[ids[:, None], local_ids].abs()
    else:
      ctrl_ids = actuator.global_ctrl_ids[local_ids].to(
        device=env.device,
        dtype=torch.long,
      )
      current = env.sim.model.actuator_forcerange[ids[:, None], ctrl_ids, 1].abs()
      default = default_forcerange[ctrl_ids, 1].abs().unsqueeze(0)

    strength[:, out_cols] = _safe_ratio(current, default, 1.0)

  return strength


def reset_joints_by_scale(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_scale_range: tuple[float, float] = (0.5, 1.5),
  velocity_range: tuple[float, float] = (0.0, 0.0),
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  """Reset joints as default_joint_pos * uniform(scale), matching original WMP."""
  ids = _as_env_id_tensor(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  default_joint_vel = asset.data.default_joint_vel
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert default_joint_pos is not None
  assert default_joint_vel is not None
  assert soft_joint_pos_limits is not None

  joint_pos = default_joint_pos[ids][:, asset_cfg.joint_ids].clone()
  scales = sample_uniform(
    position_scale_range[0],
    position_scale_range[1],
    joint_pos.shape,
    env.device,
  )
  joint_pos = joint_pos * scales
  joint_pos_limits = soft_joint_pos_limits[ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  joint_vel = default_joint_vel[ids][:, asset_cfg.joint_ids].clone()
  joint_vel += sample_uniform(
    velocity_range[0],
    velocity_range[1],
    joint_vel.shape,
    env.device,
  )

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(
    joint_pos.view(len(ids), -1),
    joint_vel.view(len(ids), -1),
    env_ids=ids,
    joint_ids=joint_ids,
  )


def cache_privileged_randomization(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  *,
  asset_cfg: SceneEntityCfg = _ROBOT,
  foot_geom_names: tuple[str, ...] = (),
) -> None:
  """Cache actual DR values consumed by WMP critic obs after DR events run."""
  ids = _as_env_id_tensor(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  cache = _priv_cache(env)
  joint_ids = _selected_joint_ids(asset, asset_cfg)
  action_dim = len(joint_ids)

  if "p_gains" not in cache:
    cache["p_gains"] = torch.zeros(env.num_envs, action_dim, device=env.device)
    cache["d_gains"] = torch.zeros(env.num_envs, action_dim, device=env.device)
    cache["motor_strength"] = torch.ones(env.num_envs, action_dim, device=env.device)
    cache["restitution"] = torch.zeros(env.num_envs, 1, device=env.device)
    cache["friction"] = torch.ones(env.num_envs, 1, device=env.device)
    cache["base_mass"] = torch.zeros(env.num_envs, 1, device=env.device)
    cache["base_com"] = torch.zeros(env.num_envs, 3, device=env.device)

  cache["p_gains"][ids], cache["d_gains"][ids] = _read_pd_gain_deltas(
    env,
    ids,
    asset,
    joint_ids,
  )
  cache["motor_strength"][ids] = _read_motor_strength(env, ids, asset, joint_ids)
  # There is currently no MuJoCo DR event that writes a physical restitution field
  # for WMP. Keep the privileged value equal to the actual configured default.
  cache["restitution"][ids] = 0.0

  try:
    trunk_local_id = asset.body_names.index("trunk")
  except ValueError:
    trunk_local_id = 0
  trunk_id = int(asset.indexing.body_ids[trunk_local_id].item())
  default_mass = env.sim.get_default_field("body_mass")
  current_mass = env.sim.model.body_mass[ids, trunk_id].reshape(-1, 1)
  cache["base_mass"][ids] = current_mass - default_mass[trunk_id]

  default_ipos = env.sim.get_default_field("body_ipos")
  current_ipos = env.sim.model.body_ipos[ids, trunk_id]
  cache["base_com"][ids] = current_ipos - default_ipos[trunk_id]

  if foot_geom_names:
    local_geom_ids = [
      idx for idx, name in enumerate(asset.geom_names) if name in foot_geom_names
    ]
    geom_ids = [
      int(asset.indexing.geom_ids[idx].item())
      for idx in local_geom_ids
    ]
    if geom_ids:
      geom_ids = torch.tensor(geom_ids, device=env.device, dtype=torch.long)
      friction = env.sim.model.geom_friction[ids[:, None], geom_ids, 0].mean(
        dim=1,
        keepdim=True,
      )
      cache["friction"][ids] = friction


def randomize_play_terrain(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  *,
  cover_terrain_types: bool = True,
) -> None:
  """Assign play resets to varied terrain patches before the base pose reset.

  Training keeps the normal WMP terrain curriculum. Play/eval needs a different
  behavior: with multiple envs, spread robots across terrain columns so the
  viewer can inspect every terrain type; with one env, rotate the terrain column
  across resets so repeated resets do not stay on one patch forever.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.terrain_origins is None:
    return
  terrain_origins = terrain.terrain_origins
  num_rows, num_cols = terrain_origins.shape[:2]
  if num_rows <= 0 or num_cols <= 0:
    return

  ids = _as_env_id_tensor(env, env_ids)
  num_reset = len(ids)
  if num_reset == 0:
    return

  device = terrain_origins.device
  levels = torch.randint(0, num_rows, (num_reset,), device=device)

  if cover_terrain_types:
    if num_reset >= num_cols:
      repeats = (num_reset + num_cols - 1) // num_cols
      terrain_types = torch.arange(num_cols, device=device).repeat(repeats)[:num_reset]
      terrain_types = terrain_types[torch.randperm(num_reset, device=device)]
    else:
      offset = getattr(env, "_wmp_play_terrain_offset", None)
      if offset is None:
        offset = int(torch.randint(0, num_cols, (1,), device=device).item())
      terrain_types = (torch.arange(num_reset, device=device) + offset) % num_cols
      setattr(env, "_wmp_play_terrain_offset", (offset + num_reset) % num_cols)
  else:
    terrain_types = torch.randint(0, num_cols, (num_reset,), device=device)

  terrain.terrain_levels[ids] = levels.to(terrain.terrain_levels.device)
  terrain.terrain_types[ids] = terrain_types.to(terrain.terrain_types.device)
  terrain.env_origins[ids] = terrain_origins[levels, terrain_types]
