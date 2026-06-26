from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _as_env_id_tensor(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None
) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if isinstance(env_ids, slice):
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
  return env_ids.to(device=env.device, dtype=torch.long)


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
