from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def mean_action_acc(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mean absolute action delta for each environment."""
  return torch.mean(
    torch.abs(env.action_manager.action - env.action_manager.prev_action),
    dim=1,
  )
