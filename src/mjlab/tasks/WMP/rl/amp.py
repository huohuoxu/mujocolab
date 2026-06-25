from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mjlab.tasks.WMP.rl.modules import build_mlp


class RunningNormalizer(nn.Module):
  """Running mean/std normalizer for AMP observations."""

  def __init__(self, obs_dim: int, eps: float = 1.0e-5) -> None:
    super().__init__()
    self.obs_dim = obs_dim
    self.eps = eps
    self.register_buffer("mean", torch.zeros(obs_dim))
    self.register_buffer("var", torch.ones(obs_dim))
    self.register_buffer("count", torch.tensor(eps))

  @torch.no_grad()
  def update(self, samples: torch.Tensor) -> None:
    if samples.numel() == 0:
      return
    samples = samples.reshape(-1, self.obs_dim).detach()
    batch_mean = samples.mean(dim=0)
    batch_var = samples.var(dim=0, unbiased=False)
    batch_count = torch.tensor(
      float(samples.shape[0]),
      device=samples.device,
      dtype=samples.dtype,
    )
    delta = batch_mean - self.mean
    total_count = self.count + batch_count
    new_mean = self.mean + delta * batch_count / total_count
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count
    self.mean.copy_(new_mean)
    self.var.copy_(torch.clamp(m_2 / total_count, min=self.eps))
    self.count.copy_(total_count)

  def forward(self, samples: torch.Tensor) -> torch.Tensor:
    return (samples - self.mean) / torch.sqrt(self.var + self.eps)


class MotionLoader:
  def __init__(
    self,
    motion_files: Iterable[str],
    amp_obs_dim: int,
    device: torch.device,
    *,
    joint_pos_scale: Iterable[float] | None = None,
    joint_pos_bias: Iterable[float] | None = None,
  ) -> None:
    motion_files = tuple(motion_files)
    self.amp_obs_dim = amp_obs_dim
    self.transition_dim = amp_obs_dim * 2
    self.device = device
    self._joint_coordinate_scale = self._resolve_joint_transform(
      joint_pos_scale, "scale"
    )
    self._joint_pos_bias = self._resolve_joint_transform(joint_pos_bias, "bias")
    transitions = []
    self.source_files: list[str] = []
    for pattern in motion_files:
      for path in self._expand_pattern(pattern):
        loaded = self._load_file(path)
        if loaded is not None:
          self.source_files.append(str(path))
          transitions.append(loaded)
    if transitions:
      self.transitions = torch.cat(transitions, dim=0).to(device=device)
      print(
        f"[WMP] Loaded {self.transitions.shape[0]} AMP transitions "
        f"from {len(self.source_files)} motion files."
      )
    else:
      self.transitions = torch.empty(0, self.transition_dim, device=device)
      if motion_files:
        print("[WMP] No AMP motion files loaded; training falls back to task reward.")

  @property
  def has_data(self) -> bool:
    return self.transitions.numel() > 0

  def joint_position_stats(self) -> dict[str, torch.Tensor]:
    if not self.has_data or self.amp_obs_dim < 12:
      empty = torch.empty(0, device=self.device)
      return {"min": empty, "mean": empty, "max": empty, "std": empty}
    joint_pos = self.transitions[:, :12]
    return {
      "min": joint_pos.min(dim=0).values,
      "mean": joint_pos.mean(dim=0),
      "max": joint_pos.max(dim=0).values,
      "std": joint_pos.std(dim=0, unbiased=False),
    }

  def _resolve_joint_transform(
    self, values: Iterable[float] | None, name: str
  ) -> np.ndarray | None:
    if values is None:
      return None
    array = np.asarray(tuple(values), dtype=np.float32)
    if array.shape != (12,):
      raise ValueError(f"AMP expert_joint_pos_{name} must contain exactly 12 values.")
    return array

  def _expand_pattern(self, pattern: str) -> list[Path]:
    expanded = os.path.expandvars(pattern)
    path = Path(expanded).expanduser()
    if path.is_dir():
      return sorted(path.glob("*.txt"))
    if path.exists():
      return [path]
    return [Path(name) for name in sorted(glob.glob(expanded))]

  def _load_file(self, path: Path) -> torch.Tensor | None:
    json_array = self._load_wmp_motion_json(path)
    if json_array is not None:
      return json_array
    try:
      array = np.loadtxt(path, dtype=np.float32)
    except Exception:
      try:
        array = np.loadtxt(path, dtype=np.float32, delimiter=",")
      except Exception:
        return None
    if array.ndim == 1:
      array = array.reshape(1, -1)
    if array.shape[0] == 0:
      return None
    if array.shape[1] >= self.transition_dim:
      data = array[:, : self.transition_dim]
      data = self._apply_transition_joint_transform(data)
    elif array.shape[1] >= self.amp_obs_dim and array.shape[0] >= 2:
      obs = array[:, : self.amp_obs_dim]
      obs = self._apply_joint_transform(obs)
      data = np.concatenate((obs[:-1], obs[1:]), axis=-1)
    else:
      return None
    return torch.from_numpy(np.ascontiguousarray(data))

  def _load_wmp_motion_json(self, path: Path) -> torch.Tensor | None:
    try:
      with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    except Exception:
      return None
    frames = np.asarray(data.get("Frames", []), dtype=np.float32)
    if frames.ndim != 2 or frames.shape[0] < 2:
      return None

    # Original WMP mocap frame layout:
    # root_pos(3), root_quat(4), joint_pos(12), toe_pos(12),
    # base_lin_vel(3), base_ang_vel(3), joint_vel(12), toe_vel(12).
    if frames.shape[1] >= 49 and self.amp_obs_dim == 30:
      amp_obs = np.concatenate(
        (
          frames[:, 7:19],
          frames[:, 31:37],
          frames[:, 37:49],
        ),
        axis=-1,
      )
    elif frames.shape[1] >= self.amp_obs_dim:
      amp_obs = frames[:, : self.amp_obs_dim]
    else:
      return None

    amp_obs = self._apply_joint_transform(amp_obs)
    transitions = np.concatenate((amp_obs[:-1], amp_obs[1:]), axis=-1)
    return torch.from_numpy(np.ascontiguousarray(transitions))

  def _apply_joint_transform(self, amp_obs: np.ndarray) -> np.ndarray:
    if self.amp_obs_dim < 12:
      return amp_obs
    if self._joint_coordinate_scale is None and self._joint_pos_bias is None:
      return amp_obs
    amp_obs = np.array(amp_obs, dtype=np.float32, copy=True)
    if self._joint_coordinate_scale is not None:
      # A sign/scale change of a joint coordinate must be applied to both
      # position and velocity. Applying it only to joint_pos makes the expert
      # transition physically inconsistent: dq/dt and joint_vel point in
      # opposite directions for flipped coordinates.
      amp_obs[:, :12] *= self._joint_coordinate_scale
      if self.amp_obs_dim >= 30:
        amp_obs[:, 18:30] *= self._joint_coordinate_scale
    if self._joint_pos_bias is not None:
      amp_obs[:, :12] += self._joint_pos_bias
    return amp_obs

  def _apply_transition_joint_transform(self, transitions: np.ndarray) -> np.ndarray:
    if self.amp_obs_dim < 12:
      return transitions
    if self._joint_coordinate_scale is None and self._joint_pos_bias is None:
      return transitions
    transitions = np.array(transitions, dtype=np.float32, copy=True)
    first = transitions[:, : self.amp_obs_dim]
    second = transitions[:, self.amp_obs_dim : self.transition_dim]
    transitions[:, : self.amp_obs_dim] = self._apply_joint_transform(first)
    transitions[:, self.amp_obs_dim : self.transition_dim] = (
      self._apply_joint_transform(second)
    )
    return transitions

  def sample(self, batch_size: int) -> torch.Tensor | None:
    if not self.has_data:
      return None
    ids = torch.randint(
      self.transitions.shape[0],
      (batch_size,),
      device=self.transitions.device,
    )
    return self.transitions[ids]


class AMPDiscriminator(nn.Module):
  def __init__(
    self,
    amp_obs_dim: int,
    *,
    hidden_dims: tuple[int, ...],
    activation: str = "elu",
  ) -> None:
    super().__init__()
    self.amp_obs_dim = amp_obs_dim
    self.transition_dim = amp_obs_dim * 2
    self.net = build_mlp(self.transition_dim, hidden_dims, 1, activation)

  def logits(self, transitions: torch.Tensor) -> torch.Tensor:
    return self.net(transitions).squeeze(-1)

  def reward(
    self,
    amp_obs: torch.Tensor,
    next_amp_obs: torch.Tensor,
    *,
    reward_scale: float,
  ) -> torch.Tensor:
    transitions = torch.cat((amp_obs, next_amp_obs), dim=-1)
    prob_expert = torch.sigmoid(self.logits(transitions))
    reward = -torch.log(torch.clamp(1.0 - prob_expert, min=1.0e-4))
    return reward_scale * torch.clamp(reward, max=10.0)

  def loss(
    self, policy_transitions: torch.Tensor, expert_transitions: torch.Tensor
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    policy_logits = self.logits(policy_transitions)
    expert_logits = self.logits(expert_transitions)
    policy_loss = F.binary_cross_entropy_with_logits(
      policy_logits,
      torch.zeros_like(policy_logits),
    )
    expert_loss = F.binary_cross_entropy_with_logits(
      expert_logits,
      torch.ones_like(expert_logits),
    )
    loss = 0.5 * (policy_loss + expert_loss)
    with torch.no_grad():
      policy_acc = (policy_logits < 0.0).float().mean()
      expert_acc = (expert_logits > 0.0).float().mean()
    return loss, {
      "policy_acc": policy_acc,
      "expert_acc": expert_acc,
      "policy_loss": policy_loss.detach(),
      "expert_loss": expert_loss.detach(),
    }
