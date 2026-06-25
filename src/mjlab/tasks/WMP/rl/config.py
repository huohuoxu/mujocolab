from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from mjlab.rl import RslRlBaseRunnerCfg


def _default_motion_files() -> Tuple[str, ...]:
  root = Path(__file__).resolve().parents[1]
  patterns = [str(root / "data" / "mocap_motions" / "*.txt")]
  env_dir = os.environ.get("WMP_MOTION_DATA_DIR")
  if env_dir:
    env_path = Path(env_dir).expanduser()
    has_glob = any(c in env_dir for c in "*?[]")
    if has_glob or env_path.suffix == ".txt":
      patterns.append(str(env_path))
    else:
      patterns.append(str(env_path / "*.txt"))
  return tuple(patterns)


@dataclass
class WmpPolicyCfg:
  hidden_dims: Tuple[int, ...] = (512, 256, 128)
  activation: str = "elu"
  history_length: int = 5
  history_latent_dim: int = 128
  wm_feature_dim: int = 128
  wm_latent_dim: int = 64
  command_dim: int = 3
  init_std: float = 1.0


@dataclass
class WmpAlgorithmCfg:
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  learning_rate: float = 1.0e-3
  gamma: float = 0.99
  lam: float = 0.95
  clip_param: float = 0.2
  entropy_coef: float = 0.01
  value_loss_coef: float = 1.0
  max_grad_norm: float = 1.0
  normalize_advantage_per_mini_batch: bool = False
  amp_task_reward_lerp: float = 0.3


@dataclass
class WmpWorldModelCfg:
  hidden_dims: Tuple[int, ...] = (512, 512)
  feature_dim: int = 128
  stoch_dim: int = 32
  min_std: float = 0.1
  learning_rate: float = 3.0e-4
  update_interval: int = 5
  kl_free: float = 1.0
  dyn_scale: float = 0.5
  rep_scale: float = 0.1
  prop_loss_scale: float = 1.0
  recon_loss_scale: float = 1.0
  reward_loss_scale: float = 1.0
  continue_loss_scale: float = 1.0
  depth_loss_scale: float = 0.1
  latent_l2_scale: float = 1.0e-4


@dataclass
class WmpDepthPredictorCfg:
  hidden_dims: Tuple[int, ...] = (512, 512)
  learning_rate: float = 3.0e-4
  update_fraction: float = 1.0
  camera_update_interval: int = 5
  camera_num_envs: int = 1024
  predict_non_camera_envs: bool = True


@dataclass
class WmpAmpCfg:
  hidden_dims: Tuple[int, ...] = (1024, 512)
  learning_rate: float = 1.0e-4
  updates_per_iteration: int = 1
  reward_scale: float = 0.01
  normalize_input: bool = True
  expert_motion_files: Tuple[str, ...] = field(default_factory=_default_motion_files)
  diagnostics_enabled: bool = True
  joint_offset_warn_abs: float = 0.75
  joint_offset_warn_mean: float = 0.35
  expert_joint_pos_scale: Tuple[float, ...] | None = (
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
  expert_joint_pos_bias: Tuple[float, ...] | None = None


@dataclass
class WmpRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "WMPRunner"
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      "actor": ("actor",),
      "critic": ("critic",),
      "wm_prop": ("wm_prop",),
      "wm_depth": ("wm_depth",),
      "wm_forward_height_map": ("wm_forward_height_map",),
      "amp": ("amp",),
    }
  )
  policy: WmpPolicyCfg = field(default_factory=WmpPolicyCfg)
  algorithm: WmpAlgorithmCfg = field(default_factory=WmpAlgorithmCfg)
  world_model: WmpWorldModelCfg = field(default_factory=WmpWorldModelCfg)
  depth_predictor: WmpDepthPredictorCfg = field(default_factory=WmpDepthPredictorCfg)
  amp: WmpAmpCfg = field(default_factory=WmpAmpCfg)
