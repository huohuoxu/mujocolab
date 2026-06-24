from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.WMP.rl.amp import MotionLoader
from mjlab.tasks.WMP.rl.modules import (
  ActorCriticWMP,
  DepthPredictor,
  SimpleWorldModel,
)
from mjlab.tasks.WMP.rl.runner import WMPRunner
from mjlab.tasks.registry import list_tasks, load_runner_cls


TASK_ID = "Mjlab-WMP-Rough-Unitree-Go1"


def test_wmp_task_registered_and_cfg_serializable():
  assert TASK_ID in list_tasks()
  assert load_runner_cls(TASK_ID) is WMPRunner

  cfg = TrainConfig.from_task(TASK_ID)
  data = asdict(cfg)

  assert data["agent"]["class_name"] == "WMPRunner"
  assert data["agent"]["obs_groups"]["wm_depth"] == ("wm_depth",)
  assert data["agent"]["depth_predictor"]["camera_update_interval"] == 5
  assert "wm_forward_height_map" in data["agent"]["obs_groups"]
  assert "front_depth_camera" in {
    sensor.name for sensor in cfg.env.scene.sensors if hasattr(sensor, "name")
  }


def test_wmp_motion_loader_reads_original_json_layout(tmp_path: Path):
  frames = torch.arange(3 * 61, dtype=torch.float32).reshape(3, 61).tolist()
  motion_path = tmp_path / "motion.txt"
  motion_path.write_text(json.dumps({"Frames": frames}), encoding="utf-8")

  loader = MotionLoader((str(motion_path),), amp_obs_dim=30, device=torch.device("cpu"))

  assert loader.has_data
  assert loader.transitions.shape == (2, 60)
  frame_tensor = torch.tensor(frames)
  expected_obs_0 = torch.cat(
    (
      frame_tensor[0, 7:19],
      frame_tensor[0, 31:37],
      frame_tensor[0, 37:49],
    )
  )
  expected_obs_1 = torch.cat(
    (
      frame_tensor[1, 7:19],
      frame_tensor[1, 31:37],
      frame_tensor[1, 37:49],
    )
  )
  assert torch.equal(loader.transitions[0], torch.cat((expected_obs_0, expected_obs_1)))


def test_wmp_modules_shapes_and_losses_are_finite():
  batch = 4
  actor_dim = 45
  critic_dim = 255
  action_dim = 12
  prop_dim = 45
  depth_dim = 64
  height_dim = 32
  feature_dim = 16
  hidden_dims = (32,)

  actor_critic = ActorCriticWMP(
    actor_dim,
    critic_dim,
    action_dim,
    history_length=3,
    hidden_dims=hidden_dims,
    activation="elu",
    history_latent_dim=16,
    wm_feature_dim=feature_dim,
    wm_latent_dim=8,
    command_dim=3,
    init_std=0.5,
  )
  world_model = SimpleWorldModel(
    prop_dim,
    action_dim,
    depth_dim,
    feature_dim=feature_dim,
    hidden_dims=hidden_dims,
  )
  depth_predictor = DepthPredictor(
    height_dim,
    prop_dim,
    depth_dim,
    hidden_dims=hidden_dims,
  )

  actor = torch.randn(batch, actor_dim)
  critic = torch.randn(batch, critic_dim)
  history = torch.randn(batch, 3, actor_dim)
  prop = torch.randn(batch, prop_dim)
  next_prop = torch.randn(batch, prop_dim)
  actions = torch.randn(batch, action_dim)
  rewards = torch.randn(batch)
  dones = torch.zeros(batch)
  depth = torch.randn(batch, depth_dim)
  height = torch.randn(batch, height_dim)
  wm_feature = world_model.features(prop)

  sampled_action, log_prob, value = actor_critic.act(
    actor,
    critic,
    history,
    wm_feature,
  )
  assert sampled_action.shape == (batch, action_dim)
  assert log_prob.shape == (batch,)
  assert value.shape == (batch,)

  loss, losses = world_model.loss(
    prop,
    actions,
    next_prop,
    rewards,
    dones,
    depth,
    reward_loss_scale=1.0,
    continue_loss_scale=1.0,
    depth_loss_scale=0.1,
    latent_l2_scale=1.0e-4,
  )
  assert torch.isfinite(loss)
  assert all(torch.isfinite(value) for value in losses.values())
  for key in ("kl", "dyn_loss", "rep_loss", "prior_ent", "post_ent"):
    assert key in losses
  assert depth_predictor(height, prop).shape == (batch, depth_dim)


def test_wmp_world_model_loss_handles_sequence_resets():
  batch = 3
  time_steps = 4
  prop_dim = 45
  action_dim = 12
  depth_dim = 16
  world_model = SimpleWorldModel(
    prop_dim,
    action_dim,
    depth_dim,
    feature_dim=16,
    hidden_dims=(32,),
    stoch_dim=8,
  )
  prop = torch.randn(time_steps, batch, prop_dim)
  next_prop = torch.randn(time_steps, batch, prop_dim)
  action = torch.randn(time_steps, batch, action_dim)
  reward = torch.randn(time_steps, batch)
  done = torch.zeros(time_steps, batch)
  done[1, 0] = 1.0
  done[2, 2] = 1.0
  depth = torch.randn(time_steps, batch, depth_dim)

  loss, losses = world_model.loss(
    prop,
    action,
    next_prop,
    reward,
    done,
    depth,
    kl_free=1.0,
    dyn_scale=0.5,
    rep_scale=0.1,
    prop_loss_scale=1.0,
    recon_loss_scale=1.0,
    reward_loss_scale=1.0,
    continue_loss_scale=1.0,
    depth_loss_scale=0.1,
    latent_l2_scale=1.0e-4,
  )

  assert torch.isfinite(loss)
  assert all(torch.isfinite(value) for value in losses.values())


class _FakeWmpEnv:
  num_envs = 2
  num_actions = 12
  max_episode_length = 20

  def __init__(self) -> None:
    self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
    self.common_step_counter = 0

  @property
  def unwrapped(self):
    return self

  def get_observations(self):
    return self._obs()

  def step(self, actions: torch.Tensor):
    self.common_step_counter += 1
    self.episode_length_buf += 1
    obs = self._obs()
    rewards = -0.01 * actions.square().mean(dim=-1)
    dones = torch.zeros(self.num_envs, dtype=torch.bool)
    return obs, rewards, dones, {}

  def _obs(self):
    value = float(self.common_step_counter)
    return {
      "actor": torch.full((self.num_envs, 45), value),
      "critic": torch.full((self.num_envs, 255), value),
      "wm_prop": torch.full((self.num_envs, 45), value),
      "wm_depth": {
        "depth": torch.full((self.num_envs, 64, 64, 1), value / 10.0),
      },
      "wm_forward_height_map": {
        "height": torch.full((self.num_envs, 525), value / 20.0),
      },
      "amp": torch.full((self.num_envs, 30), value),
    }


def _tiny_runner_cfg(expert_motion_files: tuple[str, ...] = ()) -> dict:
  return {
    "num_steps_per_env": 2,
    "save_interval": 100,
    "policy": {
      "hidden_dims": (16,),
      "activation": "elu",
      "history_length": 2,
      "history_latent_dim": 8,
      "wm_feature_dim": 16,
      "wm_latent_dim": 8,
      "command_dim": 3,
      "init_std": 0.5,
    },
    "algorithm": {
      "num_learning_epochs": 1,
      "num_mini_batches": 1,
      "learning_rate": 1.0e-3,
      "gamma": 0.99,
      "lam": 0.95,
      "clip_param": 0.2,
      "entropy_coef": 0.0,
      "value_loss_coef": 1.0,
      "max_grad_norm": 1.0,
      "amp_task_reward_lerp": 0.3,
    },
    "world_model": {
      "hidden_dims": (16,),
      "feature_dim": 16,
      "learning_rate": 1.0e-3,
      "update_interval": 1,
      "reward_loss_scale": 1.0,
      "continue_loss_scale": 1.0,
      "depth_loss_scale": 0.01,
      "latent_l2_scale": 1.0e-4,
    },
    "depth_predictor": {
      "hidden_dims": (16,),
      "learning_rate": 1.0e-3,
      "update_fraction": 1.0,
      "camera_update_interval": 2,
      "camera_num_envs": 1,
      "predict_non_camera_envs": True,
    },
    "amp": {
      "hidden_dims": (16,),
      "learning_rate": 1.0e-3,
      "updates_per_iteration": 1,
      "reward_scale": 1.0,
      "normalize_input": True,
      "expert_motion_files": expert_motion_files,
    },
  }


def test_wmp_runner_learn_save_load_with_fake_env(tmp_path: Path):
  expert_path = tmp_path / "amp_expert.txt"
  expert = torch.arange(8 * 60, dtype=torch.float32).reshape(8, 60)
  expert_path.write_text(
    "\n".join(" ".join(str(float(v)) for v in row) for row in expert),
    encoding="utf-8",
  )
  env = _FakeWmpEnv()
  runner = WMPRunner(
    env,
    _tiny_runner_cfg((str(expert_path),)),
    device="cpu",
  )

  runner.learn(num_learning_iterations=1)

  assert runner.current_learning_iteration == 1
  assert runner._depth_step == 2
  assert runner.amp_normalizer.count > 1.0
  assert runner.world_model.stoch_dim == 32

  checkpoint = tmp_path / "model.pt"
  runner.save(str(checkpoint))
  assert checkpoint.exists()

  loaded = WMPRunner(
    _FakeWmpEnv(),
    _tiny_runner_cfg((str(expert_path),)),
    device="cpu",
  )
  loaded.load(str(checkpoint))
  policy = loaded.get_inference_policy(device="cpu")
  action = policy(loaded.env.get_observations())

  assert loaded.current_learning_iteration == 1
  assert torch.equal(loaded.amp_normalizer.count, runner.amp_normalizer.count)
  assert action.shape == (loaded.env.num_envs, loaded.env.num_actions)
