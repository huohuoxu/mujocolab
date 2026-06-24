from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.WMP.rl.amp import AMPDiscriminator, MotionLoader, RunningNormalizer
from mjlab.tasks.WMP.rl.modules import ActorCriticWMP, DepthPredictor, SimpleWorldModel


class WMPRunner:
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict[str, Any],
    log_dir: str | None = None,
    device: str = "cpu",
    **kwargs,
  ) -> None:
    del kwargs
    self.env = env
    self.cfg = train_cfg
    self.log_dir = log_dir
    self.device = torch.device(device)
    self.num_envs = env.num_envs
    self.num_steps_per_env = int(train_cfg.get("num_steps_per_env", 24))
    self.current_learning_iteration = 0
    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)

    policy_cfg = train_cfg.get("policy", {})
    alg_cfg = train_cfg.get("algorithm", {})
    world_cfg = train_cfg.get("world_model", {})
    depth_cfg = train_cfg.get("depth_predictor", {})
    amp_cfg = train_cfg.get("amp", {})

    obs = env.get_observations()
    tensors = self._obs_to_tensors(obs)
    actor_dim = tensors["actor"].shape[-1]
    critic_dim = tensors["critic"].shape[-1]
    prop_dim = tensors["wm_prop"].shape[-1]
    height_dim = tensors["height"].shape[-1]
    depth_dim = tensors["depth"].shape[-1]
    amp_obs_dim = tensors["amp"].shape[-1]
    action_dim = env.num_actions

    history_length = int(policy_cfg.get("history_length", 5))
    wm_feature_dim = int(
      world_cfg.get("feature_dim", policy_cfg.get("wm_feature_dim", 128))
    )
    self.actor_critic = ActorCriticWMP(
      actor_dim,
      critic_dim,
      action_dim,
      history_length=history_length,
      hidden_dims=tuple(policy_cfg.get("hidden_dims", (512, 256, 128))),
      activation=policy_cfg.get("activation", "elu"),
      history_latent_dim=int(policy_cfg.get("history_latent_dim", 128)),
      wm_feature_dim=wm_feature_dim,
      wm_latent_dim=int(policy_cfg.get("wm_latent_dim", 64)),
      command_dim=int(policy_cfg.get("command_dim", 3)),
      init_std=float(policy_cfg.get("init_std", 1.0)),
    ).to(self.device)
    self.world_model = SimpleWorldModel(
      prop_dim,
      action_dim,
      depth_dim,
      feature_dim=wm_feature_dim,
      hidden_dims=tuple(world_cfg.get("hidden_dims", (512, 512))),
      stoch_dim=int(world_cfg.get("stoch_dim", 32)),
      min_std=float(world_cfg.get("min_std", 0.1)),
    ).to(self.device)
    self.depth_predictor = DepthPredictor(
      height_dim,
      prop_dim,
      depth_dim,
      hidden_dims=tuple(depth_cfg.get("hidden_dims", (512, 512))),
    ).to(self.device)
    self.amp_discriminator = AMPDiscriminator(
      amp_obs_dim,
      hidden_dims=tuple(amp_cfg.get("hidden_dims", (1024, 512))),
    ).to(self.device)
    self.amp_normalize_input = bool(amp_cfg.get("normalize_input", True))
    self.amp_normalizer = RunningNormalizer(amp_obs_dim).to(self.device)

    self.optimizer = torch.optim.Adam(
      self.actor_critic.parameters(),
      lr=float(alg_cfg.get("learning_rate", 1.0e-3)),
    )
    self.world_optimizer = torch.optim.Adam(
      self.world_model.parameters(),
      lr=float(world_cfg.get("learning_rate", 3.0e-4)),
    )
    depth_params = list(self.depth_predictor.parameters())
    self.depth_optimizer = (
      torch.optim.Adam(depth_params, lr=float(depth_cfg.get("learning_rate", 3.0e-4)))
      if depth_params
      else None
    )
    self.amp_optimizer = torch.optim.Adam(
      self.amp_discriminator.parameters(),
      lr=float(amp_cfg.get("learning_rate", 1.0e-4)),
    )
    self.motion_loader = MotionLoader(
      amp_cfg.get("expert_motion_files", ()),
      amp_obs_dim,
      self.device,
    )

    self.history_length = history_length
    self.actor_dim = actor_dim
    self._history = tensors["actor"].unsqueeze(1).repeat(1, history_length, 1)
    self.depth_update_interval = max(
      1, int(depth_cfg.get("camera_update_interval", 5))
    )
    self.num_camera_envs = min(
      self.num_envs, int(depth_cfg.get("camera_num_envs", self.num_envs))
    )
    self.predict_non_camera_envs = bool(
      depth_cfg.get("predict_non_camera_envs", True)
    )
    self._depth_camera_env_ids = torch.arange(
      self.num_camera_envs, device=self.device
    )
    self._depth_cache = tensors["depth"].detach().clone()
    self._depth_step = 0
    self._writer = self._make_writer(log_dir)

  def _make_writer(self, log_dir: str | None):
    if log_dir is None:
      return None
    try:
      from torch.utils.tensorboard import SummaryWriter
    except Exception:
      return None
    return SummaryWriter(log_dir=os.path.join(log_dir, "summaries"))

  def add_git_repo_to_log(self, repo_file_path: str) -> None:
    del repo_file_path

  def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
  ) -> None:
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf,
        high=self.env.max_episode_length,
      )

    obs = self.env.get_observations()
    tensors = self._obs_to_tensors(obs)
    self._history = tensors["actor"].unsqueeze(1).repeat(1, self.history_length, 1)
    start = self.current_learning_iteration
    end = start + num_learning_iterations
    for iteration in range(start, end):
      rollout = self._collect_rollout(obs)
      obs = rollout["last_obs"]
      update_stats = self._update_policy(rollout)
      wm_stats = self._update_world_model(rollout)
      depth_stats = self._update_depth_predictor(rollout)
      amp_stats = self._update_amp(rollout)
      self.current_learning_iteration = iteration + 1
      self._log(iteration + 1, rollout, update_stats, wm_stats, depth_stats, amp_stats)
      save_interval = int(self.cfg["save_interval"])
      if self.log_dir is not None and (iteration + 1) % save_interval == 0:
        self.save(str(Path(self.log_dir) / f"model_{iteration + 1}.pt"))

    if self.log_dir is not None:
      self.save(str(Path(self.log_dir) / f"model_{self.current_learning_iteration}.pt"))
    if self._writer is not None:
      self._writer.flush()

  def _collect_rollout(self, obs) -> dict[str, Any]:
    storage: dict[str, list[torch.Tensor]] = {
      "actor": [],
      "critic": [],
      "history": [],
      "wm_feature": [],
      "actions": [],
      "log_prob": [],
      "values": [],
      "rewards": [],
      "task_rewards": [],
      "dones": [],
      "wm_prop": [],
      "next_wm_prop": [],
      "depth": [],
      "height": [],
      "depth_target": [],
      "depth_real_mask": [],
      "amp": [],
      "next_amp": [],
    }
    for _ in range(self.num_steps_per_env):
      tensors = self._obs_to_tensors(obs)
      wm_depth, depth_target, depth_real_mask = self._depth_for_world_model(tensors)
      with torch.no_grad():
        wm_feature = self.world_model.features(tensors["wm_prop"], wm_depth)
        actions, log_prob, values = self.actor_critic.act(
          tensors["actor"],
          tensors["critic"],
          self._history,
          wm_feature,
        )
      next_obs, task_rewards, dones, _extras = self.env.step(actions)
      next_tensors = self._obs_to_tensors(next_obs)
      done_mask = dones.bool()
      if done_mask.any():
        self._depth_cache[done_mask] = next_tensors["depth"][done_mask].detach()
      with torch.no_grad():
        if self.motion_loader.has_data:
          amp_obs = self._normalize_amp_obs(tensors["amp"])
          next_amp_obs = self._normalize_amp_obs(next_tensors["amp"])
          amp_reward = self.amp_discriminator.reward(
            amp_obs,
            next_amp_obs,
            reward_scale=float(self.cfg["amp"].get("reward_scale", 1.0)),
          )
          task_lerp = float(self.cfg["algorithm"].get("amp_task_reward_lerp", 0.3))
          rewards = task_lerp * task_rewards + (1.0 - task_lerp) * amp_reward
        else:
          rewards = task_rewards

      storage["actor"].append(tensors["actor"])
      storage["critic"].append(tensors["critic"])
      storage["history"].append(self._history.clone())
      storage["wm_feature"].append(wm_feature.detach())
      storage["actions"].append(actions.detach())
      storage["log_prob"].append(log_prob.detach())
      storage["values"].append(values.detach())
      storage["rewards"].append(rewards.detach())
      storage["task_rewards"].append(task_rewards.detach())
      storage["dones"].append(dones.float())
      storage["wm_prop"].append(tensors["wm_prop"])
      storage["next_wm_prop"].append(next_tensors["wm_prop"])
      storage["depth"].append(wm_depth)
      storage["height"].append(tensors["height"])
      storage["depth_target"].append(depth_target)
      storage["depth_real_mask"].append(depth_real_mask)
      storage["amp"].append(tensors["amp"])
      storage["next_amp"].append(next_tensors["amp"])
      self._push_history(next_tensors["actor"], dones.bool())
      self._depth_step += 1
      obs = next_obs

    last_tensors = self._obs_to_tensors(obs)
    with torch.no_grad():
      last_feature = self.world_model.features(
        last_tensors["wm_prop"],
        last_tensors["depth"],
      )
      last_values = self.actor_critic.value(last_tensors["critic"], last_feature)
    rollout = {key: torch.stack(value, dim=0) for key, value in storage.items()}
    rollout["last_values"] = last_values
    rollout["last_obs"] = obs
    rollout["depth_real_fraction"] = rollout["depth_real_mask"].float().mean()
    return rollout

  def _depth_for_world_model(
    self, tensors: dict[str, torch.Tensor]
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_depth = tensors["depth"].detach()
    if self._depth_cache.shape != raw_depth.shape:
      self._depth_cache = raw_depth.clone()

    real_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    should_update = self._depth_step % self.depth_update_interval == 0
    if should_update:
      if self.predict_non_camera_envs and self.depth_predictor.net is not None:
        with torch.no_grad():
          predicted = self.depth_predictor(
            tensors["height"], tensors["wm_prop"]
          ).detach()
        self._depth_cache = predicted
      else:
        self._depth_cache = raw_depth.clone()
      if self.num_camera_envs > 0:
        ids = self._depth_camera_env_ids
        self._depth_cache[ids] = raw_depth[ids]
        real_mask[ids] = True

    return self._depth_cache.detach().clone(), raw_depth, real_mask

  def _push_history(self, actor_obs: torch.Tensor, dones: torch.Tensor) -> None:
    self._history = torch.roll(self._history, shifts=-1, dims=1)
    self._history[:, -1, :] = actor_obs
    if dones.any():
      self._history[dones] = actor_obs[dones].unsqueeze(1).repeat(
        1, self.history_length, 1
      )

  def _compute_returns(
    self,
    rollout: dict[str, Any],
  ) -> tuple[torch.Tensor, torch.Tensor]:
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    values = rollout["values"]
    last_values = rollout["last_values"]
    gamma = float(self.cfg["algorithm"].get("gamma", 0.99))
    lam = float(self.cfg["algorithm"].get("lam", 0.95))
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(self.num_envs, device=self.device)
    for step in reversed(range(self.num_steps_per_env)):
      next_value = (
        last_values if step == self.num_steps_per_env - 1 else values[step + 1]
      )
      next_not_done = 1.0 - dones[step]
      delta = rewards[step] + gamma * next_value * next_not_done - values[step]
      gae = delta + gamma * lam * next_not_done * gae
      advantages[step] = gae
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (
      advantages.std(unbiased=False) + 1.0e-8
    )
    return returns.detach(), advantages.detach()

  def _update_policy(self, rollout: dict[str, Any]) -> dict[str, float]:
    returns, advantages = self._compute_returns(rollout)
    batch = self.num_steps_per_env * self.num_envs
    flat = {
      "actor": rollout["actor"].reshape(batch, -1),
      "critic": rollout["critic"].reshape(batch, -1),
      "history": rollout["history"].reshape(batch, self.history_length, self.actor_dim),
      "wm_feature": rollout["wm_feature"].reshape(batch, -1),
      "actions": rollout["actions"].reshape(batch, -1),
      "old_log_prob": rollout["log_prob"].reshape(batch),
      "returns": returns.reshape(batch),
      "advantages": advantages.reshape(batch),
    }
    cfg = self.cfg["algorithm"]
    num_epochs = int(cfg.get("num_learning_epochs", 5))
    num_mini_batches = max(1, int(cfg.get("num_mini_batches", 4)))
    mini_batch_size = max(1, batch // num_mini_batches)
    clip_param = float(cfg.get("clip_param", 0.2))
    value_coef = float(cfg.get("value_loss_coef", 1.0))
    entropy_coef = float(cfg.get("entropy_coef", 0.01))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    stats = {
      "policy_loss": 0.0,
      "value_loss": 0.0,
      "entropy": 0.0,
    }
    updates = 0
    for _ in range(num_epochs):
      indices = torch.randperm(batch, device=self.device)
      for start in range(0, batch, mini_batch_size):
        mb = indices[start : start + mini_batch_size]
        new_log_prob, entropy, value = self.actor_critic.evaluate_actions(
          flat["actor"][mb],
          flat["critic"][mb],
          flat["history"][mb],
          flat["wm_feature"][mb],
          flat["actions"][mb],
        )
        adv = flat["advantages"][mb]
        if cfg.get("normalize_advantage_per_mini_batch", False):
          adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1.0e-8)
        ratio = torch.exp(new_log_prob - flat["old_log_prob"][mb])
        surrogate = ratio * adv
        surrogate_clipped = torch.clamp(
          ratio,
          1.0 - clip_param,
          1.0 + clip_param,
        ) * adv
        policy_loss = -torch.min(surrogate, surrogate_clipped).mean()
        value_loss = F.mse_loss(value, flat["returns"][mb])
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
        self.optimizer.step()
        stats["policy_loss"] += float(policy_loss.detach())
        stats["value_loss"] += float(value_loss.detach())
        stats["entropy"] += float(entropy.mean().detach())
        updates += 1
    return {key: value / max(1, updates) for key, value in stats.items()}

  def _update_world_model(self, rollout: dict[str, Any]) -> dict[str, float]:
    interval = int(self.cfg["world_model"].get("update_interval", 5))
    if interval <= 0 or (self.current_learning_iteration + 1) % interval != 0:
      return {}
    prop = rollout["wm_prop"]
    next_prop = rollout["next_wm_prop"]
    actions = rollout["actions"]
    rewards = rollout["task_rewards"]
    dones = rollout["dones"]
    depth = rollout["depth"]
    loss, losses = self.world_model.loss(
      prop,
      actions,
      next_prop,
      rewards,
      dones,
      depth,
      kl_free=float(self.cfg["world_model"].get("kl_free", 1.0)),
      dyn_scale=float(self.cfg["world_model"].get("dyn_scale", 0.5)),
      rep_scale=float(self.cfg["world_model"].get("rep_scale", 0.1)),
      prop_loss_scale=float(self.cfg["world_model"].get("prop_loss_scale", 1.0)),
      recon_loss_scale=float(self.cfg["world_model"].get("recon_loss_scale", 1.0)),
      reward_loss_scale=float(self.cfg["world_model"].get("reward_loss_scale", 1.0)),
      continue_loss_scale=float(
        self.cfg["world_model"].get("continue_loss_scale", 1.0)
      ),
      depth_loss_scale=float(self.cfg["world_model"].get("depth_loss_scale", 0.1)),
      latent_l2_scale=float(self.cfg["world_model"].get("latent_l2_scale", 1.0e-4)),
    )
    self.world_optimizer.zero_grad()
    loss.backward()
    self.world_optimizer.step()
    out = {"loss": float(loss.detach())}
    out.update({key: float(value) for key, value in losses.items()})
    return out

  def _update_depth_predictor(self, rollout: dict[str, Any]) -> dict[str, float]:
    if self.depth_optimizer is None:
      return {}
    batch = self.num_steps_per_env * self.num_envs
    height = rollout["height"].reshape(batch, -1)
    prop = rollout["wm_prop"].reshape(batch, -1)
    depth = rollout["depth_target"].reshape(batch, -1)
    real_mask = rollout["depth_real_mask"].reshape(batch).bool()
    if not real_mask.any():
      return {}
    height = height[real_mask]
    prop = prop[real_mask]
    depth = depth[real_mask]
    if height.numel() == 0 or depth.numel() == 0:
      return {}
    fraction = float(self.cfg["depth_predictor"].get("update_fraction", 1.0))
    if fraction < 1.0:
      filtered_batch = height.shape[0]
      n = max(1, int(filtered_batch * fraction))
      ids = torch.randperm(filtered_batch, device=self.device)[:n]
      height = height[ids]
      prop = prop[ids]
      depth = depth[ids]
    pred = self.depth_predictor(height, prop)
    loss = F.mse_loss(pred, depth)
    self.depth_optimizer.zero_grad()
    loss.backward()
    self.depth_optimizer.step()
    return {"loss": float(loss.detach())}

  def _update_amp(self, rollout: dict[str, Any]) -> dict[str, float]:
    if not self.motion_loader.has_data:
      return {}
    updates = int(self.cfg["amp"].get("updates_per_iteration", 1))
    batch = self.num_steps_per_env * self.num_envs
    policy_amp = rollout["amp"].reshape(batch, -1)
    policy_next_amp = rollout["next_amp"].reshape(batch, -1)
    stats = {"loss": 0.0, "policy_acc": 0.0, "expert_acc": 0.0}
    count = 0
    for _ in range(updates):
      expert = self.motion_loader.sample(batch)
      if expert is None:
        break
      expert_amp = expert[:, : self.amp_discriminator.amp_obs_dim]
      expert_next_amp = expert[:, self.amp_discriminator.amp_obs_dim :]
      if self.amp_normalize_input:
        self.amp_normalizer.update(
          torch.cat(
            (
              policy_amp,
              policy_next_amp,
              expert_amp,
              expert_next_amp,
            ),
            dim=0,
          )
        )
      policy_transitions = self._amp_transition(policy_amp, policy_next_amp)
      expert_transitions = self._amp_transition(expert_amp, expert_next_amp)
      loss, loss_stats = self.amp_discriminator.loss(
        policy_transitions,
        expert_transitions,
      )
      self.amp_optimizer.zero_grad()
      loss.backward()
      self.amp_optimizer.step()
      stats["loss"] += float(loss.detach())
      stats["policy_acc"] += float(loss_stats["policy_acc"])
      stats["expert_acc"] += float(loss_stats["expert_acc"])
      count += 1
    return {key: value / max(1, count) for key, value in stats.items()}

  def _normalize_amp_obs(self, obs: torch.Tensor) -> torch.Tensor:
    if not self.amp_normalize_input:
      return obs
    return self.amp_normalizer(obs)

  def _amp_transition(
    self,
    amp_obs: torch.Tensor,
    next_amp_obs: torch.Tensor,
  ) -> torch.Tensor:
    return torch.cat(
      (
        self._normalize_amp_obs(amp_obs),
        self._normalize_amp_obs(next_amp_obs),
      ),
      dim=-1,
    )

  def _log(
    self,
    iteration: int,
    rollout: dict[str, Any],
    ppo: dict[str, float],
    world: dict[str, float],
    depth: dict[str, float],
    amp: dict[str, float],
  ) -> None:
    reward = float(rollout["task_rewards"].mean())
    if self._writer is not None:
      self._writer.add_scalar("Train/task_reward", reward, iteration)
      self._writer.add_scalar(
        "Depth/real_fraction", float(rollout["depth_real_fraction"]), iteration
      )
      for key, value in ppo.items():
        self._writer.add_scalar(f"PPO/{key}", value, iteration)
      for key, value in world.items():
        self._writer.add_scalar(f"WorldModel/{key}", value, iteration)
      for key, value in depth.items():
        self._writer.add_scalar(f"DepthPredictor/{key}", value, iteration)
      for key, value in amp.items():
        self._writer.add_scalar(f"AMP/{key}", value, iteration)
    if iteration == 1 or iteration % 10 == 0:
      print(
        f"[WMP] iter={iteration} "
        f"reward={reward:.4f} "
        f"policy_loss={ppo.get('policy_loss', 0.0):.4f} "
        f"value_loss={ppo.get('value_loss', 0.0):.4f}"
      )

  def save(self, path: str, infos=None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    infos = infos or {}
    infos["env_state"] = {
      "common_step_counter": self.env.unwrapped.common_step_counter,
    }
    state = {
      "iter": self.current_learning_iteration,
      "infos": infos,
      "cfg": self.cfg,
      "actor_critic_state_dict": self.actor_critic.state_dict(),
      "world_model_state_dict": self.world_model.state_dict(),
      "depth_predictor_state_dict": self.depth_predictor.state_dict(),
      "amp_discriminator_state_dict": self.amp_discriminator.state_dict(),
      "amp_normalizer_state_dict": self.amp_normalizer.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
      "world_optimizer_state_dict": self.world_optimizer.state_dict(),
      "depth_optimizer_state_dict": self.depth_optimizer.state_dict()
      if self.depth_optimizer is not None
      else None,
      "amp_optimizer_state_dict": self.amp_optimizer.state_dict(),
      "runner_state": {
        "depth_step": self._depth_step,
      },
    }
    torch.save(state, path)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | torch.device | None = None,
  ) -> dict:
    try:
      state = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
      state = torch.load(path, map_location=map_location)
    load_actor_only = bool(load_cfg and load_cfg.get("actor", False))
    self.actor_critic.load_state_dict(
      state["actor_critic_state_dict"],
      strict=strict,
    )
    if "world_model_state_dict" in state:
      self.world_model.load_state_dict(state["world_model_state_dict"], strict=strict)
    if "depth_predictor_state_dict" in state:
      self.depth_predictor.load_state_dict(
        state["depth_predictor_state_dict"],
        strict=strict,
      )
    if not load_actor_only:
      self.amp_discriminator.load_state_dict(
        state["amp_discriminator_state_dict"],
        strict=strict,
      )
      if "amp_normalizer_state_dict" in state:
        self.amp_normalizer.load_state_dict(
          state["amp_normalizer_state_dict"],
          strict=strict,
        )
      if "optimizer_state_dict" in state:
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
      if "world_optimizer_state_dict" in state:
        self.world_optimizer.load_state_dict(state["world_optimizer_state_dict"])
      if self.depth_optimizer is not None and state.get("depth_optimizer_state_dict"):
        self.depth_optimizer.load_state_dict(state["depth_optimizer_state_dict"])
      if "amp_optimizer_state_dict" in state:
        self.amp_optimizer.load_state_dict(state["amp_optimizer_state_dict"])
    runner_state = state.get("runner_state", {})
    self._depth_step = int(runner_state.get("depth_step", self._depth_step))
    self.current_learning_iteration = int(state.get("iter", 0))
    infos = state.get("infos", {})
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"].get(
        "common_step_counter",
        self.env.unwrapped.common_step_counter,
      )
    return infos

  def get_inference_policy(self, device: str | torch.device | None = None):
    if device is not None:
      self.actor_critic.to(device)
      self.world_model.to(device)
      self.device = torch.device(device)
    self.actor_critic.eval()
    self.world_model.eval()
    history: dict[str, torch.Tensor | None] = {"value": None}

    def policy(obs) -> torch.Tensor:
      tensors = self._obs_to_tensors(obs)
      actor_obs = tensors["actor"]
      if history["value"] is None or history["value"].shape[0] != actor_obs.shape[0]:
        history["value"] = actor_obs.unsqueeze(1).repeat(1, self.history_length, 1)
      else:
        old_history = history["value"]
        assert old_history is not None
        old_history = torch.roll(old_history, shifts=-1, dims=1)
        old_history[:, -1, :] = actor_obs
        history["value"] = old_history
      with torch.no_grad():
        wm_feature = self.world_model.features(tensors["wm_prop"], tensors["depth"])
        assert history["value"] is not None
        return self.actor_critic.act_inference(
          actor_obs,
          history["value"],
          wm_feature,
        )

    return policy

  def _obs_to_tensors(self, obs) -> dict[str, torch.Tensor]:
    return {
      "actor": self._flatten(self._group(obs, "actor")),
      "critic": self._flatten(self._group(obs, "critic")),
      "wm_prop": self._flatten(self._group(obs, "wm_prop")),
      "depth": self._flatten(self._group(obs, "wm_depth", "depth")),
      "height": self._flatten(self._group(obs, "wm_forward_height_map", "height")),
      "amp": self._flatten(self._group(obs, "amp")),
    }

  def _group(self, obs, group_name: str, term_name: str | None = None):
    value = obs[group_name]
    if torch.is_tensor(value):
      return value
    if term_name is not None:
      return value[term_name]
    keys = list(value.keys())
    if not keys:
      raise KeyError(f"Observation group {group_name!r} is empty.")
    return value[keys[0]]

  def _flatten(self, tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.to(device=self.device, dtype=torch.float32)
    return tensor.reshape(tensor.shape[0], -1)
