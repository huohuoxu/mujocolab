from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from mjlab.tasks.WMP.rl.dreamer.models import WorldModel as DreamerWorldModel


def _activation(name: str) -> type[nn.Module]:
  table: dict[str, type[nn.Module]] = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
  }
  if name not in table:
    raise ValueError(f"Unsupported activation: {name}")
  return table[name]


def build_mlp(
  input_dim: int,
  hidden_dims: Sequence[int],
  output_dim: int,
  activation: str = "elu",
  final_activation: nn.Module | None = None,
) -> nn.Sequential:
  layers: list[nn.Module] = []
  last_dim = input_dim
  act_cls = _activation(activation)
  for hidden_dim in hidden_dims:
    layers.append(nn.Linear(last_dim, hidden_dim))
    layers.append(act_cls())
    last_dim = hidden_dim
  layers.append(nn.Linear(last_dim, output_dim))
  if final_activation is not None:
    layers.append(final_activation)
  return nn.Sequential(*layers)


class ActorCriticWMP(nn.Module):
  def __init__(
    self,
    actor_dim: int,
    critic_dim: int,
    action_dim: int,
    *,
    history_dim: int | None = None,
    history_length: int,
    hidden_dims: Sequence[int] | None = None,
    encoder_hidden_dims: Sequence[int] | None = None,
    wm_encoder_hidden_dims: Sequence[int] | None = None,
    actor_hidden_dims: Sequence[int] | None = None,
    critic_hidden_dims: Sequence[int] | None = None,
    activation: str,
    history_latent_dim: int,
    wm_feature_dim: int,
    wm_latent_dim: int,
    command_dim: int,
    init_std: float,
    min_std: float | None = None,
    max_std: float | None = None,
    command_slice: tuple[int, int] = (6, 9),
  ) -> None:
    super().__init__()
    self.actor_dim = actor_dim
    self.critic_dim = critic_dim
    self.action_dim = action_dim
    self.history_dim = actor_dim if history_dim is None else history_dim
    self.history_length = history_length
    self.command_dim = command_dim
    self.command_slice = command_slice
    self.min_log_std = math.log(min_std) if min_std is not None else None
    self.max_log_std = math.log(max_std) if max_std is not None else None
    if hidden_dims is None:
      hidden_dims = (512, 256, 128)
    if encoder_hidden_dims is None:
      encoder_hidden_dims = hidden_dims
    if wm_encoder_hidden_dims is None:
      wm_encoder_hidden_dims = (wm_latent_dim,)
    if actor_hidden_dims is None:
      actor_hidden_dims = hidden_dims
    if critic_hidden_dims is None:
      critic_hidden_dims = hidden_dims
    if (
      self.min_log_std is not None
      and self.max_log_std is not None
      and self.min_log_std > self.max_log_std
    ):
      raise ValueError("min_std must be less than or equal to max_std.")

    self.history_encoder = build_mlp(
      self.history_dim * history_length,
      encoder_hidden_dims,
      history_latent_dim,
      activation,
    )
    self.wm_feature_encoder = build_mlp(
      wm_feature_dim,
      wm_encoder_hidden_dims,
      wm_latent_dim,
      activation,
    )
    self.critic_wm_feature_encoder = build_mlp(
      wm_feature_dim,
      wm_encoder_hidden_dims,
      wm_latent_dim,
      activation,
    )
    self.actor = build_mlp(
      history_latent_dim + command_dim + wm_latent_dim,
      actor_hidden_dims,
      action_dim,
      activation,
    )
    self.critic = build_mlp(
      critic_dim + wm_latent_dim,
      critic_hidden_dims,
      1,
      activation,
    )
    self.log_std = nn.Parameter(torch.full((action_dim,), math.log(init_std)))

  def _bounded_log_std(self) -> torch.Tensor:
    log_std = self.log_std
    if self.min_log_std is None and self.max_log_std is None:
      return log_std
    min_log_std = self.min_log_std
    max_log_std = self.max_log_std
    return torch.clamp(log_std, min=min_log_std, max=max_log_std)

  def _command(self, actor_obs: torch.Tensor) -> torch.Tensor:
    start, end = self.command_slice
    if actor_obs.shape[-1] >= end:
      return actor_obs[:, start:end]
    return torch.zeros(
      actor_obs.shape[0],
      self.command_dim,
      dtype=actor_obs.dtype,
      device=actor_obs.device,
    )

  def _encode(
    self, actor_obs: torch.Tensor, history: torch.Tensor, wm_feature: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    history_flat = history.reshape(history.shape[0], -1)
    history_latent = self.history_encoder(history_flat)
    wm_latent = self.wm_feature_encoder(wm_feature)
    actor_input = torch.cat(
      (history_latent, self._command(actor_obs), wm_latent),
      dim=-1,
    )
    return actor_input, wm_latent

  def _dist(
    self, actor_obs: torch.Tensor, history: torch.Tensor, wm_feature: torch.Tensor
  ) -> Normal:
    actor_input, _ = self._encode(actor_obs, history, wm_feature)
    mean = self.actor(actor_input)
    std = torch.exp(self._bounded_log_std()).expand_as(mean)
    return Normal(mean, std)

  def distribution_stats(
    self,
    actor_obs: torch.Tensor,
    history: torch.Tensor,
    wm_feature: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    dist = self._dist(actor_obs, history, wm_feature)
    return dist.mean, dist.stddev

  def value(self, critic_obs: torch.Tensor, wm_feature: torch.Tensor) -> torch.Tensor:
    wm_latent = self.critic_wm_feature_encoder(wm_feature)
    return self.critic(torch.cat((critic_obs, wm_latent), dim=-1)).squeeze(-1)

  def act(
    self,
    actor_obs: torch.Tensor,
    critic_obs: torch.Tensor,
    history: torch.Tensor,
    wm_feature: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist = self._dist(actor_obs, history, wm_feature)
    actions = dist.sample()
    log_prob = dist.log_prob(actions).sum(dim=-1)
    value = self.value(critic_obs, wm_feature)
    return actions, log_prob, value

  def evaluate_actions(
    self,
    actor_obs: torch.Tensor,
    critic_obs: torch.Tensor,
    history: torch.Tensor,
    wm_feature: torch.Tensor,
    actions: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist = self._dist(actor_obs, history, wm_feature)
    log_prob = dist.log_prob(actions).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1)
    value = self.value(critic_obs, wm_feature)
    return log_prob, entropy, value

  def act_inference(
    self,
    actor_obs: torch.Tensor,
    history: torch.Tensor,
    wm_feature: torch.Tensor,
  ) -> torch.Tensor:
    return self._dist(actor_obs, history, wm_feature).mean

  def predict_linear_velocity(self, history: torch.Tensor) -> torch.Tensor:
    history_flat = history.reshape(history.shape[0], -1)
    latent = self.history_encoder(history_flat)
    if latent.shape[-1] < 3:
      raise ValueError("history_latent_dim must be at least 3 for velocity prediction.")
    return latent[:, -3:]


def _to_namespace(value):
  if isinstance(value, dict):
    converted = {}
    for key, val in value.items():
      if key in {"encoder", "decoder", "actor", "critic", "reward_head", "cont_head"}:
        converted[key] = val
      else:
        converted[key] = _to_namespace(val)
    return SimpleNamespace(**converted)
  if isinstance(value, list):
    return [_to_namespace(val) for val in value]
  return value


def _namespace_to_dict(value):
  if isinstance(value, SimpleNamespace):
    return {key: _namespace_to_dict(val) for key, val in vars(value).items()}
  if isinstance(value, list):
    return [_namespace_to_dict(val) for val in value]
  return value


class DreamerEpisodeDataset:
  """Episode-level replay buffer matching the original WMP world-model input."""

  def __init__(self, limit_steps: int = 200_000, seed: int = 0) -> None:
    self.limit_steps = int(limit_steps)
    self._episodes: list[dict[str, np.ndarray]] = []
    self._rng = np.random.RandomState(seed)
    self._steps = 0

  @property
  def num_steps(self) -> int:
    return self._steps

  @property
  def num_episodes(self) -> int:
    return len(self._episodes)

  def add_episode(self, episode: dict[str, torch.Tensor]) -> None:
    if not episode:
      return
    converted = {
      key: value.detach().cpu().numpy().astype(np.float32)
      for key, value in episode.items()
    }
    length = int(next(iter(converted.values())).shape[0])
    if length < 2:
      return
    self._episodes.append(converted)
    self._steps += length
    self._trim()

  def sample(self, batch_size: int, batch_length: int) -> dict[str, np.ndarray] | None:
    valid = [
      episode
      for episode in self._episodes
      if int(next(iter(episode.values())).shape[0]) >= batch_length
    ]
    if not valid:
      return None
    batch: list[dict[str, np.ndarray]] = []
    probs = np.asarray(
      [int(next(iter(episode.values())).shape[0]) for episode in valid],
      dtype=np.float64,
    )
    probs = probs / probs.sum()
    for _ in range(batch_size):
      episode = valid[int(self._rng.choice(len(valid), p=probs))]
      total = int(next(iter(episode.values())).shape[0])
      start = int(self._rng.randint(0, total - batch_length + 1))
      item = {
        key: value[start : start + batch_length].copy()
        for key, value in episode.items()
      }
      item["is_first"][0] = 1.0
      batch.append(item)
    return {
      key: np.stack([item[key] for item in batch], axis=0)
      for key in batch[0].keys()
    }

  def state_dict(self) -> dict[str, object]:
    return {
      "episodes": self._episodes,
      "steps": self._steps,
      "limit_steps": self.limit_steps,
      "rng_state": self._rng.get_state(),
    }

  def load_state_dict(self, state: dict[str, object]) -> None:
    self.limit_steps = int(state.get("limit_steps", self.limit_steps))
    self._episodes = list(state.get("episodes", []))
    self._steps = int(state.get("steps", 0))
    rng_state = state.get("rng_state")
    if rng_state is not None:
      self._rng.set_state(rng_state)

  def _trim(self) -> None:
    if self.limit_steps <= 0:
      return
    while self._steps > self.limit_steps and self._episodes:
      episode = self._episodes.pop(0)
      self._steps -= int(next(iter(episode.values())).shape[0])


class DreamerWorldModelAdapter(nn.Module):
  """Adapter around the original WMP Dreamer world model."""

  def __init__(
    self,
    prop_dim: int,
    action_dim: int,
    depth_shape: tuple[int, int, int],
    *,
    device: torch.device,
    config_overrides: dict[str, object] | None = None,
    use_camera: bool = True,
  ) -> None:
    super().__init__()
    self.prop_dim = int(prop_dim)
    self.action_dim = int(action_dim)
    self.depth_shape = tuple(int(v) for v in depth_shape)
    self.use_camera = bool(use_camera)
    config = self._load_config()
    config["device"] = str(device)
    config["num_actions"] = self.action_dim
    if config_overrides:
      config = self._merge_config(config, config_overrides)
    self.config = _to_namespace(config)
    obs_shape = {
      "prop": (self.prop_dim,),
      "image": self.depth_shape,
    }
    self.model = DreamerWorldModel(self.config, obs_shape, use_camera=self.use_camera)
    self.feature_dim = int(self.config.dyn_deter)
    self._state: dict[str, torch.Tensor] | None = None

  def reset_state(self, done: torch.Tensor | None = None) -> None:
    if done is None or self._state is None:
      self._state = None
      return
    if not done.any():
      return
    init = self.model.dynamics.initial(done.shape[0])
    mask = done.to(dtype=torch.float32, device=done.device)
    for key, value in self._state.items():
      view_shape = (mask.shape[0],) + (1,) * (value.dim() - 1)
      self._state[key] = value * (1.0 - mask.view(view_shape)) + init[key] * mask.view(
        view_shape
      )

  def features(
    self,
    prop: torch.Tensor,
    depth: torch.Tensor | None = None,
    action: torch.Tensor | None = None,
    is_first: torch.Tensor | None = None,
  ) -> torch.Tensor:
    batch = prop.shape[0]
    if self._state is not None:
      state_batch = next(iter(self._state.values())).shape[0]
      if state_batch != batch:
        self._state = None
    if action is None:
      action = torch.zeros(batch, self.action_dim, device=prop.device)
    if is_first is None:
      is_first = torch.zeros(batch, device=prop.device, dtype=torch.bool)
    image = self._prepare_depth(depth, batch, prop.device)
    data = {
      "prop": prop.unsqueeze(1),
      "image": image.unsqueeze(1),
      "action": action.unsqueeze(1),
      "is_first": is_first.reshape(batch, 1).float(),
    }
    with torch.no_grad():
      embed = self.model.encoder(data)
      post, _prior = self.model.dynamics.observe(
        embed,
        data["action"],
        data["is_first"],
        self._state,
      )
      self._state = {key: value[:, -1].detach() for key, value in post.items()}
      return self.model.dynamics.get_deter_feat(post)[:, -1].detach()

  def train_world_model(
    self,
    batch: dict[str, np.ndarray],
    *,
    train_steps: int = 1,
  ) -> dict[str, float]:
    metrics_sum: dict[str, float] = {}
    for _ in range(max(1, int(train_steps))):
      _post, _context, metrics = self.model._train(batch)
      for key, value in metrics.items():
        array = np.asarray(value, dtype=np.float64)
        metrics_sum[key] = metrics_sum.get(key, 0.0) + float(np.mean(array))
    return {key: value / max(1, int(train_steps)) for key, value in metrics_sum.items()}

  def state_dict(self, *args, **kwargs):  # type: ignore[override]
    return {
      "model": self.model.state_dict(*args, **kwargs),
      "config": _namespace_to_dict(self.config),
    }

  def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
    if "model" in state_dict:
      self.model.load_state_dict(state_dict["model"], strict=strict)
      self._state = None
      return
    self.model.load_state_dict(state_dict, strict=strict)
    self._state = None

  def _prepare_depth(
    self,
    depth: torch.Tensor | None,
    batch: int,
    device: torch.device,
  ) -> torch.Tensor:
    if depth is None or depth.numel() == 0:
      return torch.zeros(batch, *self.depth_shape, device=device)
    depth = depth.to(device=device, dtype=torch.float32)
    if depth.dim() == 2:
      height, width, channels = self.depth_shape
      return depth.reshape(batch, height, width, channels)
    if depth.dim() == 3:
      return depth.unsqueeze(-1)
    if depth.dim() == 4:
      return depth
    raise ValueError(f"Unsupported depth tensor shape: {tuple(depth.shape)}")

  def _load_config(self) -> dict[str, object]:
    path = Path(__file__).resolve().parent / "dreamer" / "configs.yaml"
    with path.open("r", encoding="utf-8") as f:
      data = yaml.safe_load(f)
    return self._coerce_numeric_strings(deepcopy(data["defaults"]))

  def _coerce_numeric_strings(self, value):
    if isinstance(value, dict):
      return {key: self._coerce_numeric_strings(val) for key, val in value.items()}
    if isinstance(value, list):
      return [self._coerce_numeric_strings(val) for val in value]
    if isinstance(value, str):
      try:
        return float(value)
      except ValueError:
        return value
    return value

  def _merge_config(
    self,
    base: dict[str, object],
    overrides: dict[str, object],
  ) -> dict[str, object]:
    merged = deepcopy(base)
    for key, value in overrides.items():
      if (
        isinstance(value, dict)
        and isinstance(merged.get(key), dict)
      ):
        merged[key] = self._merge_config(merged[key], value)  # type: ignore[arg-type]
      else:
        merged[key] = value
    return merged


class SimpleWorldModel(nn.Module):
  """A compact RSSM-style WMP representation learner.

  The original WMP code trains a Dreamer world model with an RSSM prior,
  posterior, KL free-nats, reconstruction heads, and reward prediction. This
  module keeps the MjLab runner interface small while moving the training target
  toward that structure.
  """

  def __init__(
    self,
    prop_dim: int,
    action_dim: int,
    depth_dim: int,
    *,
    feature_dim: int,
    hidden_dims: Sequence[int],
    activation: str = "elu",
    stoch_dim: int = 32,
    min_std: float = 0.1,
  ) -> None:
    super().__init__()
    self.prop_dim = prop_dim
    self.action_dim = action_dim
    self.depth_dim = depth_dim
    self.feature_dim = feature_dim
    self.stoch_dim = stoch_dim
    self.min_std = min_std
    encoder_dim = prop_dim + max(0, depth_dim)
    state_dim = feature_dim + stoch_dim

    self.encoder = build_mlp(
      encoder_dim,
      hidden_dims,
      feature_dim,
      activation,
      final_activation=nn.Tanh(),
    )
    self.feature_from_embed = build_mlp(
      feature_dim,
      hidden_dims,
      feature_dim,
      activation,
      final_activation=nn.Tanh(),
    )
    self.initial_deter = nn.Parameter(torch.zeros(1, feature_dim))
    self.img_in = build_mlp(
      stoch_dim + action_dim,
      hidden_dims,
      feature_dim,
      activation,
      final_activation=nn.Tanh(),
    )
    self.gru = nn.GRUCell(feature_dim, feature_dim)
    self.prior_stats = build_mlp(feature_dim, hidden_dims, 2 * stoch_dim, activation)
    self.post_stats = build_mlp(
      feature_dim + feature_dim,
      hidden_dims,
      2 * stoch_dim,
      activation,
    )
    self.prop_decoder = build_mlp(state_dim, hidden_dims, prop_dim, activation)
    self.next_prop_head = build_mlp(state_dim, hidden_dims, prop_dim, activation)
    self.reward_head = build_mlp(state_dim, hidden_dims, 1, activation)
    self.continue_head = build_mlp(state_dim, hidden_dims, 1, activation)
    self.depth_head = (
      build_mlp(state_dim, hidden_dims, depth_dim, activation)
      if depth_dim > 0
      else None
    )

  def features(
    self,
    prop: torch.Tensor,
    depth: torch.Tensor | None = None,
  ) -> torch.Tensor:
    embed = self._encode(prop, depth)
    return self.feature_from_embed(embed)

  def _zero_depth(self, prop: torch.Tensor) -> torch.Tensor:
    return torch.zeros(
      prop.shape[0],
      self.depth_dim,
      dtype=prop.dtype,
      device=prop.device,
    )

  def _encode(
    self,
    prop: torch.Tensor,
    depth: torch.Tensor | None,
  ) -> torch.Tensor:
    if self.depth_dim > 0:
      if depth is None or depth.numel() == 0:
        depth = self._zero_depth(prop)
      else:
        depth = depth.reshape(prop.shape[0], -1)
      x = torch.cat((prop, depth), dim=-1)
    else:
      x = prop
    return self.encoder(x)

  def _initial_state(
    self,
    batch_size: int,
    device: torch.device,
  ) -> dict[str, torch.Tensor]:
    deter = torch.tanh(self.initial_deter).repeat(batch_size, 1).to(device=device)
    stoch = torch.zeros(batch_size, self.stoch_dim, device=device)
    mean = torch.zeros(batch_size, self.stoch_dim, device=device)
    std = torch.ones(batch_size, self.stoch_dim, device=device)
    return {"deter": deter, "stoch": stoch, "mean": mean, "std": std}

  def _stats(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
    mean, std = torch.chunk(raw, 2, dim=-1)
    return {"mean": mean, "std": F.softplus(std) + self.min_std}

  def _dist(self, state: dict[str, torch.Tensor]) -> torch.distributions.Independent:
    return torch.distributions.Independent(
      torch.distributions.Normal(state["mean"], state["std"]),
      1,
    )

  def _mode(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
    return state["mean"]

  def _img_step(
    self,
    prev_state: dict[str, torch.Tensor],
    action: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    x = torch.cat((prev_state["stoch"], action), dim=-1)
    x = self.img_in(x)
    deter = self.gru(x, prev_state["deter"])
    stats = self._stats(self.prior_stats(deter))
    stoch = self._dist(stats).rsample()
    return {"deter": deter, "stoch": stoch, **stats}

  def _obs_step(
    self,
    prior: dict[str, torch.Tensor],
    embed: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    stats = self._stats(self.post_stats(torch.cat((prior["deter"], embed), dim=-1)))
    stoch = self._dist(stats).rsample()
    return {"deter": prior["deter"], "stoch": stoch, **stats}

  def _feat(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat((state["stoch"], state["deter"]), dim=-1)

  def _detach_state(
    self,
    state: dict[str, torch.Tensor],
  ) -> dict[str, torch.Tensor]:
    return {key: value.detach() for key, value in state.items()}

  def _kl_loss(
    self,
    post: dict[str, torch.Tensor],
    prior: dict[str, torch.Tensor],
    *,
    kl_free: float,
    dyn_scale: float,
    rep_scale: float,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    post_dist = self._dist(post)
    prior_dist = self._dist(prior)
    rep_loss = torch.distributions.kl_divergence(
      post_dist,
      self._dist(self._detach_state(prior)),
    )
    dyn_loss = torch.distributions.kl_divergence(
      self._dist(self._detach_state(post)),
      prior_dist,
    )
    kl_value = torch.distributions.kl_divergence(post_dist, prior_dist)
    rep_loss = torch.clamp(rep_loss, min=kl_free)
    dyn_loss = torch.clamp(dyn_loss, min=kl_free)
    return dyn_scale * dyn_loss + rep_scale * rep_loss, kl_value, dyn_loss, rep_loss

  def _sequence_inputs(
    self,
    prop: torch.Tensor,
    action: torch.Tensor,
    next_prop: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    depth: torch.Tensor | None,
  ) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
  ]:
    if prop.dim() == 2:
      prop = prop.unsqueeze(0)
    if action.dim() == 2:
      action = action.unsqueeze(0)
    if next_prop.dim() == 2:
      next_prop = next_prop.unsqueeze(0)
    if reward.dim() == 1:
      reward = reward.unsqueeze(0)
    if done.dim() == 1:
      done = done.unsqueeze(0)
    if depth is not None and depth.dim() == 2:
      depth = depth.unsqueeze(0)
    return prop, action, next_prop, reward, done, depth

  def loss(
    self,
    prop: torch.Tensor,
    action: torch.Tensor,
    next_prop: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    depth: torch.Tensor | None,
    *,
    kl_free: float = 1.0,
    dyn_scale: float = 0.5,
    rep_scale: float = 0.1,
    prop_loss_scale: float = 1.0,
    recon_loss_scale: float = 1.0,
    reward_loss_scale: float,
    continue_loss_scale: float,
    depth_loss_scale: float,
    latent_l2_scale: float,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prop, action, next_prop, reward, done, depth = self._sequence_inputs(
      prop,
      action,
      next_prop,
      reward,
      done,
      depth,
    )
    time_steps, batch_size = prop.shape[:2]
    device = prop.device
    state = self._initial_state(batch_size, device)
    zero_action = torch.zeros(batch_size, self.action_dim, device=device)
    metrics: dict[str, list[torch.Tensor]] = {
      "prop_loss": [],
      "recon_prop_loss": [],
      "reward_loss": [],
      "continue_loss": [],
      "depth_loss": [],
      "latent_l2": [],
      "kl": [],
      "dyn_loss": [],
      "rep_loss": [],
      "prior_ent": [],
      "post_ent": [],
    }

    for step in range(time_steps):
      prev_action = zero_action if step == 0 else action[step - 1]
      if step == 0:
        is_first = torch.ones(batch_size, 1, dtype=prop.dtype, device=device)
      else:
        is_first = done[step - 1].reshape(batch_size, 1).float()
      if torch.any(is_first > 0):
        init = self._initial_state(batch_size, device)
        state = {
          key: value * (1.0 - is_first) + init[key] * is_first
          for key, value in state.items()
        }
        prev_action = prev_action * (1.0 - is_first)

      step_depth = None if depth is None else depth[step].reshape(batch_size, -1)
      embed = self._encode(prop[step], step_depth)
      prior = self._img_step(state, prev_action)
      post = self._obs_step(prior, embed)
      post_feat = self._feat(post)
      trans_prior = self._img_step(post, action[step])
      trans_feat = self._feat(trans_prior)

      prop_pred = self.next_prop_head(trans_feat)
      recon_prop = self.prop_decoder(post_feat)
      reward_pred = self.reward_head(trans_feat).squeeze(-1)
      continue_pred = self.continue_head(trans_feat).squeeze(-1)
      target_continue = 1.0 - done[step].float()
      kl_loss, kl_value, dyn_loss, rep_loss = self._kl_loss(
        post,
        prior,
        kl_free=kl_free,
        dyn_scale=dyn_scale,
        rep_scale=rep_scale,
      )

      metrics["prop_loss"].append(
        F.mse_loss(prop_pred, next_prop[step], reduction="none").mean(dim=-1)
      )
      metrics["recon_prop_loss"].append(
        F.mse_loss(recon_prop, prop[step], reduction="none").mean(dim=-1)
      )
      metrics["reward_loss"].append(
        F.mse_loss(reward_pred, reward[step], reduction="none")
      )
      metrics["continue_loss"].append(
        F.binary_cross_entropy_with_logits(
          continue_pred,
          target_continue,
          reduction="none",
        )
      )
      if self.depth_head is not None and step_depth is not None and step_depth.numel():
        depth_pred = self.depth_head(post_feat)
        metrics["depth_loss"].append(
          F.mse_loss(depth_pred, step_depth, reduction="none").mean(dim=-1)
        )
      else:
        metrics["depth_loss"].append(torch.zeros(batch_size, device=device))
      metrics["latent_l2"].append(torch.square(post_feat).mean(dim=-1))
      metrics["kl"].append(kl_value)
      metrics["dyn_loss"].append(dyn_loss)
      metrics["rep_loss"].append(rep_loss)
      metrics["prior_ent"].append(self._dist(prior).entropy())
      metrics["post_ent"].append(self._dist(post).entropy())
      state = post

    reduced = {
      key: torch.stack(value, dim=0).mean()
      for key, value in metrics.items()
    }

    total = (
      prop_loss_scale * reduced["prop_loss"]
      + recon_loss_scale * reduced["recon_prop_loss"]
      + reward_loss_scale * reduced["reward_loss"]
      + continue_loss_scale * reduced["continue_loss"]
      + depth_loss_scale * reduced["depth_loss"]
      + reduced["dyn_loss"] * dyn_scale
      + reduced["rep_loss"] * rep_scale
      + latent_l2_scale * reduced["latent_l2"]
    )
    return total, {key: value.detach() for key, value in reduced.items()}


class DepthPredictor(nn.Module):
  def __init__(
    self,
    height_dim: int,
    prop_dim: int,
    depth_dim: int,
    *,
    hidden_dims: Sequence[int],
    activation: str = "elu",
  ) -> None:
    super().__init__()
    self.height_dim = height_dim
    self.prop_dim = prop_dim
    self.depth_dim = depth_dim
    self.net = (
      build_mlp(height_dim + prop_dim, hidden_dims, depth_dim, activation)
      if height_dim > 0 and depth_dim > 0
      else None
    )

  def forward(
    self, height_map: torch.Tensor, prop: torch.Tensor | None = None
  ) -> torch.Tensor:
    if self.net is None:
      return torch.empty(
        height_map.shape[0],
        0,
        dtype=height_map.dtype,
        device=height_map.device,
      )
    if self.prop_dim > 0:
      if prop is None:
        raise ValueError("DepthPredictor requires proprioception input.")
      x = torch.cat((height_map, prop), dim=-1)
    else:
      x = height_map
    return self.net(x)
