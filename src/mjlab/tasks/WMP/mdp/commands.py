from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat, wrap_to_pi

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class WmpVelocityCommand(CommandTerm):
  cfg: WmpVelocityCommandCfg

  def __init__(self, cfg: WmpVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs,
      dtype=torch.bool,
      device=self.device,
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)
    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None
    self._rough_flat_type_ids = self._resolve_rough_flat_type_ids()

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2],
        dim=-1,
      )
      / max_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
      self.vel_command_b[env_ids, 2] = 0.0
    else:
      self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

    rough_flat_ids = self._filter_rough_flat_envs(env_ids)
    if len(rough_flat_ids) > 0:
      flat_ranges = self.cfg.flat_ranges
      self.vel_command_b[rough_flat_ids, 0] = r.new_empty(len(rough_flat_ids)).uniform_(
        *flat_ranges.lin_vel_x
      )
      self.vel_command_b[rough_flat_ids, 1] = r.new_empty(len(rough_flat_ids)).uniform_(
        *flat_ranges.lin_vel_y
      )
      self.vel_command_b[rough_flat_ids, 2] = r.new_empty(len(rough_flat_ids)).uniform_(
        *flat_ranges.ang_vel_z
      )
      self.is_heading_env[rough_flat_ids] = False

    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
    obstacle_ids = self._filter_obstacle_envs(env_ids)
    forward_ids = obstacle_ids[
      r.new_empty(len(obstacle_ids)).uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
    ]
    if len(forward_ids) > 0:
      self.vel_command_b[forward_ids, 0] = (
        self.vel_command_b[forward_ids, 0]
        .abs()
        .clamp(min=self.cfg.min_forward_velocity)
      )
      self.vel_command_b[forward_ids, 1:] = 0.0
    moving_norm = torch.norm(self.vel_command_b[env_ids, :2], dim=1)
    self.vel_command_b[env_ids, :2] *= (moving_norm > 0.2).unsqueeze(1)

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * heading_error[env_ids],
        self.cfg.ranges.ang_vel_z[0],
        self.cfg.ranges.ang_vel_z[1],
      )
    standing_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_ids] = 0.0

  def _resolve_rough_flat_type_ids(self) -> torch.Tensor:
    terrain = self._env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
      return torch.empty(0, dtype=torch.long, device=self.device)
    terrain_generator = terrain.cfg.terrain_generator
    if terrain_generator is None:
      return torch.empty(0, dtype=torch.long, device=self.device)
    ids = [
      index
      for index, name in enumerate(terrain_generator.sub_terrains)
      if name.startswith(self.cfg.rough_flat_terrain_prefix)
    ]
    return torch.tensor(ids, dtype=torch.long, device=self.device)

  def _filter_rough_flat_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
    terrain = self._env.scene.terrain
    if (
      terrain is None
      or terrain.terrain_origins is None
      or self._rough_flat_type_ids.numel() == 0
    ):
      return torch.empty(0, dtype=torch.long, device=self.device)
    mask = torch.isin(terrain.terrain_types[env_ids], self._rough_flat_type_ids)
    return env_ids[mask]

  def _filter_obstacle_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
    rough_flat_ids = self._filter_rough_flat_envs(env_ids)
    if len(rough_flat_ids) == 0:
      return env_ids
    mask = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
    for flat_id in rough_flat_ids:
      mask &= env_ids != flat_id
    return env_ids[mask]

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    del on_change, request_action
    from viser import Icon

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      sliders = []
      for label, max_val in (
        ("lin_vel_x", self.cfg.ranges.lin_vel_x[1]),
        ("lin_vel_y", max(abs(v) for v in self.cfg.ranges.lin_vel_y)),
        ("ang_vel_z", max(abs(v) for v in self.cfg.ranges.ang_vel_z)),
      ):
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )
        sliders.append(slider)
      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for slider in sliders:
          slider.value = 0.0

    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, slider in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = slider.value

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    cmds = self.command.cpu().numpy()
    base_pos = self.robot.data.root_link_pos_w.cpu().numpy()
    base_mat = matrix_from_quat(self.robot.data.root_link_quat_w).cpu().numpy()
    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset
    for env_id in env_indices:
      start = base_pos[env_id] + base_mat[env_id] @ np.array([0, 0, z_offset])
      lin_cmd_b = np.array([cmds[env_id, 0], cmds[env_id, 1], 0.0])
      yaw_cmd_b = np.array([0.0, 0.0, cmds[env_id, 2]])

      visualizer.add_arrow(
        start,
        start + base_mat[env_id] @ (lin_cmd_b * scale),
        color=(0.2, 0.2, 0.8, 0.7),
        width=0.015,
      )
      visualizer.add_arrow(
        start,
        start + base_mat[env_id] @ (yaw_cmd_b * scale),
        color=(0.2, 0.7, 0.2, 0.7),
        width=0.015,
      )


@dataclass(kw_only=True)
class WmpVelocityCommandCfg(CommandTermCfg):
  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 0.0
  rel_forward_envs: float = 0.8
  min_forward_velocity: float = 0.2

  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]
    heading: tuple[float, float] | None = None

  ranges: Ranges
  flat_ranges: Ranges
  rough_flat_terrain_prefix: str = "rough_flat"

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> WmpVelocityCommand:
    return WmpVelocityCommand(self, env)

  def __post_init__(self) -> None:
    if self.heading_command and self.ranges.heading is None:
      raise ValueError("heading_command=True requires ranges.heading.")
