from __future__ import annotations

import math
from dataclasses import replace

from mjlab.asset_zoo.robots import GO1_ACTION_SCALE, get_go1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  CameraSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.WMP import mdp
from mjlab.tasks.WMP.mdp import WmpVelocityCommandCfg
from mjlab.tasks.WMP.terrains import WMP_PLAY_TERRAINS_CFG, WMP_ROUGH_TERRAINS_CFG
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def _quat_from_matrix(rows: tuple[tuple[float, float, float], ...]):
  m00, m01, m02 = rows[0]
  m10, m11, m12 = rows[1]
  m20, m21, m22 = rows[2]
  trace = m00 + m11 + m22
  if trace > 0.0:
    s = math.sqrt(trace + 1.0) * 2.0
    quat = (
      0.25 * s,
      (m21 - m12) / s,
      (m02 - m20) / s,
      (m10 - m01) / s,
    )
  elif m00 > m11 and m00 > m22:
    s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
    quat = (
      (m21 - m12) / s,
      0.25 * s,
      (m01 + m10) / s,
      (m02 + m20) / s,
    )
  elif m11 > m22:
    s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
    quat = (
      (m02 - m20) / s,
      (m01 + m10) / s,
      0.25 * s,
      (m12 + m21) / s,
    )
  else:
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    quat = (
      (m10 - m01) / s,
      (m02 + m20) / s,
      (m12 + m21) / s,
      0.25 * s,
    )
  norm = math.sqrt(sum(value * value for value in quat))
  return tuple(value / norm for value in quat)


def _camera_quat_forward_down(degrees: float) -> tuple[float, float, float, float]:
  down = math.radians(degrees)
  cos_down = math.cos(down)
  sin_down = math.sin(down)
  # MuJoCo cameras look along local -Z.  Go1 trunk +X is forward and +Y is left,
  # so local +X is robot-right (-Y), while local -Z points forward and down.
  x_axis = (0.0, -1.0, 0.0)
  y_axis = (sin_down, 0.0, cos_down)
  z_axis = (-cos_down, 0.0, sin_down)
  return _quat_from_matrix(
    (
      (x_axis[0], y_axis[0], z_axis[0]),
      (x_axis[1], y_axis[1], z_axis[1]),
      (x_axis[2], y_axis[2], z_axis[2]),
    )
  )


def make_wmp_go1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="trunk", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
  )
  forward_scan = RayCastSensorCfg(
    name="forward_height_scan",
    frame=ObjRef(type="body", name="trunk", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(2.0, 2.4), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
  )
  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=tuple(
      ObjRef(type="site", name=name, entity="robot")
      for name in ("FR", "FL", "RR", "RL")
    ),
    ray_alignment="yaw",
    pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=4),
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=False,
  )
  depth_camera = CameraSensorCfg(
    name="front_depth_camera",
    parent_body="robot/trunk",
    pos=(0.27, 0.0, 0.03),
    quat=_camera_quat_forward_down(5.0),
    fovy=58.0,
    width=64,
    height=64,
    data_types=("depth",),
    use_textures=False,
    use_shadows=False,
    enabled_geom_groups=(0, 1, 2),
  )

  feet = ("FR", "FL", "RR", "RL")
  foot_geoms = tuple(f"{name}_foot_collision" for name in feet)
  thigh_geoms = tuple(f"{leg}_thigh_collision{i}" for leg in feet for i in (1, 2, 3))
  calf_geoms = tuple(f"{leg}_calf_collision{i}" for leg in feet for i in (1, 2))

  feet_contact = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=foot_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  thigh_contact = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(mode="geom", pattern=thigh_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  shank_contact = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(mode="geom", pattern=calf_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  trunk_contact = ContactSensorCfg(
    name="trunk_ground_touch",
    primary=ContactMatch(
      mode="geom",
      pattern=("trunk_collision", "head_collision"),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
      scale=(0.25, 0.25, 0.25),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(func=mdp.command, params={"command_name": "twist"}),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
      scale=0.05,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  critic_terms = {
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    **actor_terms,
    "height_scan": ObservationTermCfg(
      func=mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=5.0,
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"sensor_name": "foot_height_scan"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "wm_prop": ObservationGroupCfg(
      terms={
        "prop": ObservationTermCfg(
          func=mdp.wm_prop,
          params={"command_name": "twist"},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "wm_depth": ObservationGroupCfg(
      terms={
        "depth": ObservationTermCfg(
          func=mdp.depth_image,
          params={
            "sensor_name": "front_depth_camera",
            "near_clip": 0.0,
            "far_clip": 2.0,
          },
        )
      },
      concatenate_terms=False,
      enable_corruption=False,
    ),
    "wm_forward_height_map": ObservationGroupCfg(
      terms={
        "height": ObservationTermCfg(
          func=mdp.forward_height_map,
          params={"sensor_name": "forward_height_scan", "base_height": 0.3},
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "amp": ObservationGroupCfg(
      terms={"amp": ObservationTermCfg(func=mdp.amp_observation)},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=GO1_ACTION_SCALE,
      use_default_offset=True,
    )
  }
  commands: dict[str, CommandTermCfg] = {
    "twist": WmpVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(10.0, 10.0),
      heading_command=True,
      heading_control_stiffness=0.5,
      rel_standing_envs=0.05,
      rel_heading_envs=1.0,
      rel_forward_envs=1.0,
      debug_vis=True,
      ranges=WmpVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 0.8),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-1.0, 1.0),
        heading=(0.0, 0.0),
      ),
      flat_ranges=WmpVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 0.8),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-1.0, 1.0),
        heading=(-math.pi / 4.0, math.pi / 4.0),
      ),
    )
  }

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05),
          "yaw": (-math.pi, math.pi),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(10.0, 15.0),
      params={
        "velocity_range": {
          "x": (-1.0, 1.0),
          "y": (-1.0, 1.0),
          "z": (0.0, 0.0),
          "roll": (0.0, 0.0),
          "pitch": (0.0, 0.0),
          "yaw": (-0.5, 0.5),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geoms),
        "operation": "abs",
        "axes": [0],
        "ranges": (0.5, 2.0),
        "shared_random": True,
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
        "operation": "add",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
  }

  rewards = {
    "tracking_lin_vel": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.5,
      params={"command_name": "twist", "std": 0.15},
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=0.5,
      params={"command_name": "twist", "std": 0.15},
    ),
    "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.0),
    "torques": RewardTermCfg(
      func=mdp.torques_l2,
      weight=-1.0e-4,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "dof_acc": RewardTermCfg(func=mdp.dof_acc_l2, weight=-2.5e-7),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.03),
    "dof_error": RewardTermCfg(func=mdp.dof_error_l2, weight=-0.04),
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.5,
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    ),
    "collision": RewardTermCfg(
      func=mdp.collision_cost,
      weight=-1.0,
      params={
        "sensor_names": ("thigh_ground_touch", "shank_ground_touch"),
        "force_threshold": 0.1,
      },
    ),
    "feet_stumble": RewardTermCfg(
      func=mdp.feet_stumble,
      weight=-0.1,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "feet_edge": RewardTermCfg(
      func=mdp.feet_edge,
      weight=-1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=feet, preserve_order=True),
      },
    ),
    "cheat": RewardTermCfg(func=mdp.cheat, weight=-1.0),
    "stuck": RewardTermCfg(
      func=mdp.stuck,
      weight=-1.0,
      params={"command_name": "twist"},
    ),
    "only_positive_clip": RewardTermCfg(
      func=mdp.only_positive_reward_clip,
      weight=1.0,
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "base_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": "trunk_ground_touch"},
    ),
    "out_of_terrain_bounds": TerminationTermCfg(
      func=mdp.out_of_terrain_bounds,
      time_out=True,
    ),
  }

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_wmp,
      params={"command_name": "twist"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (0.0, 0.8), "ang_vel_z": (-1.0, 1.0)},
          {"step": 5000 * 24, "lin_vel_x": (0.0, 1.0)},
        ],
      },
    ),
  }

  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    )
  }

  terrain_cfg = WMP_PLAY_TERRAINS_CFG if play else WMP_ROUGH_TERRAINS_CFG
  if play:
    events.pop("push_robot", None)
    curriculum = {}

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(terrain_cfg),
        max_init_terrain_level=0,
      ),
      entities={"robot": get_go1_robot_cfg()},
      sensors=(
        terrain_scan,
        forward_scan,
        foot_height_scan,
        depth_camera,
        feet_contact,
        thigh_contact,
        shank_contact,
        trunk_contact,
      ),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="trunk",
      distance=2.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=80,
      njmax=3000,
      contact_sensor_maxmatch=500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=200,
        cone="elliptic",
        impratio=10,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
