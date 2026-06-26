"""Unitree A1 constants for the WMP task."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

A1_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_a1" / "xmls" / "a1.xml"
)
assert A1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(A1_XML))


##
# Actuator config.
##

A1_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_joint",),
  stiffness=40.0,
  damping=1.0,
  effort_limit=20.0,
)
A1_THIGH_CALF_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_thigh_joint", ".*_calf_joint"),
  stiffness=40.0,
  damping=1.0,
  effort_limit=30.0,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.35),
  joint_pos={
    "FR_hip_joint": -0.1,
    "FL_hip_joint": 0.1,
    "RR_hip_joint": -0.1,
    "RL_hip_joint": 0.1,
    "FR_thigh_joint": 0.8,
    "FL_thigh_joint": 0.8,
    "RR_thigh_joint": 1.0,
    "RL_thigh_joint": 1.0,
    ".*_calf_joint": -1.5,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision.*",),
  solref=(0.01, 1),
  condim={_foot_regex: 6, ".*_collision.*": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (1.0, 5e-3, 5e-4)},
)

##
# Final config.
##

A1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    A1_HIP_ACTUATOR_CFG,
    A1_THIGH_CALF_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_a1_robot_cfg() -> EntityCfg:
  """Get a fresh A1 robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=A1_ARTICULATION,
  )


A1_ACTION_SCALE: float = 0.25


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_a1_robot_cfg())

  viewer.launch(robot.spec.compile())
