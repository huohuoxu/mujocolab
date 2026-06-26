"""Tests for a1_constants.py."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.unitree_a1 import a1_constants
from mjlab.entity import Entity
from mjlab.utils.string import resolve_expr


@pytest.fixture(scope="module")
def a1_entity() -> Entity:
  return Entity(a1_constants.get_a1_robot_cfg())


@pytest.fixture(scope="module")
def a1_model(a1_entity: Entity) -> mujoco.MjModel:
  return a1_entity.spec.compile()


def test_a1_compiles_as_floating_base_quadruped(a1_entity, a1_model) -> None:
  assert a1_model.nq == 19
  assert a1_model.nv == 18
  assert a1_model.nu == 12
  assert a1_entity.num_actuators == 12
  assert a1_entity.num_joints == 12
  assert a1_entity.is_actuated
  assert not a1_entity.is_fixed_base
  assert a1_model.joint("floating_base_joint").type[0] == mujoco.mjtJoint.mjJNT_FREE


def test_a1_actuator_parameters(a1_model) -> None:
  for i in range(a1_model.nu):
    actuator = a1_model.actuator(i)
    actuator_name = actuator.name
    assert actuator.gainprm[0] == 40.0
    assert actuator.biasprm[1] == -40.0
    assert actuator.biasprm[2] == -1.0
    if re.match(".*_hip_joint", actuator_name):
      assert actuator.forcerange[0] == -20.0
      assert actuator.forcerange[1] == 20.0
    else:
      assert actuator.forcerange[0] == -30.0
      assert actuator.forcerange[1] == 30.0


def test_a1_keyframe_joint_positions(a1_entity, a1_model) -> None:
  key = a1_model.key("init_state")
  expected_joint_pos = a1_constants.INIT_STATE.joint_pos
  assert expected_joint_pos is not None
  expected_values = resolve_expr(expected_joint_pos, a1_entity.joint_names, 0.0)
  for joint_name, expected_value in zip(
    a1_entity.joint_names, expected_values, strict=True
  ):
    joint = a1_model.joint(joint_name)
    qpos_idx = joint.qposadr[0]
    np.testing.assert_allclose(key.qpos[qpos_idx], expected_value, rtol=1e-5)


def test_a1_required_wmp_names_exist(a1_model) -> None:
  for body_name in ("trunk", "FR_hip", "FL_hip", "RR_hip", "RL_hip"):
    assert a1_model.body(body_name).id >= 0
  for site_name in ("FR", "FL", "RR", "RL"):
    assert a1_model.site(site_name).id >= 0
  for geom_name in (
    "trunk_collision",
    "head_collision",
    "FR_foot_collision",
    "FL_foot_collision",
    "RR_foot_collision",
    "RL_foot_collision",
    "FR_thigh_collision1",
    "FR_calf_collision1",
  ):
    assert a1_model.geom(geom_name).id >= 0


def test_a1_foot_collision_geoms(a1_model) -> None:
  foot_pattern = r"^[FR][LR]_foot_collision$"
  foot_geoms = []
  for i in range(a1_model.ngeom):
    geom = a1_model.geom(i)
    if re.match(foot_pattern, geom.name):
      foot_geoms.append(geom.name)
      assert geom.condim == 6
      assert geom.priority == 1
      assert geom.friction[0] == 1.0
  assert len(foot_geoms) == 4
