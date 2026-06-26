from mjlab.asset_zoo.robots import A1_ACTION_SCALE, get_a1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.WMP.wmp_env_cfg import make_wmp_env_cfg


def unitree_a1_wmp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_wmp_env_cfg(
    play=play,
    robot_cfg_factory=get_a1_robot_cfg,
    action_scale=A1_ACTION_SCALE,
  )


def unitree_a1_wmp_stairs_only_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return make_wmp_env_cfg(
    play=play,
    terrain_profile="stairs_only",
    robot_cfg_factory=get_a1_robot_cfg,
    action_scale=A1_ACTION_SCALE,
  )
