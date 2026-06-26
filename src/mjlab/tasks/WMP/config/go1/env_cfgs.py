from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.WMP.wmp_env_cfg import make_wmp_go1_env_cfg


def unitree_go1_wmp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return make_wmp_go1_env_cfg(play=play)


def unitree_go1_wmp_stairs_only_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_wmp_go1_env_cfg(play=play, terrain_profile="stairs_only")
  cfg.rewards["only_positive_clip"].params["min_reward"] = -5.0
  return cfg
