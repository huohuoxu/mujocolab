from dataclasses import replace

from mjlab.tasks.WMP.rl.config import WmpRunnerCfg


def unitree_a1_wmp_runner_cfg() -> WmpRunnerCfg:
  cfg = WmpRunnerCfg(
    experiment_name="a1_wmp",
    run_name="rough",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=10_000,
    clip_actions=6.0,
  )
  cfg.amp = replace(
    cfg.amp,
    expert_joint_pos_scale=None,
    expert_joint_pos_bias=None,
  )
  return cfg


def unitree_a1_wmp_stairs_only_runner_cfg() -> WmpRunnerCfg:
  cfg = unitree_a1_wmp_runner_cfg()
  cfg.run_name = "stairs_only_amp"
  return cfg
