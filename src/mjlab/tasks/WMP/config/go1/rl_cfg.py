from dataclasses import replace

from mjlab.tasks.WMP.rl.config import WmpRunnerCfg


def unitree_go1_wmp_runner_cfg() -> WmpRunnerCfg:
  return WmpRunnerCfg(
    experiment_name="go1_wmp",
    run_name="rough",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=10_000,
    clip_actions=6.0,
  )


def unitree_go1_wmp_stairs_only_runner_cfg() -> WmpRunnerCfg:
  cfg = unitree_go1_wmp_runner_cfg()
  cfg.run_name = "stairs_only_no_amp"
  cfg.clip_actions = 2.0
  cfg.policy = replace(
    cfg.policy,
    init_std=0.3,
    min_std=0.05,
    max_std=1.0,
  )
  cfg.algorithm = replace(
    cfg.algorithm,
    entropy_coef=0.001,
  )
  cfg.amp = replace(
    cfg.amp,
    expert_motion_files=(),
    reward_scale=0.0,
    updates_per_iteration=0,
    diagnostics_enabled=False,
  )
  return cfg
