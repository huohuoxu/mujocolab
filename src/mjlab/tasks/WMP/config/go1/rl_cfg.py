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
