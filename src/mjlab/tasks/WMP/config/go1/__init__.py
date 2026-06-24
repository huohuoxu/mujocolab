from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.WMP.config.go1.env_cfgs import unitree_go1_wmp_rough_env_cfg
from mjlab.tasks.WMP.config.go1.rl_cfg import unitree_go1_wmp_runner_cfg
from mjlab.tasks.WMP.rl.runner import WMPRunner

register_mjlab_task(
  task_id="Mjlab-WMP-Rough-Unitree-Go1",
  env_cfg=unitree_go1_wmp_rough_env_cfg(),
  play_env_cfg=unitree_go1_wmp_rough_env_cfg(play=True),
  rl_cfg=unitree_go1_wmp_runner_cfg(),
  runner_cls=WMPRunner,
)
