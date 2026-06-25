"""WMP Go1 粗糙地形任务的环境装配文件
这个文件只声明配置，不直接执行 reset、step 或 learn

真正运行时的链路是：
1. config/go1/__init__.py 调用本文件生成 ManagerBasedRlEnvCfg，并把它注册到任务表
2. train.py / play.py 通过任务 ID 取出配置，然后创建 ManagerBasedRlEnv
3. ManagerBasedRlEnv 根据这里的 scene、sensor、action、observation、reward、termination、curriculum 等配置实例化各个 manager
4. WMP 专用的 WMPRunner 从命名观测组中读取张量：actor、critic、world model、depth 和 AMP

因此阅读这个文件时，可以把它当成 WMP 任务的 总装配图：
它决定环境中有什么、策略看到什么、critic 额外知道什么、奖励怎么塑形、失败条件是什么，以及训练和 play 模式有哪些差异。
"""

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
  """把 3x3 旋转矩阵转换成 MuJoCo 使用的单位四元数。
  MuJoCo 的四元数顺序是 (w, x, y, z)
  这里单独写一个轻量 helper，是为了在配置文件内部完成相机姿态设置，避免额外引入矩阵/姿态依赖
  """
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
  """构造安装在 trunk 上、朝前并略微向下看的深度相机四元数
  Go1 trunk 坐标系近似为：+X 向前、+Y 向左、+Z 向上
  MuJoCo 相机默认沿自身局部 -Z 方向看出去，这里把相机局部 -Z 映射到机器人坐标系的 向前并下俯一点
  """
  down = math.radians(degrees)
  cos_down = math.cos(down)
  sin_down = math.sin(down)
  # MuJoCo 相机沿局部 -Z 方向成像。Go1 trunk 的 +X 是前方、+Y 是左方，
  # 因此这里令相机局部 +X 指向机器人右侧，也就是 trunk 坐标系的 -Y。
  # 同时让局部 -Z 指向前方并带一点向下俯角。
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
  """创建 WMP + Go1 的完整 manager-based 环境配置

  Args:
    play: 为 True 时进入评估/可视化模式：使用 play 地形集合、关闭随机推扰，并关闭课程学习
          训练模式则保留扰动和课程，以提升鲁棒性

  Returns:
    一个已经装配好的 ManagerBasedRlEnvCfg
    这个 cfg 后续由 ManagerBasedRlEnv 消费，本函数本身不执行仿真
  """
  # ---------------------------------------------------------------------------
  # 1. Sensors：传感器配置
  # ---------------------------------------------------------------------------
  # 传感器先注册到 Scene 中，后面的 observation、reward、termination 会通过 `sensor_name` 字符串引用它们
  # - terrain_scan -> critic.height_scan，用作 critic 的局部地形特权信息。
  # - forward_height_scan -> wm_forward_height_map.height，送给 depth predictor。
  # - foot_height_scan -> critic.foot_height，给 critic 足端附近地面高度。
  # - front_depth_camera -> wm_depth.depth，作为 WMP world model 的视觉输入。
  # - feet_ground_contact -> critic 接触观测 + 多个足端奖励项。
  # - thigh/shank_ground_touch -> collision reward，惩罚非足端触地。
  # - trunk_ground_touch -> base_contact termination，检测摔倒/机身撞地。

  # terrain_scan：trunk 周围的局部地形扫描
  # 采集内容：
  # - 以 robot/trunk 为参考坐标系，在 yaw 对齐的 1.6m x 1.0m 网格上发射向下/地形方向 ray，得到每条 ray 的命中位置和距离
  # 后续使用：
  # - 在 critic_terms["height_scan"] 中通过 mdp.height_scan(sensor_name="terrain_scan")读取
  #   转换成 trunk 高度 - 地面命中高度 的地形高度向量，并乘 scale=5.0
  # 用途：
  # - 这是 critic 的 privileged terrain 信息，帮助 value function 判断前后左右地形难度和未来风险
  # - actor 观测里故意不放它，避免部署策略依赖仿真中才容易获得的稠密地形真值
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

  # forward_height_scan：更宽的前向地形高度扫描
  # 采集内容：
  # - 以 trunk 为参考，在机器人前方 2.0m x 2.4m 网格上做 ray cast，得到前方更大范围的可通行几何轮廓
  # 后续使用：
  # - 在 observations["wm_forward_height_map"] 中通过 mdp.forward_height_map读取，得到名为 height 的 world-model 辅助输入
  # - WMPRunner._depth_for_world_model() 会把 tensors["height"] 和 wm_prop 送进 DepthPredictor，用来预测/刷新 world model 使用的 depth cache
  # 用途：
  # - 给 depth predictor 提供稠密、稳定的几何线索，让它学习从高度扫描预测 camera-depth 风格的表示
  # - 当真实深度相机只在部分 env 或较低频率更新时，用预测深度补齐其余 env
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

  # foot_height_scan：每只脚附近的地面高度探针
  # 采集内容：
  # - 在 FR/FL/RR/RL 四个 foot site 周围各发射一个小圆环 ray pattern，估计每只脚附近局部地面的高度
  # 后续使用：
  # - 在 critic_terms["foot_height"] 中通过 mdp.foot_height 读取
  # 用途：
  # - 给 critic 一个足端局部地形高度特权观测，使 value function 更容易判断当前落脚是否接近台阶、坑边、gap 等复杂区域。
  # - 目前它不直接进入 reward/termination，也不进入 actor。
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

  # depth_camera：安装在 trunk 前方的深度相机
  # 采集内容：
  # - 输出 64x64 depth 图像，视场角 fovy=58，并向前下方俯视 5 度
  # - 只渲染 depth，不使用纹理和阴影，降低渲染成本并突出几何信息
  # 后续使用：
  # - 在 observations["wm_depth"] 中通过 mdp.depth_image 读取，裁剪到near_clip=0.0、far_clip=2.0 后归一化
  # - WMPRunner._depth_for_world_model() 把它作为真实 raw_depth：
  #   一方面可直接进入 world_model.features()，另一方面作为 depth_target 训练 DepthPredictor
  # 用途：
  # - 这是 WMP 视觉腿式运动 的核心视觉输入，world model 用它形成地形感知特征，ActorCriticWMP 再利用这些特征决策
  # - play/viser 中也可以查看 front_depth_camera_depth，用于确认相机方向和深度图
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

  # Go1 XML 中 site/geom 使用的腿部命名。
  # 这里统一维护 FR、FL、RR、RL 顺序，保证接触传感器、足端 site、
  # edge reward 中的 site_names 和 critic 接触观测按同一条腿序对齐。
  feet = ("FR", "FL", "RR", "RL")
  foot_geoms = tuple(f"{name}_foot_collision" for name in feet)
  thigh_geoms = tuple(f"{leg}_thigh_collision{i}" for leg in feet for i in (1, 2, 3))
  calf_geoms = tuple(f"{leg}_calf_collision{i}" for leg in feet for i in (1, 2))

  # feet_contact：足端与地形的接触传感器。
  # 采集内容：
  # - primary 是四个 foot collision geom，secondary 是 terrain。
  # - fields=("found", "force") 表示记录是否接触和接触力。
  # - reduce="netforce" 表示每只脚聚合为净接触力。
  # - track_air_time=True 会维护每只脚当前离地时长和 first-contact 事件。
  # 后续使用：
  # - critic_terms["foot_contact"] 读取 found，给 critic 接触状态。
  # - critic_terms["foot_contact_forces"] 读取 force，给 critic 接触力大小/方向。
  # - reward["feet_air_time"] 用 current_air_time 和 first_contact 奖励合理摆腿。
  # - reward["feet_stumble"] 用接触力横向/竖向比例惩罚绊脚。
  # - reward["feet_edge"] 用 found 结合 terrain edge mask，惩罚踩在障碍边缘。
  # 用途：
  # - 这是 gait shaping 的核心信号，同时也给 critic 提供“脚是否真正落地”
  #   以及“落地受力是否异常”的特权信息。
  feet_contact = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=foot_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  # thigh_contact：大腿碰到地形的接触传感器。
  # 采集内容：
  # - 监测每条腿 thigh_collision1/2/3 与 terrain 的接触。
  # - history_length=4 保存最近几个 step 的接触历史，使短暂撞击也能被捕捉。
  # 后续使用：
  # - reward["collision"] 通过 mdp.collision_cost 同时读取 thigh_ground_touch
  #   和 shank_ground_touch。
  # 用途：
  # - 大腿触地通常表示跨障碍失败、身体姿态过低或腿部撞到台阶/坑边；
  #   这里用负奖励压制它，但不直接 termination。
  thigh_contact = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(mode="geom", pattern=thigh_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  # shank_contact：小腿/胫部碰到地形的接触传感器。
  # 采集内容：
  # - 监测每条腿 calf_collision1/2 与 terrain 的接触。
  # 后续使用：
  # - 同样被 reward["collision"] 消费。
  # 用途：
  # - 惩罚小腿刮蹭、拖腿、撞台阶等不健康动作，促使策略真正抬脚跨越障碍。
  shank_contact = ContactSensorCfg(
    name="shank_ground_touch",
    primary=ContactMatch(mode="geom", pattern=calf_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  # trunk_contact：机身/头部与地形的接触传感器。
  # 采集内容：
  # - 监测 trunk_collision 和 head_collision 是否与 terrain 接触。
  # - history_length=4 能覆盖短时间撞击，减少漏检。
  # 后续使用：
  # - terminations["base_contact"] 通过 mdp.illegal_contact 读取该传感器。
  # 用途：
  # - trunk/head 触地基本代表摔倒、翻滚或身体撞上障碍，因此作为真正失败
  #   条件终止 episode，而不是简单给负奖励。
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

  # ---------------------------------------------------------------------------
  # 2. Observations：观测组配置
  # ---------------------------------------------------------------------------
  # WMP runner 依赖多个命名观测组。改组名时要非常小心：
  # rl/config.py 和 WMPRunner._obs_to_tensors() 会把这些名字映射到actor、critic、world model、depth predictor、AMP 的输入

  # actor 观测：本体感知 + 速度命令 + 上一步动作
  # 不包含真实 base 线速度、接触力、稠密地形扫描等 privileged 信息，更接近真实部署时可获得的输入。训练时是否加噪由 group cfg 控制
  # 注意这里的 actor_terms 只是 环境侧 actor 观测组，不是 actor 网络的完整最终输入
  # World model 从 depth 里提取的 h_t 是在 WMPRunner 里算出来，再作为单独参数喂给 ActorCriticWMP
  # 所以不会出现在 当前的 actor_terms 里
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

  # critic 观测：actor terms 加上仿真/传感器特权状态。
  # value function 训练时可以看更丰富的信息，但这些信息不会暴露给部署策略。
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

  # 观测组会成为 env.reset()/env.step() 返回 obs 字典中的键。
  # concatenate_terms=True 的组会被拼成扁平向量；深度图组保持非拼接，
  # 这样 runner 可以单独处理图像/通道维度。
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
    # world model 使用的本体状态。它和 actor 观测分开，是因为 world model
    # 学的是状态转移/奖励/继续概率等动力学目标，而不是直接输出动作。
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
    # world model 的深度输入。这里保持 nested term，不拼成一维向量，
    # 是为了保留图像维度，方便后续网络处理。
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
    # 由 ray cast 得到的稠密几何目标。depth predictor 会学习从本体状态
    # 和前向高度图中预测与深度相机相关的表示。
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
    # AMP 观测是 discriminator 用来和 mocap transition 对比的状态向量。
    # 这里保持无噪声，避免专家/策略判别信号被观测扰动污染。
    "amp": ObservationGroupCfg(
      terms={"amp": ObservationTermCfg(func=mdp.amp_observation)},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  # ---------------------------------------------------------------------------
  # 3. Actions and commands：动作和命令
  # ---------------------------------------------------------------------------
  # 策略输出的是归一化动作。JointPositionActionCfg 会把动作映射为 Go1
  # 关节位置目标，大致形式为：
  #   target = default_joint_pos + action * GO1_ACTION_SCALE
  # 其中 GO1_ACTION_SCALE 来自资产库，会按不同关节/执行器尺度设置动作幅度。
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=GO1_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  # `twist` 是本任务追踪的速度命令，会同时进入 actor/critic 观测和速度奖励。
  # WMP 自定义 command term 支持 obstacle terrain 与 rough-flat terrain 使用
  # 不同采样范围，更贴近原始 WMP 的命令语义。
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

  # ---------------------------------------------------------------------------
  # 4. Events：reset 逻辑和 domain randomization
  # ---------------------------------------------------------------------------
  # EventManager 负责执行这些事件：
  # - mode="reset"：每次环境 reset 时执行。
  # - mode="interval"：训练过程中按时间间隔执行。
  # - mode="startup"：manager 创建后执行一次，常用于域随机化。
  events = {
    # reset 时随机化 base 初始位姿；velocity_range 为空表示速度保持默认值。
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
    # 每个 episode 从机器人默认关节姿态和零关节速度开始。
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # 随机外部速度扰动，用于提升鲁棒性。play 模式下会移除它，
    # 这样可视化评估时更容易判断策略本身表现。
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
    # startup 域随机化：为每个环境采样足端摩擦系数。
    # 这样策略不会过度依赖某一个固定接触参数。
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
    # startup 域随机化：扰动 trunk 质心位置。
    # 这是应对质量分布误差的一个简单鲁棒性开关。
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

  # ---------------------------------------------------------------------------
  # 5. Rewards：奖励项
  # ---------------------------------------------------------------------------
  # RewardManager 会逐项计算 reward term，乘以各自 weight，并默认按 env.step_dt
  # 缩放。正奖励主要鼓励速度追踪和合理步态；负奖励抑制不稳定、低效、
  # 碰撞、卡住和利用地形漏洞等行为。
  rewards = {
    # 追踪 base 坐标系下的线速度命令，主要是前进速度。
    "tracking_lin_vel": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.5,
      params={"command_name": "twist", "std": 0.15},
    ),
    # 追踪 yaw 角速度命令，或者 heading controller 生成的 yaw 速度目标。
    "tracking_ang_vel": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=0.5,
      params={"command_name": "twist", "std": 0.15},
    ),
    # 抑制 base 垂直方向速度，减少上下弹跳。
    "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.0),
    # 抑制执行器力矩、关节加速度、动作变化和关节偏离默认姿态。
    # 这些项共同让步态更平滑，也更接近硬件可承受的控制信号。
    "torques": RewardTermCfg(
      func=mdp.torques_l2,
      weight=-1.0e-4,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "dof_acc": RewardTermCfg(func=mdp.dof_acc_l2, weight=-2.5e-7),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.03),
    "dof_error": RewardTermCfg(func=mdp.dof_error_l2, weight=-0.04),
    # 当存在移动命令时，鼓励合理的摆腿/离地时间。
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.5,
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    ),
    # 惩罚大腿/小腿等非足端链接与地形接触。
    "collision": RewardTermCfg(
      func=mdp.collision_cost,
      weight=-1.0,
      params={
        "sensor_names": ("thigh_ground_touch", "shank_ground_touch"),
        "force_threshold": 0.1,
      },
    ),
    # 惩罚足端在障碍地形上的侧向绊倒式接触。
    "feet_stumble": RewardTermCfg(
      func=mdp.feet_stumble,
      weight=-0.1,
      params={"sensor_name": "feet_ground_contact"},
    ),
    # 惩罚脚直接踩在障碍边缘上。地形生成器会保存 edge metadata，
    # 这个 reward 读取这些 metadata，并叠加课程系数。
    "feet_edge": RewardTermCfg(
      func=mdp.feet_edge,
      weight=-1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=feet, preserve_order=True),
      },
    ),
    # 任务特定的反作弊/反投机项：
    # - cheat：避免策略利用 rough-flat 地形绕开障碍语义。
    # - stuck：有前进命令时不能原地不动。
    "cheat": RewardTermCfg(func=mdp.cheat, weight=-1.0),
    "stuck": RewardTermCfg(
      func=mdp.stuck,
      weight=-1.0,
      params={"command_name": "twist"},
    ),
    # 在 reward 函数层把总奖励裁剪到非负范围。
    # 这是很多 locomotion 任务常见的 reward shaping 技巧。
    "only_positive_clip": RewardTermCfg(
      func=mdp.only_positive_reward_clip,
      weight=1.0,
    ),
  }

  # ---------------------------------------------------------------------------
  # 6. Terminations：终止/截断条件
  # ---------------------------------------------------------------------------
  # TerminationManager 会返回两类 mask：terminated 和 time_out/truncated。
  # `time_out=True` 表示这是人为截断或时间限制，不一定是任务失败。
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # trunk/head 碰到地形是真失败，因为这通常意味着机身撞地或摔倒。
    "base_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": "trunk_ground_touch"},
    ),
    # 离开生成地形边界按 timeout 风格处理，而不是物理失败信号。
    "out_of_terrain_bounds": TerminationTermCfg(
      func=mdp.out_of_terrain_bounds,
      time_out=True,
    ),
  }

  # ---------------------------------------------------------------------------
  # 7. Curriculum and metrics：课程学习和指标
  # ---------------------------------------------------------------------------
  # Curriculum term 会在 reset 流程中执行。WMP 这里主要做两件事：
  # - 根据机器人在地形块上的前进情况调整 terrain level。
  # - 训练到一定环境步数后扩大速度命令范围。
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
          # 5000 个 runner iteration * 每个 iteration 24 个 rollout step。
          {"step": 5000 * 24, "lin_vel_x": (0.0, 1.0)},
        ],
      },
    ),
  }

  # MetricsManager 额外记录的自定义标量，主要用于日志和调试。
  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    )
  }

  # play 模式使用独立地形集合，关闭随机推扰，并冻结课程学习。
  # 这样可视化时环境变化更少，方便判断策略是否真的会走。
  terrain_cfg = WMP_PLAY_TERRAINS_CFG if play else WMP_ROUGH_TERRAINS_CFG
  if play:
    events.pop("push_robot", None)
    curriculum = {}

  # ---------------------------------------------------------------------------
  # 8. Final environment config：最终环境配置对象
  # ---------------------------------------------------------------------------
  # 这个对象是本文件真正对外输出的产品。ManagerBasedRlEnv 会根据它按固定顺序
  # 构建 Scene、Simulation，以及各类 manager。
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      # 使用 WMP terrain preset 生成程序化地形。`replace()` 用来复制配置，
      # 避免不同注册任务副本共享同一个可变 terrain generator 对象。
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(terrain_cfg),
        max_init_terrain_level=0,
      ),
      # 机器人本体来自 asset zoo。Go1 的 XML、执行器、默认姿态、site 和
      # collision geom 都在 unitree_go1 constants 中定义。
      entities={"robot": get_go1_robot_cfg()},
      # 把上面声明的所有传感器挂到 scene。传感器顺序不是语义 API，
      # 但集中列在这里便于核对依赖。
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
    # viewer 跟随 trunk，这样 native/viser 播放时机器人会保持在视野中心附近。
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="trunk",
      distance=2.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    # MuJoCo 求解器和接触相关设置。环境控制周期为：
    #   timestep * decimation = 0.005 * 4 = 0.02 s
    # 也就是策略以 50 Hz 频率输出动作。
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
    # 每个动作对应 4 个物理子步；每个 episode 最长 20 秒。
    decimation=4,
    episode_length_s=20.0,
  )
