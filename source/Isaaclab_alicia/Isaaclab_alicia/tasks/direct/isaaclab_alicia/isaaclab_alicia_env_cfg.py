from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# Absolute path to the converted USD asset.
_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "alicia_duo"))
_USD_PATH = os.path.join(_ASSETS_DIR, "usd", "alicia_duo.usd")


ALICIA_DUO_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=False,#是否启用接触传感器
        # ===== 刚体物理属性 =====
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        # ===== 关节链（Articulation）属性 =====
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  #是否启用自碰撞
            solver_position_iteration_count=12,  #位置求解器迭代次数
            solver_velocity_iteration_count=1,  #速度求解器迭代次数
        ),
    ),
     # ===== 初始状态配置 =====
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  #机器人在世界坐标系中的位置（x, y, z）
        joint_pos={
            "Joint1": 0.0,
            "Joint2": 0.0,
            "Joint3": 0.0,
            "Joint4": 0.0,
            "Joint5": 0.0,
            "Joint6": 0.0,
            "left_finger": 0.0,  #夹爪左指的初始位置（通常为0表示张开）
            "right_finger": 0.0,  #夹爪右指的初始位置（通常为0表示张开）
        },
    ),
     # ===== 执行器（电机）配置 =====
    actuators={
        # ---------- 手臂执行器 ----------
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["Joint[1-6]"],
            effort_limit_sim=5.0,  #电机在仿真中的最大扭矩限制（单位：牛顿米）
            velocity_limit_sim=12.0,  #电机在仿真中的最大速度限制（单位：弧度/秒）
            stiffness=0.0,  #弹簧刚度（单位：牛顿米/弧度），0表示无弹簧力
            damping=0.5,  #阻尼系数（单位：牛顿米/弧度/秒），用于模拟关节的摩擦和能量耗散
        ),
        # Gripper: small return-to-centre spring (stiffness=20) so the
        # fingers don't drift asymmetrically under arm-induced vibrations
        # when Phase 1 leaves torque=0. The max spring force at full travel
        # is 20 * 0.05 = 1 N*m -- easily overpowered by the 5 N*m effort
        # limit, so Phase 2 grasp control is unaffected.
        # ---------- 夹爪执行器（Phase 1特殊处理）----------
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_finger", "right_finger"],
            effort_limit_sim=5.0,
            velocity_limit_sim=12.0,
            stiffness=20.0,
            damping=1.0,
        ),
    },
)


@configclass
class IsaaclabAliciaEnvCfg(DirectRLEnvCfg):  #继承自DirectRLEnvCfg，表示Isaaclab Alicia环境的配置类
    # ==================== 核心控制参数 ====================
    decimation = 2  #控制频率与仿真频率的比率。仿真频率为120 Hz，控制频率为60 Hz（120 / 2）。
    episode_length_s = 4.0  # 4s episode @ 60 Hz control -> 240 steps

    # ==================== 动作与观测空间 ====================
    action_space = 8
    # obs = relative_pos(3) + joint_pos(8) + joint_vel(8) + prev_action(8) + ee_vel(3) = 30
    observation_space = 30
    state_space = 0

    # ==================== 仿真配置 ====================
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,  #仿真时间步长为1/120秒，即仿真频率为120 Hz
        render_interval=decimation,  #每隔decimation个仿真步骤进行一次渲染，即每隔2个仿真步骤渲染一次，渲染频率为60 Hz
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",  #摩擦力的组合模式，"multiply"表示两个接触物体的摩擦系数相乘
            restitution_combine_mode="multiply",  #弹性恢复系数的组合模式，"multiply"表示两个接触物体的弹性恢复系数相乘
            static_friction=1.0,  #静摩擦系数，表示物体在开始滑动前的摩擦力大小
            dynamic_friction=1.0,  #动摩擦系数，表示物体在滑动时的摩擦力大小
            restitution=0.0,  #弹性恢复系数，表示物体碰撞后的反弹程度，0表示完全没有弹性（即不反弹）
        ),
    )

    # ==================== 机器人与场景 ====================
    robot_cfg: ArticulationCfg = ALICIA_DUO_CFG  #使用之前定义的ALICIA_DUO_CFG作为机器人配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.0, replicate_physics=True  #并行环境数：4096个环境同时训练,每个环境之间的间距为2.0米，物理属性在所有环境中复制（即每个环境中的物体具有相同的物理属性）
    )

    # ==================== 关节与末端执行器命名 ====================
    arm_joint_names = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]
    gripper_joint_names = ["left_finger", "right_finger"]

    # 末端执行器定义（URDF转换后的适配）
    ee_body_name = "Link6"  #末端执行器所在的链接名称
    ee_offset_local = (-0.0002, -0.0003, 0.10)  #末端执行器相对于Link6的局部偏移，单位为米。这个偏移将末端执行器定义在Link6的前方0.10米处，稍微向内侧（-0.0002, -0.0003）以便更好地抓取目标。

    # ---- action scaling ----
    # Policy outputs in [-1, 1]; multiply to get N*m. effort_limit_sim already
    # caps this in PhysX, so we choose 5.0 to match the joint effort limit.
    action_scale_arm = 5.0  #将策略输出的动作值（在[-1, 1]范围内）乘以5.0，以将其转换为实际的关节扭矩（单位：牛顿米）。这个缩放因子与之前定义的关节执行器的effort_limit_sim一致，确保动作值不会超过物理仿真中的最大扭矩限制。
    action_scale_gripper = 5.0  #同样地，将策略输出的夹爪动作值乘以5.0，以将其转换为实际的夹爪扭矩（单位：牛顿米）。这个缩放因子也与夹爪执行器的effort_limit_sim一致，确保夹爪动作值不会超过物理仿真中的最大扭矩限制。
    # Reach 阶段夹爪保持在中位，抑制由手臂加减速耦合引入的轻微抖动
    gripper_hold_kp = 35.0
    gripper_hold_kd = 3.0
    gripper_hold_torque_limit = 2.0
    # 到位自动闭合（默认关闭）：达到触发距离后直接给夹爪闭合指令。
    # 为避免阈值附近反复开合，使用 close/reopen 双阈值（迟滞）。
    enable_auto_gripper_close = False
    gripper_close_trigger_dist = 0.018
    gripper_reopen_trigger_dist = 0.028
    gripper_close_torque = 2.0

    # ==================== Sim-to-Real 鲁棒训练参数 ====================
    # 随机化总开关：默认关闭，先保证 reach baseline 收敛稳定；
    # 需要 sim2real 训练时再显式开启。
    enable_sim2real_randomization = False
    # 观测延迟：按环境随机取整帧数 [min, max]
    obs_latency_steps_min = 0
    obs_latency_steps_max = 2
    # 动作执行延迟：按环境随机取整帧数 [min, max]
    act_latency_steps_min = 0
    act_latency_steps_max = 1
    # 执行链路扰动：随机丢包 + 动作噪声（归一化动作空间）
    action_dropout_prob = 0.01
    action_noise_std = 0.02
    # 目标附近抑制随机扰动，避免“已到位仍被噪声推走”
    randomization_near_disable_dist = 0.04
    randomization_near_noise_scale = 0.2
    # 执行器增益随机化（用于模拟摩擦、模型误差和驱动衰减）
    actuator_arm_gain_range = (0.85, 1.15)
    actuator_gripper_gain_range = (0.90, 1.10)
    # 观测噪声与偏置（传感器噪声 + 零点偏差）
    obs_joint_pos_noise_std = 0.003
    obs_joint_vel_noise_std = 0.03
    obs_ee_vel_noise_std = 0.02
    obs_target_pos_noise_std = 0.002
    obs_joint_pos_bias_range = (-0.01, 0.01)
    obs_joint_vel_bias_range = (-0.08, 0.08)
    # 动力学随机化占位（默认关闭，防止依赖底层 API 版本差异）
    randomize_joint_friction = False
    joint_friction_scale_range = (0.9, 1.1)
    randomize_payload_mass = False
    payload_mass_delta_range = (-0.05, 0.05)

    # ==================== 目标采样参数 ====================
    # 在机械臂工作空间内采样目标位置（球壳分布）
    target_r_min = 0.18         # 最小半径0.18m（距基座）
    target_r_max = 0.35         # 最大半径0.35m（距基座）
    target_elev_min = -0.17     # 最小高度角[-0.17]（约-10度，略低于地平线）
    target_elev_max = 0.87      # 最大高度角[0.87]（约50度，高于地平线）
    target_center_z = 0.20      # 工作空间中心高度[0.20]（约肩部高度）
    target_z_min = 0.05         # 目标最小高度[0.05]（避免目标接触地面）

    # ==================== 奖励函数配置 ====================
     # 距离阈值定义
    success_threshold = 0.01    # 成功阈值：1cm（用于日志和奖励放大）
    close_threshold = 0.05      # 接近阈值：5cm

    # 基础惩罚（鼓励快速完成）
    rew_step_penalty = 0.001          # 每一步的基础惩罚，鼓励快速完成任务
    rew_dist_gain = 0.5               # 距离奖励增益，控制距离奖励的总体规模
    rew_dist_beta = 2.0               # 距离奖励的平滑度参数，较高的值使奖励在接近目标时更陡峭
    rew_close_multiplier = 1.5        # 接近奖励乘数，距离在close_threshold内时，距离奖励乘以这个因子，鼓励机器人更接近目标
    rew_success_multiplier = 8.0      # 成功奖励乘数，距离在success_threshold内时，距离奖励乘以这个因子，给予成功完成任务的显著奖励
    # 运动质量奖励
    rew_direction_gain = 0.05         # 方向奖励增益，控制方向奖励的总体规模。这个奖励鼓励机器人在接近目标时朝正确的方向移动。
    rew_speed_threshold = 1.0         # 速度奖励的速度阈值，单位为m/s。只有当末端执行器的速度小于这个阈值时，才会给予速度奖励。这有助于防止机器人以过快的速度接近目标，从而保持稳定和安全。
    rew_speed_penalty = 0.01          # 速度惩罚增益，控制速度惩罚的总体规模。这个惩罚鼓励机器人以适当的速度接近目标，避免过快或过慢。

    # 姿态对齐奖励（关键创新）
    rew_gripper_align_gain = 0.5      # 对齐奖励增益，控制对齐奖励的总体规模。这个奖励鼓励机器人将夹爪朝向目标，从而提高抓取成功率。

    # 接近阶段奖励：在接近阶段内，根据距离给予额外奖励，鼓励机器人更快地进入成功区域。距离越接近成功阈值，奖励越高。
    rew_close_phase_thresholds = (0.04, 0.03, 0.02, 0.01)
    rew_close_phase_rewards = (1.0, 2.0, 3.0, 5.0)

    # 平滑性惩罚
    rew_joint_vel_change = 0.001      # 关节速度变化惩罚（鼓励平滑）
    rew_action_rate = 0.01            # 相邻两步动作变化惩罚（抑制真机抖振）

    # 手腕正则化（软约束）
    rew_wrist_action = 0.05           # 手腕动作惩罚，鼓励手腕关节（J4, J5, J6）动作较小
    rew_wrist_pos = 0.15              # 手腕位置惩罚，鼓励手腕关节（J4, J5, J6）位置较小（即更伸展）
    rew_wrist_vel = 0.02              # 手腕速度惩罚，鼓励手腕关节（J4, J5, J6）速度较小

    # 夹爪对称性惩罚（替代URDF缺失的等式约束）
    rew_gripper_asymmetry = 5.0       # 左右手指不对称惩罚

    # 稳定奖励：当机械臂既处于成功阈值范围内，又近乎静止时，额外发放奖励。
    # 该奖励与其他正向奖励一同做了幅度缩减，避免其数值过大而掩盖正则化惩罚项的作用。
    rew_settle_speed_threshold = 0.05  # 稳定速度阈值（m/s）
    rew_settle_bonus = 4.0             # 稳定奖励（在成功阈值内且低速时）
    # 目标附近动作柔化：抑制到位后的过冲和抖动
    near_target_dist = 0.05
    near_target_action_scale = 0.35
    near_target_action_deadband = 0.03
    # 高于速度阈值时，按超量连续惩罚，避免极端大速度冲击
    rew_speed_excess = 0.01
