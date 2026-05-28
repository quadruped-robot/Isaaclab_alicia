from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .isaaclab_alicia_env_cfg import IsaaclabAliciaEnvCfg

#定义目标点的可视化样式（绿色球体）
_TARGET_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/TargetSphere",  #可视化标记在USD场景中的路径
    markers={
        "target": sim_utils.SphereCfg(  #定义标记为一个球体
            radius=0.015,  #球体半径为0.015米
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),  #球体颜色为绿色（RGB: 0, 255, 0）
        ),
    },
)

# -------------------------- 主环境类 --------------------------
class IsaaclabAliciaEnv(DirectRLEnv):
    cfg: IsaaclabAliciaEnvCfg  #环境配置类，包含了环境的各种参数设置

    # ------------------------------------------------------------------ #
    # 场景初始化与缓冲区设置                                              #
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: IsaaclabAliciaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)  #调用父类的初始化方法，传入环境配置和渲染模式

        # 解析关节索引（按配置中的名称查找）
        # preserve_order=True 保持配置中的关节顺序（Joint1-Joint6）
        self._arm_dof_idx, _ = self.robot.find_joints(self.cfg.arm_joint_names, preserve_order=True)  
        self._gripper_dof_idx, _ = self.robot.find_joints(self.cfg.gripper_joint_names, preserve_order=True)
        self._all_dof_idx = self._arm_dof_idx + self._gripper_dof_idx  # 所有关节的索引列表（手臂关节 + 夹爪关节）
        self._ee_body_idx = self.robot.find_bodies(self.cfg.ee_body_name)[0][0]  #末端执行器所在刚体的索引（假设只有一个匹配项）

        # 末端执行器偏移量（Link6 局部坐标系，复制到所有环境）
        self._ee_offset_local = torch.tensor(self.cfg.ee_offset_local, device=self.device).repeat(self.num_envs, 1)

        # 初始化各类缓冲区（张量），避免训练中重复分配内存
        n = self.num_envs
        self.target_pos = torch.zeros(n, 3, device=self.device)             # 目标位置的世界坐标，单位为米
        self.target_pos_local = torch.zeros(n, 3, device=self.device)       # 目标位置的局部坐标（相对于环境原点），单位为米
        self.prev_actions = torch.zeros(n, 8, device=self.device)           # 上一步的动作值，单位为无量纲（策略输出在[-1, 1]范围内），用于观测和奖励计算中的历史动作信息
        self.actions = torch.zeros(n, 8, device=self.device)                # 当前的动作值，单位为无量纲（策略输出在[-1, 1]范围内），在_pre_physics_step中更新并在_apply_action中应用到仿真中
        self.applied_torques = torch.zeros(n, 8, device=self.device)        # 实际应用到仿真中的关节扭矩，单位为牛顿米（N*m），在_apply_action中计算并应用到机器人关节执行器中
        self.prev_joint_vel = torch.zeros(n, 8, device=self.device)         # 上一步的关节速度，单位为弧度每秒（rad/s），用于奖励计算中的关节速度变化惩罚
        # control dt (sim_dt * decimation) for downstream maths
        self.dt = self.cfg.sim.dt * self.cfg.decimation                     # 控制时间步长，单位为秒（s），等于仿真时间步长乘以decimation（控制频率与仿真频率的比率），用于奖励计算中的速度和加速度等时间相关的计算
        
        # 末端执行器状态缓存（供奖励和终止条件复用）
        self._ee_pos_w = torch.zeros(n, 3, device=self.device)  # 末端执行器的世界坐标位置，单位为米，奖励计算中用于距离和方向奖励
        self._ee_vel_w = torch.zeros(n, 3, device=self.device)  # 末端执行器的世界坐标线速度，单位为米每秒（m/s），奖励计算中用于速度相关的奖励和惩罚
        self._distance = torch.full((n,), float("inf"), device=self.device)  # 末端执行器与目标之间的距离，单位为米，奖励计算中用于距离奖励和成功判定

        # 阶段性奖励状态跟踪（确保每个距离里程碑只触发一次）
        self._close_thresholds = torch.tensor(self.cfg.rew_close_phase_thresholds, device=self.device)  #距离里程碑阈值列表，单位为米，用于阶段性奖励的触发条件（例如接近目标的不同距离阶段）
        self._close_rewards = torch.tensor(self.cfg.rew_close_phase_rewards, device=self.device)  #距离里程碑奖励值列表，单位为奖励分数，用于阶段性奖励的数值（例如每个距离阶段达到时给予的奖励分数）
        self.close_phase_given = torch.zeros(n, len(self.cfg.rew_close_phase_thresholds),   
                                             dtype=torch.bool, device=self.device)  #每个环境和每个距离里程碑的奖励是否已经发放的布尔标记，用于确保每个阶段性奖励只触发一次

        # 夹爪朝向向量（Link6 局部坐标系 +Z 轴，复制到所有环境）
        self._gripper_forward_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(n, 1)
    
    """设置仿真场景"""
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)  # 加载机器人资产
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())  #生成地面平面
        self.scene.clone_environments(copy_from_source=False)  #克隆环境实例，copy_from_source=False表示不复制源环境的物理状态（每个环境将独立初始化），但会复制资产和配置                        
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])  #在CPU上禁用所有碰撞检测（因为CPU仿真不支持碰撞），global_prim_paths=[]表示不保留任何碰撞对
        self.scene.articulations["robot"] = self.robot  #将机器人添加到场景的关节列表中，方便后续访问和控制
        #添加穹顶光（均匀照明）
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # 创建目标可视化标记（每个环境一个绿色球体）
        self.target_marker = VisualizationMarkers(_TARGET_MARKER_CFG)

    # ------------------------------------------------------------------ #
    # 动作处理流程                                                         #
    # ------------------------------------------------------------------ #

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        物理仿真前处理动作：
        1. 裁剪动作到 [-1, 1]
        2. 缩放动作到实际力矩范围
        3. 清零夹爪力矩（Phase 1 只训练到达任务）
        """
        self.actions = actions.clone().clamp(-1.0, 1.0)   # 克隆并裁剪动作到 [-1, 1] 范围，确保输入动作在预期范围内，避免异常值导致仿真不稳定
        torques = torch.empty_like(self.actions)  # 创建一个与动作张量形状相同的空张量，用于存储实际应用到仿真中的关节扭矩值
        torques[:, :6] = self.actions[:, :6] * self.cfg.action_scale_arm  # 将前6个动作分量（对应手臂关节）缩放到实际的力矩范围，action_scale_arm 是一个标量，定义了动作值到力矩值的缩放比例
        torques[:, 6:] = 0.0  # 将后2个动作分量（对应夹爪关节）的力矩设置为0，Phase 1 只训练到达任务，不需要夹爪动作，因此直接清零夹爪的力矩输出
        self.applied_torques = torques  # 将计算得到的实际力矩值存储在 applied_torques 张量中，供后续在 _apply_action 中应用到仿真中

    def _apply_action(self) -> None:
        """将计算好的力矩应用到机器人关节"""
        # 注意：Isaac Lab 要求 joint_ids 以元组/列表形式传递
        self.robot.set_joint_effort_target(self.applied_torques, joint_ids=self._all_dof_idx)

    # ------------------------------------------------------------------ #
    # 观测计算                                                            #
    # ------------------------------------------------------------------ #

    def _update_ee_state(self) -> None:
        """更新末端执行器状态（世界坐标系下的位置和速度）"""
        # 获取 Link6 的世界位置和四元数
        link6_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx]
        link6_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]

        # 计算末端执行器偏移量（将局部偏移旋转到世界坐标系）
        offset_world = quat_apply(link6_quat_w, self._ee_offset_local)
        self._ee_pos_w = link6_pos_w + offset_world

        # 计算末端执行器速度：线速度 + 角速度 × 偏移量（叉积）
        link6_lin_vel = self.robot.data.body_lin_vel_w[:, self._ee_body_idx]
        link6_ang_vel = self.robot.data.body_ang_vel_w[:, self._ee_body_idx]
        
        # 更新末端到目标的距离
        self._ee_vel_w = link6_lin_vel + torch.linalg.cross(link6_ang_vel, offset_world, dim=-1)

        self._distance = torch.linalg.norm(self._ee_pos_w - self.target_pos, dim=-1)

    def _get_observations(self) -> dict:
        """返回当前观测值（供策略网络输入）"""
        self._update_ee_state()  # 确保末端执行器状态在计算观测前是最新的
        
        # 拼接观测向量：[相对位置(3), 关节位置(8), 关节速度(8), 上一动作(8), 末端速度(3)]
        joint_pos = self.robot.data.joint_pos[:, self._all_dof_idx]
        joint_vel = self.robot.data.joint_vel[:, self._all_dof_idx]
        relative_pos = self.target_pos - self._ee_pos_w  # 目标相对末端的位置

        obs = torch.cat(
            (
                relative_pos,
                joint_pos,
                joint_vel,
                self.prev_actions,
                self._ee_vel_w,
            ),
            dim=-1,
        )
        return {"policy": obs}

    # ------------------------------------------------------------------ #
    # 奖励计算（PPO 友好型密集奖励）                                         #
    # ------------------------------------------------------------------ #

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg  # 奖励配置参数，包含了各种奖励权重和阈值设置
        dist = self._distance  #  末端执行器与目标之间的距离，单位为米，奖励计算中用于距离奖励和成功判定
        rew = torch.full_like(dist, -c.rew_step_penalty)  # 每步固定的负奖励，单位为奖励分数，用于鼓励策略尽快完成任务，避免无效的徘徊和浪费时间

        # 1. 距离奖励：1/(1+(βd)²)²，距离越近奖励越高
        inner = 1.0 / (1.0 + (c.rew_dist_beta * dist) ** 2)
        dist_reward = c.rew_dist_gain * inner * inner
        # 近距离放大器：5cm 内 ×1.5，2cm 内 ×8（强激励最后冲刺）
        close_mask = dist < c.close_threshold
        succ_mask = dist < c.success_threshold
        dist_reward = torch.where(close_mask, dist_reward * c.rew_close_multiplier, dist_reward)
        dist_reward = torch.where(succ_mask, dist_reward * c.rew_success_multiplier, dist_reward)
        rew = rew + dist_reward

        # 2. 方向奖励：末端速度方向与目标方向一致时奖励（cos²）
        to_target = self.target_pos - self._ee_pos_w  # 从末端指向目标的向量
        to_target_norm = torch.linalg.norm(to_target, dim=-1, keepdim=True).clamp_min(1e-6)  # 目标向量的范数，单位为米，clamp_min 防止除零
        to_target_unit = to_target / to_target_norm
        vel_norm = torch.linalg.norm(self._ee_vel_w, dim=-1, keepdim=True).clamp_min(1e-6)
        vel_cos = (to_target_unit * (self._ee_vel_w / vel_norm)).sum(dim=-1)  # 末端速度方向与目标方向的余弦值，范围 [-1, 1]，单位为无量纲
        rew = rew + c.rew_direction_gain * torch.clamp(vel_cos, min=0.0) ** 2  # 只奖励正向运动

        # 3. 夹爪对齐奖励：Link6 +Z 轴指向目标（带符号 cos²）
        # 关键：背对目标会受罚（而非零奖励），迫使使用腕关节对准
        link6_quat_w = self.robot.data.body_quat_w[:, self._ee_body_idx]
        gripper_forward_w = quat_apply(link6_quat_w, self._gripper_forward_local)  # 夹爪朝向向量在世界坐标系下的表示，单位为无量纲
        align_cos = (gripper_forward_w * to_target_unit).sum(dim=-1)  # 夹爪朝向与目标方向的余弦值，范围 [-1, 1]，单位为无量纲
        rew = rew + c.rew_gripper_align_gain * align_cos * align_cos.abs()  # 夹爪对齐奖励，cos² 带符号，正向对齐奖励更高，背向对齐会受罚

        # 4. 速度惩罚：末端速度过快时惩罚
        ee_speed = torch.linalg.norm(self._ee_vel_w, dim=-1)
        rew = rew - torch.where(ee_speed > c.rew_speed_threshold,
                                torch.full_like(ee_speed, c.rew_speed_penalty),
                                torch.zeros_like(ee_speed))

        # 5. 关节速度变化惩罚：抑制振荡（平滑运动）
        cur_joint_vel = self.robot.data.joint_vel[:, self._all_dof_idx]
        joint_vel_change = (cur_joint_vel - self.prev_joint_vel).abs().sum(dim=-1)
        rew = rew - c.rew_joint_vel_change * joint_vel_change
        self.prev_joint_vel = cur_joint_vel

        # 6. 手腕抑制：惩罚腕关节（J4/J5/J6）的动作、位置和速度
        wrist_action_abs = self.actions[:, 3:6].abs().sum(dim=-1)  # J4/J5/J6 的动作绝对值之和，单位为无量纲，范围 [0, 3]，用于奖励中抑制过度使用腕关节
        rew = rew - c.rew_wrist_action * wrist_action_abs  #

        # arm joint indices: [J1,J2,J3,J4,J5,J6]; slice 3:6 = J4/J5/J6
        wrist_q = self.robot.data.joint_pos[:, self._arm_dof_idx[3:6]]  # J4/J5/J6 的关节位置，单位为弧度，用于奖励中抑制腕关节的偏离（过度弯曲或伸展）
        wrist_qvel = self.robot.data.joint_vel[:, self._arm_dof_idx[3:6]]  # J4/J5/J6 的关节速度，单位为弧度每秒（rad/s），用于奖励中抑制腕关节的快速运动
        wrist_pos_pen = wrist_q.abs().sum(dim=-1)
        wrist_vel_pen = wrist_qvel.abs().sum(dim=-1)
        rew = rew - c.rew_wrist_pos * wrist_pos_pen - c.rew_wrist_vel * wrist_vel_pen

        # 7. 夹爪对称性惩罚：强制左右手指对称运动（替代 URDF 缺失的等式约束）
        gripper_q = self.robot.data.joint_pos[:, self._gripper_dof_idx]
        gripper_asym = (gripper_q[:, 0] + gripper_q[:, 1]).abs()  # 左右手指的关节位置之和的绝对值，单位为弧度，用于奖励中惩罚夹爪的不对称运动（理想情况下应该相等但符号相反）
        rew = rew - c.rew_gripper_asymmetry * gripper_asym

        # 8. 阶段性奖励：每进入一个新的距离里程碑发放一次性奖励
        # 例如：首次进入 4cm、3cm、2cm、1cm 时分别奖励 1.0、2.0、3.0、5.0
        crossed = (dist.unsqueeze(-1) < self._close_thresholds) & (~self.close_phase_given)  # 首次跨越阈值
        rew = rew + (crossed.float() * self._close_rewards).sum(dim=-1)  # 累加奖励
        self.close_phase_given = self.close_phase_given | crossed  # 标记已发放

        # 9. 稳定奖励：在成功阈值内且速度很慢时，额外奖励（鼓励停止）
        settled = succ_mask & (ee_speed < c.rew_settle_speed_threshold)
        rew = rew + c.rew_settle_bonus * settled.float()

        # 日志记录（供 TensorBoard 等工具监控训练）
        self.extras["log"] = {
            "rew/distance_mean": dist.mean(),
            "rew/success_rate": succ_mask.float().mean(),
            "rew/settled_rate": settled.float().mean(),
            "rew/align_cos_mean": align_cos.mean(),
            "diag/wrist_pos_l1": wrist_pos_pen.mean(),    # 腕关节位置 L1 范数
            "diag/wrist_vel_l1": wrist_vel_pen.mean(),    # 腕关节速度 L1 范数
            "diag/gripper_asymmetry": gripper_asym.mean(),
            "rew/total_mean": rew.mean(),
        }
        # 更新上一时刻动作（必须在奖励计算后，供下一帧观测使用）
        self.prev_actions = self.actions.clone()
        return rew

    # ------------------------------------------------------------------ #
    # 终止条件                                                            #
    # ------------------------------------------------------------------ #

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1  # 超时终止条件：当当前步数达到最大步数时，time_out 标记为 True，表示需要终止当前环境的 episode
        """
        返回终止标志：
        - terminated: 任务成功终止（这里永远为 False，不提前终止）
        - time_out: 超时终止（达到最大步数）
        """
        terminated = torch.zeros_like(time_out)  # 任务成功终止条件：这里永远返回 False，因为我们不希望提前终止环境，而是让它自然结束（例如让机器人在目标位置停留直到 episode 结束），因此 terminated 标记始终为 False
        return terminated, time_out

    # ------------------------------------------------------------------ #
    # 环境重置                                                            #
    # ------------------------------------------------------------------ #

    def _sample_targets(self, env_ids: torch.Tensor) -> torch.Tensor:
        """在球形壳内随机采样目标位置"""
        n = env_ids.numel()  # 采样目标位置的数量，等于要重置的环境数量
        c = self.cfg  # 目标采样配置参数，包含了采样范围和中心位置等设置
        
        # 随机采样球坐标：半径、仰角、方位角
        r = torch.empty(n, device=self.device).uniform_(c.target_r_min, c.target_r_max)
        elev = torch.empty(n, device=self.device).uniform_(c.target_elev_min, c.target_elev_max)
        az = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)

        # 球坐标转笛卡尔坐标
        cos_e, sin_e = torch.cos(elev), torch.sin(elev)
        cos_a, sin_a = torch.cos(az), torch.sin(az)
        local = torch.stack(
            [
                r * cos_e * cos_a,
                r * cos_e * sin_a,
                r * sin_e + c.target_center_z,
            ],
            dim=-1,
        )
        local[:, 2] = local[:, 2].clamp_min(c.target_z_min)  # 确保目标位置的 z 坐标不低于 target_z_min，避免目标出现在机器人下方或地面以下
        return local + self.scene.env_origins[env_ids]  # 将局部坐标转换为世界坐标，单位为米

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """重置指定环境（env_ids 为空则重置所有环境）"""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES  # 如果 env_ids 为空，则重置所有环境，_ALL_INDICES 是一个包含所有环境索引的列表或张量
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)  # 调用父类的重置方法，执行一些通用的重置逻辑，例如重置 episode 长度计数器、奖励和终止标志等

        # 1. 重置机器人到初始姿态
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()  # 从机器人资产中获取默认的关节位置，并克隆到一个新的张量，单位为弧度
        joint_vel = torch.zeros_like(joint_pos)  # 关节速度重置为零，单位为弧度每秒（rad/s）
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)  # 将重置后的关节位置和速度写入仿真中，更新机器人的状态到初始姿态，确保每个环境的机器人都从相同的起始状态开始

        # 2. 采样新目标位置
        targets_world = self._sample_targets(env_ids)
        self.target_pos[env_ids] = targets_world
        self.target_pos_local[env_ids] = targets_world - self.scene.env_origins[env_ids]

        # 3. 重置环境内部状态
        self.prev_actions[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = 0.0
        self.close_phase_given[env_ids] = False

        # 4. 更新目标可视化标记位置
        self.target_marker.visualize(translations=self.target_pos)
