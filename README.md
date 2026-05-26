# Isaaclab_alicia
alicia机械臂的强化学习训练

最后更新：2026-05-25
---

## 一、目录结构

```
Isaaclab_alicia/
├── convert_alicia_urdf.sh                          ← URDF→USD 一键脚本
├── scripts/
│   ├── random_agent.py                             ← 随机扭矩烟雾测试
│   ├── zero_agent.py                               ← 零扭矩观察姿态
│   └── rsl_rl/
│       ├── train.py                                ← PPO 训练入口
│       └── play.py                                 ← 加载 checkpoint 演示
├── source/Isaaclab_alicia/
│   ├── pyproject.toml / setup.py                   ← editable 包定义
│   └── Isaaclab_alicia/
│       ├── assets/alicia_duo/                      ← URDF + STL + USD
│       │   ├── urdf/Alicia_D_v5_6_gripper_100mm.urdf
│       │   ├── meshes/*.STL
│       │   └── usd/alicia_duo.usd                  ← convert 后产生
│       └── tasks/direct/isaaclab_alicia/
│           ├── __init__.py                         ← task 注册
│           ├── isaaclab_alicia_env_cfg.py          ← 所有可调参数（!! 重点）
│           ├── isaaclab_alicia_env.py              ← DirectRLEnv 主体
│           └── agents/rsl_rl_ppo_cfg.py            ← PPO 网络 + 超参
└── logs/rsl_rl/alicia_duo_reach/<timestamp>/       ← 训练日志 + checkpoint
```

Task ID: `Template-Isaaclab-Alicia-Direct-v0`（沿用 IsaacLab template 命名，方便后续 IDE 补全）。

---

## 二、运行指令速查

> 所有命令前提：`conda activate env_isaaclab`。路径含空格，记得整串引号 / 用 `.sh` 包装。

### 1. URDF → USD 转换（首次或换臂时）

```bash
bash "/home/spy/graduate_student/quadruped_robot/Reinforcement_Learning/mechanical arm/Isaaclab_alicia/convert_alicia_urdf.sh"
```

参数固定：`--merge-joints --fix-base --joint-stiffness 0 --joint-damping 0 --joint-target-type none --headless`。
fix-base 必须开，否则机械臂会从原点坠落。merge-joints 会把 world / tool0 这种 fixed joint 合并 — 因此末端追踪要用 Link6 + 局部偏移（见 `ee_offset_local`）。

### 2. 安装项目（editable）

```bash
cd "/home/spy/graduate_student/quadruped_robot/Reinforcement_Learning/mechanical arm/Isaaclab_alicia"
python -m pip install -e source/Isaaclab_alicia
```

### 3. 烟雾测试（带 GUI 看效果）

```bash
# 随机扭矩，机械臂乱摆 + 看绿球分布
python scripts/random_agent.py --task=Template-Isaaclab-Alicia-Direct-v0 --num_envs=4

# 零扭矩，看臂在重力下慢慢下垂（验证 fix-base + 关节连接）
python scripts/zero_agent.py --task=Template-Isaaclab-Alicia-Direct-v0 --num_envs=4
```

### 4. 训练（headless，4096 envs，建议 2000 iter）

```bash
python scripts/rsl_rl/train.py \
    --task=Template-Isaaclab-Alicia-Direct-v0 \
    --num_envs=4096 \
    --max_iterations=2000 \
    --headless
```

显存吃紧（< 12 GB）就降到 `--num_envs=2048` 同时把 `--max_iterations` 翻倍。
训练 log 自动写到 `logs/rsl_rl/alicia_duo_reach/<timestamp>/`，每 50 iter 存一个 `model_*.pt`。

### 5. TensorBoard

```bash
tensorboard --logdir "logs/rsl_rl/alicia_duo_reach"
```

关键指标：
- `rew/distance_mean`：末端到目标距离（收敛应 < 0.03 m）
- `rew/success_rate`：dist < 2 cm 占比
- `rew/settled_rate`：dist < 2 cm **且** ee_speed < 5 cm/s 的占比
- `rew/align_cos_mean`：夹爪指向 vs 目标方向的 cos（**必须 > 0**，否则朝错了）
- `diag/wrist_pos_l1` / `wrist_vel_l1`：J4/5/6 是否被压住
- `diag/gripper_asymmetry`：左右指是否同步

### 6. 演示训练好的 policy

```bash
python scripts/rsl_rl/play.py \
    --task=Template-Isaaclab-Alicia-Direct-v0 \
    --num_envs=8
```

默认加载最新 run 的 `model_*.pt`。用 `--load_run <timestamp> --checkpoint <model_*.pt>` 指定。

---

## 三、环境设计要点（`isaaclab_alicia_env.py`）

### Action（8 维）
```
0..2 → Joint1/2/3 扭矩 × action_scale_arm （主动控制：yaw / shoulder / elbow）
3..5 → Joint4/5/6 扭矩 × action_scale_arm （wrist：软惩罚下按需使用）
6,7  → left_finger / right_finger 扭矩 × action_scale_gripper （Phase 1 硬锁 0）
```

policy 输出 clamp 到 [-1, 1] 后乘 scale。Phase 1 阶段 `_pre_physics_step` 内 `torques[:, 6:] = 0`，
靠 gripper actuator stiffness=20 把双指拉回中位（替代 MuJoCo equality 耦合）。

### Observation（30 维）
```
relative_pos  (3) = target - ee_pos
joint_pos     (8) = q[Joint1..Joint6, left_finger, right_finger]
joint_vel     (8) = qvel[同上]
prev_actions  (8) = 上一步 clamp 后的 policy 输出
ee_vel_w      (3) = end-effector linear velocity in world
```

### End-effector 定义
`tool0` 在 URDF 是 Link6 的 fixed-leaf，转 USD 时被 merge 进 Link6，因此运行时无 tool0 body。
`ee_pos = Link6.pos + Link6.rotation @ ee_offset_local`，
默认 `ee_offset_local = (-0.0002, -0.0003, 0.10)` 大致对应两指 grasp center（finger tip 在 z=0.131）。

### Target 采样
球壳采样（base frame）：
- 半径 r ∈ [`target_r_min`, `target_r_max`]
- elevation ∈ [`target_elev_min`, `target_elev_max`]，相对水平面
- azimuth 全 360°
- z clamp 到 ≥ `target_z_min` 防止贴地

不能用 FK 采样：IsaacLab 没有"不跑 physics 就刷新 body_pos_w"的 API。

### Reward 结构（**重点**）

每步累加，所有系数都在 `env_cfg`，调参直接改这里：

| 项 | 公式 | cfg 字段 | 说明 |
|---|---|---|---|
| step_penalty | -`s` | `rew_step_penalty` | 鼓励快完成 |
| dist_reward | `g × (1/(1+(β·d)²))²`，dist<5cm 时 ×`mc`，dist<2cm 时 ×`ms` | `rew_dist_gain` / `_beta` / `_close_multiplier` / `_success_multiplier` | 主信号（连续、单调）|
| direction | `g × max(0, cos(ee_vel, to_target))²` | `rew_direction_gain` | 鼓励朝目标运动 |
| **gripper align** | `g × cos × \|cos\|`（**signed** cos²）| `rew_gripper_align_gain` | 夹爪 +z 指向目标。**有符号** —— 反向**扣分** |
| close-tier 一次性 | 跨过 4/3/2/1 cm 各发奖 | `rew_close_phase_thresholds/rewards` | 防止 hover，激励真的推进 |
| speed penalty | -`p` if speed > thr | `rew_speed_penalty / _threshold` | 抑制狂奔 |
| joint_vel_change | -`g × Σ\|Δqvel\|` | `rew_joint_vel_change` | 抑制抖动 |
| **wrist 软惩罚** | `-α·\|a[3:6]\| - β·Σ\|q[J4-6]\| - γ·Σ\|qvel[J4-6]\|` | `rew_wrist_action / _pos / _vel` | "只在必要时用 wrist" |
| **gripper 对称** | `-g × \|q[left] + q[right]\|` | `rew_gripper_asymmetry` | 替代 MuJoCo equality |
| settle bonus | +`b` if (succ + slow) | `rew_settle_bonus / _settle_speed_threshold` | 教 policy 到了就停下来 |

**不再有：** 一次性 +10000 success bonus（PPO 不喜欢大值），episode-terminating 成功（让 policy 学到"hover" → 改成不 terminate 让累积稠密 reward）。

### Termination
只有 timeout：`episode_length_buf >= max_episode_length - 1`（约 240 步 / 4 秒）。

---

## 四、可调参数完整清单（`isaaclab_alicia_env_cfg.py`）

> 调参原则：**先动一个，跑 800-2000 iter 看曲线，再动下一个**。一次动多个不知道哪个起作用。

### 基础任务参数

| 字段 | 当前值 | 含义 / 调参建议 |
|---|---|---|
| `decimation` | 2 | 物理 120Hz → 控制 60Hz |
| `episode_length_s` | 4.0 | 240 步 / episode；reach 4s 够 |
| `action_space` | 8 | 不要改 |
| `observation_space` | 30 | 改 obs 内容时同步更新 |

### Actuator（关节阻抗）

| Group | stiffness | damping | effort_limit | 说明 |
|---|---|---|---|---|
| `arm_main` (J1-3) | 0 | 0.5 | 5 | 纯扭矩 |
| `arm_wrist` (J4-6) | 100 | 10 | 10 | 高刚度位置锁在 0；Phase 2 可改为纯扭矩 |
| `gripper` | 20 | 1 | 5 | 弱回中弹簧，Phase 2 5N·m 扭矩能压过弹簧 |

### Target 采样（球壳）

| 字段 | 当前值 | 说明 |
|---|---|---|
| `target_r_min` | 0.18 m | 太近会和 base 干涉 |
| `target_r_max` | 0.35 m | 太远不可达。改 `ee_offset` 后需要同步缩 |
| `target_elev_min` | -0.17 rad (~-10°) | 略向下 |
| `target_elev_max` | 0.87 rad (~50°) | 向上 |
| `target_center_z` | 0.20 m | 球壳中心高度（约肩高）|
| `target_z_min` | 0.05 m | 兜底防止贴地 |

### Reward 系数（**最频繁调的**）

主奖励：

| 字段 | 当前值 | 量级感受 |
|---|---|---|
| `rew_dist_gain` | 0.5 | 主信号；砍太多 policy 学不动，加太多 hover |
| `rew_dist_beta` | 2.0 | 控制曲线"宽度"；大 = 集中近距 |
| `rew_close_multiplier` | 1.5 | dist<5cm 时 ×系数；不要 >2 防 hover 边缘 |
| `rew_success_multiplier` | 8.0 | dist<2cm 时再 ×；让到位 >>> 近距 |
| `rew_direction_gain` | 0.05 | 速度方向 cos² |
| `rew_gripper_align_gain` | 0.5 | **signed** cos²，朝错扣分 |
| `rew_close_phase_rewards` | (1, 2, 3, 5) | 4/3/2/1 cm 一次性 |
| `rew_settle_bonus` | 4.0 | 到位 + 静止才发 |
| `rew_settle_speed_threshold` | 0.05 m/s | 多慢算"静止" |

惩罚：

| 字段 | 当前值 | 说明 |
|---|---|---|
| `rew_step_penalty` | 0.001 | 时间压力 |
| `rew_wrist_action` | 0.05 | |a[J4-6]| |
| `rew_wrist_pos` | 0.15 | |q[J4-6]| 偏离零位 |
| `rew_wrist_vel` | 0.02 | |qvel[J4-6]| |
| `rew_gripper_asymmetry` | 5.0 | |q[L] + q[R]| 不对称 |
| `rew_joint_vel_change` | 0.001 | Σ|Δqvel| 步间变化 |
| `rew_speed_penalty / _threshold` | 0.01 / 1.0 | ee_speed 上限 |

### Reach 成功判定

| 字段 | 当前值 |
|---|---|
| `success_threshold` | 0.02 m |
| `close_threshold` | 0.05 m |

### Action scale

| 字段 | 当前值 | 说明 |
|---|---|---|
| `action_scale_arm` | 5.0 | policy [-1,1] × 5 = ±5 N·m |
| `action_scale_gripper` | 5.0 | Phase 2 启用 |

### PPO 超参（`agents/rsl_rl_ppo_cfg.py`）

| 字段 | 当前值 | 说明 |
|---|---|---|
| `num_steps_per_env` | 24 | 4096 × 24 ≈ 98k samples/iter |
| `max_iterations` | 2000 | reach 800-1500 iter 够，2000 保险 |
| `policy.actor_hidden_dims` | [256,128,64] | reach 任务够；复杂 grasp 可上 [512,256,128] |
| `algorithm.entropy_coef` | 0.002 | 后期收敛用 0.001-0.003 |
| `algorithm.learning_rate` | 3e-4 | KL adaptive，会自适应 |
| `algorithm.gamma` | 0.99 | |
| `algorithm.desired_kl` | 0.008 | |

---

## 五、训练历史 & 主要 bug 记录

| 里程碑 | 现象 / 解决 |
|---|---|
| **小球频繁跳** | 两个原因合并：FK 采样依赖 stale `body_pos_w`，target 收敛到 ee_pos → 假 success；out_of_bounds margin 太紧（0.05 rad）random 扭矩立刻越界。**修**：FK 改球壳采样；删 oob termination；用 `episode_length_buf > 0` mask 抑制 reset 后第一帧 distance stale |
| **PPO `std >= 0` 报错** | reward 第一步 `inf - dist = inf`，反传把 log_std 推 NaN。**修**：improvement reward 加 `isfinite()` mask，inf baseline 时不发 |
| **PPO plateau 在 0.21 m** | 7 档 phase 阶梯太离散，PPO 信号断了。**修**：换成 dense `1/(1+(β·d)²)²` + 2 档 tier amplifier |
| **小球附近一直抖** | success terminate + 一次性 bonus < 累积 hover reward → policy 学到边缘收菜。**修**：去掉 terminate；加 settle bonus（succ + slow）|
| **wrist 乱转 / 夹爪不对称** | URDF→USD 丢了 MuJoCo equality；wrist action penalty 太弱。**修**：J4/5/6 软惩罚（pos+vel+action）；gripper actuator 加小 stiffness=20 + asymmetry penalty |
| **align_cos 出负值** | wrist 软惩罚太强（pos=0.5），policy 不敢用 wrist 调向 → 反向也无所谓。**修**：弱化 wrist 惩罚（0.5→0.15）；align reward 改 **signed** cos²（反向**扣分**）|
| **正负 reward 比 200:1** | 正奖励 4000 / 惩罚 -18，软约束被淹没。**修**：dist_gain 1.0→0.5，settle 8→4，把比例压到 ~100:1 |

---

## 六、Phase 2（grasp）规划

完成 reach 后切到 grasp：

1. **解锁夹爪**：`_pre_physics_step` 里 `torques[:, 6:7] = a[6:7] × action_scale_gripper`，
   再额外做镜像 `torques[:, 7:8] = -torques[:, 6:7]` 保证对称（替代缺失的 equality）
2. **加 grasp 成功条件**：`close_progress = 0.5*(-q[left]/0.025 + q[right]/0.025)`，
   reach_ok + `close_progress >= 0.6` 才算完整 success
3. **二阶段 reward**：reach phase 不立刻 done，per-step 给 `+1.0 + 10×close_progress`，
   完整 success 时再发大奖（参考 PROGRESS run6 设计）
4. **可选 hard mask**：dist > 0.08 m 时 `torques[:, 6:8] = 0`，强制远距离不闭夹爪

需新增的 cfg 字段（占位）：
```
grasp_threshold = 0.6        # close_progress 阈值
rew_close_progress = 10.0    # 每步给的 grasp shaping
rew_grasp_success = 100.0    # 完整成功 bonus
gripper_enable_dist = 0.08   # 硬掩码距离
```

---

## 七、依赖版本

```
isaaclab           2.3.2
isaacsim           5.1
rsl-rl-lib         5.3.0   # < 5.0 没有 share_cnn_encoders 字段会报错
gymnasium          (随 IsaacLab)
torch              (随 IsaacLab，cu126)
```

---

## 八、下次接手怎么开始

1. `conda activate env_isaaclab`
2. 在项目根跑 `bash convert_alicia_urdf.sh` 确认 USD 还在
3. 跑一遍烟雾测试 `python scripts/random_agent.py --task=Template-Isaaclab-Alicia-Direct-v0 --num_envs=4`，
   看绿球分布 + 机械臂在 random 扭矩下乱摆
4. 想训练就 `python scripts/rsl_rl/train.py --task=Template-Isaaclab-Alicia-Direct-v0 --num_envs=4096 --max_iterations=2000 --headless`
5. 改 reward 优先编辑 `source/Isaaclab_alicia/Isaaclab_alicia/tasks/direct/isaaclab_alicia/isaaclab_alicia_env_cfg.py`
6. 出现"成功率突降"、"align_cos 变负"、"机械臂乱转/抖"等异常，先查本文档第六节 bug 记录

如果换机械臂：换 URDF 路径 + 关节名（`arm_joint_names` / `gripper_joint_names` / `ee_body_name` / `ee_offset_local`）+ 重转 USD + 重新调 `target_r_*` 球壳即可。
