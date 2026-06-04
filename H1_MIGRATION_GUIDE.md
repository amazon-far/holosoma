# H1 机器人迁移到 Holosoma 项目 — 完整操作手册

> **阅读说明**：这份手册假设你是第一次修改这个项目。每一步都会告诉你：
> - 要做什么
> - 为什么这样做
> - 哪个文件、哪一行
> - 写完代码后用什么命令验证

---

## 📁 项目代码结构速览

在动手前，先搞清楚哪些目录和文件是你要碰的：

```
src/holosoma/holosoma/
├── config_types/          # 配置数据类的类型定义（不需要改）
│   └── robot.py           # RobotConfig 的字段定义
├── config_values/         # 所有配置的默认值 ← 你要大量改这里
│   ├── robot.py           # G1/T1 的 RobotConfig 实例
│   ├── experiment.py      # 注册所有 exp:xxx 的入口
│   ├── reward.py          # 注册所有 reward:xxx
│   ├── termination.py     # 注册所有 termination:xxx
│   ├── action.py          # 注册所有 action:xxx
│   ├── observation.py     # 注册所有 observation:xxx
│   ├── command.py         # 注册所有 command:xxx
│   ├── curriculum.py      # 注册所有 curriculum:xxx
│   ├── randomization.py   # 注册所有 randomization:xxx
│   ├── algo.py            # PPO/FastSAC 算法配置
│   └── loco/
│       ├── g1/            # G1 locomotion 配置
│       │   ├── experiment.py
│       │   ├── action.py
│       │   ├── observation.py
│       │   ├── reward.py
│       │   ├── termination.py
│       │   ├── command.py
│       │   ├── curriculum.py
│       │   └── randomization.py
│       └── t1/            # T1 locomotion 配置
│           └── ...        # 和 g1/ 结构一样
├── data/robots/
│   ├── g1/                # G1 的 URDF + XML + meshes
│   └── t1/                # T1 的 URDF + XML + meshes
├── envs/locomotion/
│   └── locomotion_manager.py  # 运动环境管理器
├── bridge/unitree/
│   └── unitree_sdk2py_bridge.py  # Unitree sim2sim 桥接
└── agents/fast_sac/
    └── fast_sac_agent.py  # FastSAC 算法

src/holosoma_inference/holosoma_inference/
└── config/
    ├── config_types/      # 推理侧配置类型定义
    │   ├── robot.py       # RobotConfig 字段
    │   ├── inference.py   # InferenceConfig 字段
    │   ├── observation.py # ObservationConfig 字段
    │   └── task.py        # TaskConfig 字段
    └── config_values/     # 推理侧配置默认值
        ├── robot.py       # G1/T1 的 RobotConfig
        ├── inference.py   # 注册所有 inference:xxx
        ├── observation.py # 注册所有 observation
        └── task.py        # 注册所有 task
```

---

## 🎯 阶段 A：资产和 RobotConfig（训练侧）

**目标**：让 MuJoCo 或 IsaacGym 能加载 H1 机器人模型，并且能站着不动。

### 第 1 步：准备 H1 的 URDF/XML 和网格文件

你需要在 `src/holosoma/holosoma/data/robots/` 下新建一个 `h1/` 目录，结构和 G1 一样：

```text
src/holosoma/holosoma/data/robots/h1/
    h1.urdf          ← 给 IsaacGym 用
    h1.xml           ← 给 MuJoCo / MJWarp 用
    meshes/           ← 所有 STL/DAE/OBJ 网格文件
```

**注意**：URDF 和 XML 里用到的 mesh 路径要能对上。项目里的惯例是用 `@holosoma/data/robots` 作为根路径，你在 `asset.asset_root` 里设置的。

#### 如果你没有 XML（MJCF）文件

如果你只有 URDF 而没有 XML，有两个办法：
1. 用 MuJoCo 自带的 `compile` 工具把 URDF 转成 MJCF
2. 先只用 IsaacGym 训练，MJWarp 之后再说

**命令**：无需运行命令，这一步只是准备文件。

**检查点**：
- [ ] `h1.urdf` 存在
- [ ] `h1.xml` 存在（如果要用 MuJoCo）
- [ ] `meshes/` 目录有所有需要的网格文件

---

### 第 2 步：确定 H1 的关节名、身体名、默认角度

**这是整个迁移里最重要的一步。** 你需要从 URDF/XML 里提取下面这些信息。

#### 2a. 从 URDF/XML 提取所有 joint name

打开 `h1.urdf` 或 `h1.xml`，找出所有 `revolute` / `continuous` / `prismatic` 类型的 joint。你需要列出它们的**名字**和**顺序**。

> 注意：URDF 里的 `fixed` joint 不算。你要的是**可控关节**。

你需要做出一个像这样的列表（这是 G1 的例子，你要对照 H1 的 URDF 来填）：

```python
dof_names = [
    "left_hip_pitch_joint",   # 第 0 个 action
    "left_hip_roll_joint",    # 第 1 个 action
    # ... H1 的所有可控关节
]
```

**关节顺序为什么重要？** 
Policy 输出的 action 是一个向量 `[a0, a1, a2, ..., a28]`。`a0` 对应 `dof_names[0]` 这个关节，`a1` 对应 `dof_names[1]`……所以顺序一旦错了，策略就会让错误的关节动。

#### 2b. 从 URDF/XML 提取所有 body/link name

body 名字用于查找脚、查找 torso、确定哪些身体接触算摔倒。你需要一个**完整的 body 名字列表**（按 URDF/XML 里出现的顺序）。

#### 2c. 确定 H1 的默认站姿

你需要知道 H1 在"自然站立"时候每个关节的角度（弧度制）。如果你不确定，可以先全写 0.0，后面再调。但至少腿关节（髋、膝、踝）的角度要尽量接近真实站立姿态。

#### 2d. 确定 H1 的脚名和脚高名

G1 有两个概念：
- `foot_body_name`：脚的真实 body，用于接触检测。G1 用的是 `"ankle_roll_link"`（这是一个模式匹配，会匹配到所有包含这个字符串的 body）
- `foot_height_name`：一个辅助用的 body，用于测量脚离地高度。G1 用的是 `"foot_contact_point"`

你需要从 H1 的 URDF 里找对应的名字。

#### 2e. 确定 H1 的关节限位、力矩限位

从 URDF 的 `<limit lower="..." upper="..." effort="..." velocity="..."/>` 提取每个关节的：
- 位置下限
- 位置上限
- 速度上限
- 力矩上限

---

### 第 3 步：在 `config_values/robot.py` 中新增 H1 的 RobotConfig

打开文件：
```
src/holosoma/holosoma/config_values/robot.py
```

在你写代码前，先理解 `RobotConfig` 每个字段的含义（不需要背，照着 G1 改就行）：

| 字段 | 含义 | 怎么填 |
|------|------|--------|
| `num_bodies` | 机器人有多少个 body（link） | 数 URDF 里的 link 数量 |
| `dof_obs_size` | 在 obs 里记录多少个 DOF | 和 `actions_dim` 一样 |
| `actions_dim` | policy 输出的 action 维度 | = len(dof_names) |
| `policy_obs_dim` | actor obs 的总维度 | -1 表示自动计算 |
| `critic_obs_dim` | critic obs 的总维度 | -1 表示自动计算 |
| `key_bodies` | 关键的 body 名 | 脚 contact point 的名字 |
| `num_feet` | 脚的数量 | 2 |
| `foot_body_name` | 脚 body 的模式匹配字符串 | 如 `"ankle_roll_link"` |
| `foot_height_name` | 脚高 body 的模式匹配字符串 | 如 `"foot_contact_point"` |
| `knee_name` | 膝盖 body 的模式匹配字符串 | 如 `"knee_link"` |
| `torso_name` | 躯干 body 的精确名字 | 如 `"torso_link"` |
| `dof_names` | **所有可控关节名，按顺序** | 从 URDF 提取 |
| `upper_dof_names` | 上半身 DOF | 腰 + 手臂 |
| `lower_dof_names` | 下半身 DOF | 腿 |
| `has_torso` | 是否有躯干 | True |
| `has_upper_body_dof` | 是否有上半身关节 | True/False |
| `*_dof_names` | 各种 DOF 分组 | 按你的 H1 版本填 |
| `dof_pos_lower_limit_list` | 每个关节的位置下限 | 从 URDF limit 提取 |
| `dof_pos_upper_limit_list` | 每个关节的位置上限 | 从 URDF limit 提取 |
| `dof_vel_limit_list` | 每个关节的速度上限 | 从 URDF limit 提取 |
| `dof_effort_limit_list` | 每个关节的力矩上限 | 从 URDF limit 提取 |
| `dof_armature_list` | 每个关节的电枢惯量 | 先用全 0.01，后面再调 |
| `dof_joint_friction_list` | 每个关节的摩擦 | 先用全 0.0 |
| `body_names` | **所有 body 名，按 URDF 里的顺序** | 从 URDF 提取 |
| `terminate_after_contacts_on` | 哪些 body 接触地面算摔倒 | 如 `["pelvis", "shoulder", "hip"]` |
| `penalize_contacts_on` | 哪些 body 接触地面要惩罚 | 比上面的列表更广 |
| `init_state.pos` | 初始 base 位置 `[x, y, z]` | z 要等于 H1 站立时的 base 高度 |
| `init_state.rot` | 初始 base 姿态 `[x, y, z, w]` | 四元数，`[0,0,0,1]` 表示平放 |
| `init_state.default_joint_angles` | **每个关节的初始角度（字典）** | 从第 2c 步获取 |
| `randomize_link_body_names` | 哪些 body 可以做质量/COM 随机化 | 选主要的 link |
| `symmetry_joint_names` | 左右对称关节的对应关系 | 如 `"left_hip": "right_hip"` |
| `flip_sign_joint_names` | 哪些关节在对称镜像时要取反 | 如 roll/yaw 关节 |
| `control.stiffness` | **每组关节的 P 增益** | 见下方说明 |
| `control.damping` | **每组关节的 D 增益** | 见下方说明 |
| `control.action_scale` | action 的缩放系数 | locomotion 一般 0.25 |
| `contact_pairs_multiplier` | 接触对乘数 | 一般 16 |
| `asset.asset_root` | 资产根路径 | `"@holosoma/data/robots"` |
| `asset.urdf_file` | URDF 文件路径 | `"h1/h1.urdf"` |
| `asset.xml_file` | XML 文件路径 | `"h1/h1.xml"` |
| `asset.robot_type` | 机器人类型标识符 | `"h1_29dof"` 或你的命名 |
| `bridge.sdk_type` | SDK 类型 | `"unitree"`（H1 用 Unitree SDK） |

#### 3a. 关于 stiffness/damping（PD 增益）

G1 的 PD 增益是按关节**组**（joint group）来分的，不是按关节名字，而是按关键词匹配：

```python
stiffness={
    "hip_yaw": 40.17,     # 匹配所有名字里含 "hip_yaw" 的关节
    "hip_roll": 99.09,
    "hip_pitch": 40.17,
    "knee": 99.09,
    # ...
}
```

你的 H1 关节名字如果和 G1 一样用 snake_case（如 `left_hip_yaw_joint`），那这些 key 就可以复用。如果不一样（如 T1 用的是 `Left_Hip_Yaw`），你就要按你的实际关节名字来写 key。

**H1 的 PD 初值建议**：
- 如果你的 H1 关节名字和 G1 类似（snake_case），直接复制 G1 的 stiffness/damping 作为初值
- 如果你的 H1 关节名字和 T1 类似（PascalCase），复制 T1 的 stiffness/damping 作为初值
- PD 值后期要调，但先用一个合理的初值让机器人能站住

#### 3b. 写代码

在 `robot.py` 的**文件最底部**，找到 `DEFAULTS` 字典（约第 1107 行），在 `DEFAULTS` 的**上面**（即 `DEFAULTS` 的前一行）添加你的 H1 配置，然后在 `DEFAULTS` 字典里加上一行。

**具体操作**：

```python
# 在 robot.py 文件的 g1_29dof_w_object 之后、DEFAULTS 之前，添加：

h1_29dof = RobotConfig(
    num_bodies=?,        # 填 H1 的 body 数量
    dof_obs_size=?,      # 填 H1 的 DOF 数量
    actions_dim=?,       # 填 H1 的 DOF 数量
    policy_obs_dim=-1,
    critic_obs_dim=-1,
    algo_obs_dim_dict={},
    key_bodies=["left_foot_contact_point", "right_foot_contact_point"],  # 改成 H1 的
    num_feet=2,
    foot_body_name="?",   # 改成 H1 的脚 body 名
    foot_height_name="?", # 改成 H1 的脚高 body 名
    knee_name="?",        # 改成 H1 的膝盖 body 名
    torso_name="?",       # 改成 H1 的躯干 body 名
    dof_names=[...],      # 从第 2a 步填写
    # ... 其他字段按上面表格逐一填写
    asset=RobotAssetConfig(
        asset_root="@holosoma/data/robots",
        # ... 其他 asset 字段照抄 G1
        urdf_file="h1/h1.urdf",    # ← 注意这里是 h1
        xml_file="h1/h1.xml",      # ← 注意这里是 h1
        robot_type="h1_29dof",     # ← 你的 H1 类型名
        # ...
    ),
    bridge=RobotBridgeConfig(
        sdk_type="unitree",        # H1 用 unitree
        motor_type="serial",
    ),
)

# 然后在 DEFAULTS 字典里添加一行：
DEFAULTS = {
    "g1_29dof": g1_29dof,
    "t1_29dof_waist_wrist": t1_29dof_waist_wrist,
    "g1_29dof_w_object": g1_29dof_w_object,
    "h1_29dof": h1_29dof,          # ← 添加这一行
}
```

**验证命令**（写完 robot config 后）：

```bash
cd /home/pjm/Desktop/holosoma

# 试着用 MuJoCo 加载 H1
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:h1-29dof \
  simulator:mujoco \
  terrain:terrain-locomotion-plane

# 如果没有 MuJoCo 环境，用 IsaacGym 加载
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
  exp:h1-29dof \
  simulator:isaacgym \
  --training.num-envs 1 \
  --training.headless False
```

**你希望看到**：
- 日志里打印 `num_dof` 等于你填的 `actions_dim`
- 日志里打印的 DOF names 和你填的完全一致
- 日志里打印的 Body names 和你填的完全一致
- `feet_indices` 找到了（不是 -1）
- `termination_contact_indices` 找到了（不是 -1）

**常见报错和排查**：

| 报错 | 原因 | 怎么修 |
|------|------|--------|
| `Missing default joint angle for DOF 'xxx'` | `default_joint_angles` 字典里少了一个关节 | 把你所有的 dof_names 都加到字典里 |
| `find_rigid_body_indice` 返回 -1 | body 名字写错了 | 对照 URDF 里的 link name 仔细检查 |
| 机器人一出生就穿地 | `init_state.pos[2]`（z 高度）太小 | 加大 z 值 |
| 机器人飞走 | stiffness 太小或太大 | 调整 PD 值 |
| `No module named 'holosoma.config_values.loco.h1'` | 还没创建 h1 loco 配置 | 这是正常的，先跳到下一步 |

---

## 🎯 阶段 B：训练配置（loco/h1/）

**目标**：让 `exp:h1-29dof` 和 `exp:h1-29dof-fast-sac` 能被命令行识别。

### 第 4 步：创建 `config_values/loco/h1/` 目录

创建以下文件（直接复制 G1 的然后改）：

```bash
cd /home/pjm/Desktop/holosoma

# 创建目录
mkdir -p src/holosoma/holosoma/config_values/loco/h1

# 你需要创建这些文件（下一步逐一说明）：
touch src/holosoma/holosoma/config_values/loco/h1/__init__.py
touch src/holosoma/holosoma/config_values/loco/h1/action.py
touch src/holosoma/holosoma/config_values/loco/h1/observation.py
touch src/holosoma/holosoma/config_values/loco/h1/reward.py
touch src/holosoma/holosoma/config_values/loco/h1/termination.py
touch src/holosoma/holosoma/config_values/loco/h1/command.py
touch src/holosoma/holosoma/config_values/loco/h1/curriculum.py
touch src/holosoma/holosoma/config_values/loco/h1/randomization.py
touch src/holosoma/holosoma/config_values/loco/h1/experiment.py
```

**为什么需要这么多文件？**
这个项目用了一种"管理器模式"：训练环境的每个功能（动作、观测、奖励、终止条件、命令、课程学习、随机化）都由一个单独的 manager 负责。每个 manager 的配置独立成一个文件。你需要为 H1 创建一套 locomotion 专用的配置。

---

### 第 5 步：逐个创建配置文件

**原则**：第一版直接复制 G1 的，只改名字。等跑通了再回来调参数。

#### 5a. `loco/h1/__init__.py`

保持空文件即可（或复制 G1 的，它也是空的）。

#### 5b. `loco/h1/action.py`

直接复制 `loco/g1/action.py`，把变量名从 `g1_29dof_joint_pos` 改成 `h1_29dof_joint_pos`：

```python
"""Locomotion action presets for the H1 robot."""

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg

h1_29dof_joint_pos = ActionManagerCfg(
    terms={
        "joint_control": ActionTermCfg(
            func="holosoma.managers.action.terms.joint_control:JointPositionActionTerm",
            params={},
            scale=1.0,
            clip=None,
        ),
    }
)

__all__ = ["h1_29dof_joint_pos"]
```

**需要注意**：`action_scale` 不在这里设，而是在 `robot.py` 的 `control.action_scale` 里设。这里 scale=1.0 是对的。

#### 5c. `loco/h1/observation.py`

直接复制 `loco/g1/observation.py`，改变量名。

```python
"""Locomotion observation presets for the H1 robot."""

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

h1_29dof_loco_single_wolinvel = ObservationManagerCfg(
    groups={
        "actor_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=True,
            history_length=1,
            terms={
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=0.25,
                    noise=0.0,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_lin_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_ang_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                    noise=0.01,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.05,
                    noise=0.1,
                ),
                "actions": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:actions",
                    scale=1.0,
                    noise=0.0,
                ),
                "sin_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:sin_phase",
                    scale=1.0,
                    noise=0.0,
                ),
                "cos_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:cos_phase",
                    scale=1.0,
                    noise=0.0,
                ),
            },
        ),
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms={
                "base_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_lin_vel",
                    scale=2.0,
                    noise=0.0,
                ),
                "base_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:base_ang_vel",
                    scale=0.25,
                    noise=0.0,
                ),
                "projected_gravity": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:projected_gravity",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_lin_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_lin_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "command_ang_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:command_ang_vel",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_pos": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_pos",
                    scale=1.0,
                    noise=0.0,
                ),
                "dof_vel": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:dof_vel",
                    scale=0.05,
                    noise=0.0,
                ),
                "actions": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:actions",
                    scale=1.0,
                    noise=0.0,
                ),
                "sin_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:sin_phase",
                    scale=1.0,
                    noise=0.0,
                ),
                "cos_phase": ObsTermCfg(
                    func="holosoma.managers.observation.terms.locomotion:cos_phase",
                    scale=1.0,
                    noise=0.0,
                ),
            },
        ),
    }
)

__all__ = ["h1_29dof_loco_single_wolinvel"]
```

**重要**：actor_obs 各项的**顺序**就是 policy 看到的观测向量的拼接顺序。后面 inference 侧必须和这里完全一致。

#### 5d. `loco/h1/reward.py`

直接复制 `loco/g1/reward.py`，改变量名。

**特别要注意 `pose` 奖励的 `pose_weights`**：这个列表的长度必须等于 `actions_dim`（即 DOF 数量）。G1 有 29 个 DOF 所以有 29 个权重。如果你的 H1 DOF 数量不同，就要调整。

```python
"""Locomotion reward presets for the H1 robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

h1_29dof_loco = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=2.0,
            params={"tracking_sigma": 0.25},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_ang_vel",
            weight=1.5,
            params={"tracking_sigma": 0.25},
        ),
        "penalty_ang_vel_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_ang_vel_xy",
            weight=-1.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_orientation": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_orientation",
            weight=-10.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "penalty_action_rate": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_action_rate",
            weight=-2.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "feet_phase": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:feet_phase",
            weight=5.0,
            params={"swing_height": 0.09, "tracking_sigma": 0.008},
        ),
        "pose": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:pose",
            weight=-0.5,
            params={
                "pose_weights": [
                    0.01, 1.0, 5.0, 0.01, 5.0, 5.0,    # 左腿
                    0.01, 1.0, 5.0, 0.01, 5.0, 5.0,    # 右腿
                    50.0, 50.0, 50.0,                     # 腰
                    50.0, 50.0, 50.0, 50.0,              # 左臂
                    50.0, 50.0, 50.0,                     # 左腕
                    50.0, 50.0, 50.0, 50.0,              # 右臂
                    50.0, 50.0, 50.0,                     # 右腕
                ],
            },
            tags=["penalty_curriculum"],
        ),
        "penalty_close_feet_xy": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_close_feet_xy",
            weight=-10.0,
            params={"close_feet_threshold": 0.15},
            tags=["penalty_curriculum"],
        ),
        "penalty_feet_ori": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:penalty_feet_ori",
            weight=-5.0,
            params={},
            tags=["penalty_curriculum"],
        ),
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=1.0,
            params={},
        ),
    },
)

h1_29dof_loco_fast_sac = RewardManagerCfg(
    only_positive_rewards=False,
    terms={
        "tracking_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:tracking_lin_vel",
            weight=2.0,
            params={"tracking_sigma": 0.25},
        ),
        # ... (同 h1_29dof_loco，但 alive 的 weight 是 10.0 而不是 1.0)
        "alive": RewardTermCfg(
            func="holosoma.managers.reward.terms.locomotion:alive",
            weight=10.0,   # ← FastSAC 版本 alive 权重更高
            params={},
        ),
    },
)

__all__ = ["h1_29dof_loco", "h1_29dof_loco_fast_sac"]
```

**`pose_weights` 的含义**：
- 数值越大，policy 越倾向于让这个关节保持在 `default_joint_angles`
- 腿部关节设较小值（允许自由运动），上半身关节设较大值（尽量保持默认姿势）
- 如果你想让 H1 的手臂完全不动，就给手臂关节设更大的值（如 100.0）

#### 5e. `loco/h1/termination.py`

直接复制 G1 的即可，因为 `contact` 终止条件是从 `robot_config.terminate_after_contacts_on` 找身体索引的。

```python
"""Locomotion termination presets for the H1 robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

h1_29dof_termination = TerminationManagerCfg(
    terms={
        "contact": TerminationTermCfg(
            func="holosoma.managers.termination.terms.locomotion:contact_forces_exceeded",
            params={
                "force_threshold": 1.0,
                "contact_indices_attr": "termination_contact_indices",
            },
        ),
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
    }
)

__all__ = ["h1_29dof_termination"]
```

#### 5f. `loco/h1/command.py`

直接复制 G1 的。第一版速度命令范围可以设小一点：

```python
"""Locomotion command presets for the H1 robot."""

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg

h1_29dof_command = CommandManagerCfg(
    params={
        "locomotion_command_resampling_time": 10.0,
    },
    setup_terms={
        "locomotion_gait": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionGait",
            params={
                "gait_period": 1.0,
                "gait_period_randomization_width": 0.2,
            },
        ),
        "locomotion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.locomotion:LocomotionCommand",
            params={
                "command_ranges": {
                    "lin_vel_x": [-0.5, 1.0],    # ← 第一版保守一点，向前 0.5~1.0 m/s
                    "lin_vel_y": [-0.3, 0.3],    # ← 侧向速度也小
                    "ang_vel_yaw": [-0.5, 0.5],  # ← 转向也小
                    "heading": [-3.14, 3.14],
                },
                "stand_prob": 0.2,               # 20% 概率发站立命令
            },
        ),
    },
    reset_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionCommand"),
    },
    step_terms={
        "locomotion_gait": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionGait"),
        "locomotion_command": CommandTermCfg(func="holosoma.managers.command.terms.locomotion:LocomotionCommand"),
    },
)

__all__ = ["h1_29dof_command"]
```

#### 5g. `loco/h1/curriculum.py`

直接复制 G1 的，改变量名：

```python
"""Locomotion curriculum presets for the H1 robot."""

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg

h1_29dof_curriculum = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
        ),
        "penalty_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:PenaltyCurriculum",
            params={
                "enabled": True,
                "tag": "penalty_curriculum",
                "initial_scale": 0.1,
                "min_scale": 0.0,
                "max_scale": 1.0,
                "level_down_threshold": 150.0,
                "level_up_threshold": 750.0,
                "degree": 0.00025,
            },
        ),
    },
    reset_terms={},
    step_terms={},
)

h1_29dof_curriculum_fast_sac = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
        ),
        "penalty_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:PenaltyCurriculum",
            params={
                "enabled": True,
                "tag": "penalty_curriculum",
                "initial_scale": 0.5,
                "min_scale": 0.5,
                "max_scale": 1.0,
                "level_down_threshold": 150.0,
                "level_up_threshold": 750.0,
                "degree": 0.001,
            },
        ),
    },
    reset_terms={},
    step_terms={},
)

__all__ = ["h1_29dof_curriculum", "h1_29dof_curriculum_fast_sac"]
```

#### 5h. `loco/h1/randomization.py`

第一版先大幅减少随机化，让训练更稳定：

```python
"""Locomotion randomization presets for the H1 robot."""

from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg

h1_29dof_randomization = RandomizationManagerCfg(
    setup_terms={
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState",
            params={
                "push_interval_s": [10, 20],
                "max_push_vel": [0.5, 0.5],
                "enabled": False,   # ← 第一版关闭推机器人
            },
        ),
        "setup_action_delay_buffers": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:setup_action_delay_buffers",
            params={
                "ctrl_delay_step_range": [0, 1],
                "enabled": False,   # ← 第一版关闭延迟
            },
        ),
        "setup_torque_rfi": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:setup_torque_rfi",
            params={
                "enabled": False,
                "rfi_lim": 0.1,
            },
        ),
        "setup_dof_pos_bias": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:setup_dof_pos_bias",
            params={
                "dof_pos_bias_range": [-0.01, 0.01],
                "enabled": False,
            },
        ),
        "actuator_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState",
            params={
                "kp_range": [0.95, 1.05],    # ← 缩小 PD 随机化范围
                "kd_range": [0.95, 1.05],
                "rfi_lim_range": [0.5, 1.5],
                "enable_pd_gain": False,     # ← 第一版关闭 PD 随机化
                "enable_rfi_lim": False,
            },
        ),
        "mass_randomizer": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_mass_startup",
            params={
                "enable_link_mass": False,   # ← 第一版关闭质量随机化
                "link_mass_range": [0.9, 1.2],
                "enable_base_mass": False,
                "added_mass_range": [-1.0, 3.0],
            },
        ),
        "randomize_friction_startup": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_friction_startup",
            params={
                "friction_range": [0.5, 1.25],
                "enabled": False,            # ← 第一版关闭摩擦随机化
            },
        ),
        "randomize_base_com_startup": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_base_com_startup",
            params={
                "base_com_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [-0.05, 0.05]},
                "enabled": False,            # ← 第一版关闭
            },
        ),
    },
    reset_terms={
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
        ),
        "actuator_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:ActuatorRandomizerState"
        ),
        "randomize_push_schedule": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_push_schedule",
        ),
        "randomize_action_delay": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_action_delay",
        ),
        "randomize_dof_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:randomize_dof_state",
            params={
                "joint_pos_scale_range": [0.5, 1.5],
                "joint_pos_bias_range": [0.0, 0.0],
                "joint_vel_range": [0.0, 0.0],
                "randomize_dof_pos_bias": False,
            },
        ),
        "configure_torque_rfi": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:configure_torque_rfi",
        ),
    },
    step_terms={
        "push_randomizer_state": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:PushRandomizerState"
        ),
        "apply_pushes": RandomizationTermCfg(
            func="holosoma.managers.randomization.terms.locomotion:apply_pushes",
        ),
    },
)

__all__ = ["h1_29dof_randomization"]
```

第一版能训练了之后，再逐步打开这些随机化。

#### 5i. `loco/h1/experiment.py`

这是最重要的文件，它把所有子配置拼到一起：

```python
from dataclasses import replace

from holosoma.config_types.experiment import ExperimentConfig, NightlyConfig, TrainingConfig
from holosoma.config_values import (
    action,
    algo,
    command,
    curriculum,
    observation,
    randomization,
    reward,
    robot,
    simulator,
    termination,
    terrain,
)

h1_29dof = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-h1-manager", name="h1_29dof_manager"),
    algo=replace(algo.ppo, config=replace(algo.ppo.config, num_learning_iterations=25000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.h1_29dof,
    terrain=terrain.terrain_locomotion_plane,   # ← 第一版用平地，不是混合地形
    observation=observation.h1_29dof_loco_single_wolinvel,
    action=action.h1_29dof_joint_pos,
    termination=termination.h1_29dof_termination,
    randomization=randomization.h1_29dof_randomization,
    command=command.h1_29dof_command,
    curriculum=curriculum.h1_29dof_curriculum,
    reward=reward.h1_29dof_loco,
    nightly=NightlyConfig(
        iterations=5000,
        metrics={"Episode/rew_tracking_ang_vel": [0.7, "inf"], "Episode/rew_tracking_lin_vel": [0.55, "inf"]},
    ),
)

h1_29dof_fast_sac = ExperimentConfig(
    env_class="holosoma.envs.locomotion.locomotion_manager.LeggedRobotLocomotionManager",
    training=TrainingConfig(project="hv-h1-manager", name="h1_29dof_fast_sac_manager"),
    algo=replace(algo.fast_sac, config=replace(algo.fast_sac.config, num_learning_iterations=50000, use_symmetry=True)),
    simulator=simulator.isaacgym,
    robot=robot.h1_29dof,
    terrain=terrain.terrain_locomotion_plane,   # ← 第一版用平地
    observation=observation.h1_29dof_loco_single_wolinvel,
    action=action.h1_29dof_joint_pos,
    termination=termination.h1_29dof_termination,
    randomization=randomization.h1_29dof_randomization,
    command=command.h1_29dof_command,
    curriculum=curriculum.h1_29dof_curriculum_fast_sac,
    reward=reward.h1_29dof_loco_fast_sac,
    nightly=NightlyConfig(
        iterations=50000,
        metrics={"Episode/rew_tracking_ang_vel": [0.8, "inf"], "Episode/rew_tracking_lin_vel": [0.95, "inf"]},
    ),
)

__all__ = ["h1_29dof", "h1_29dof_fast_sac"]
```

注意：
- `robot=robot.h1_29dof` 中的 `h1_29dof` 是你在 `robot.py` 里定义的变量名
- `observation=observation.h1_29dof_loco_single_wolinvel` 中的名字是你第 5c 步定义的变量名
- 以此类推：每个字段对应第 5x 步里定义的变量名

---

### 第 6 步：在全局注册文件里注册所有 H1 配置

你需要改 8 个全局文件，每个文件里加一行 import 和一行 DEFAULTS 映射。

#### 6a. `config_values/experiment.py`

```python
# 在文件顶部 import 区添加：
from holosoma.config_values.loco.h1.experiment import h1_29dof, h1_29dof_fast_sac

# 在 DEFAULTS 字典里添加：
DEFAULTS = {
    "g1_29dof": g1_29dof,
    "g1_29dof_fast_sac": g1_29dof_fast_sac,
    "t1_29dof": t1_29dof,
    "t1_29dof_fast_sac": t1_29dof_fast_sac,
    "h1_29dof": h1_29dof,                      # ← 添加
    "h1_29dof_fast_sac": h1_29dof_fast_sac,    # ← 添加
    # ... 其他已有项
}
```

**这步之后**，命令行就能识别 `exp:h1-29dof` 和 `exp:h1-29dof-fast-sac` 了（注意 Tyro 会把下划线自动转成连字符）。

#### 6b. `config_values/reward.py`

```python
# import：
from holosoma.config_values.loco.h1.reward import h1_29dof_loco, h1_29dof_loco_fast_sac

# DEFAULTS：
DEFAULTS = {
    # ... 已有项
    "h1_29dof_loco": h1_29dof_loco,
    "h1_29dof_loco_fast_sac": h1_29dof_loco_fast_sac,
}
```

#### 6c. `config_values/termination.py`

```python
# import：
from holosoma.config_values.loco.h1.termination import h1_29dof_termination

# DEFAULTS：
DEFAULTS = {
    # ... 已有项
    "h1_29dof": h1_29dof_termination,
}
```

#### 6d. `config_values/action.py`

```python
# import：
from holosoma.config_values.loco.h1.action import h1_29dof_joint_pos

# DEFAULTS：
DEFAULTS = {
    # ... 已有项
    "h1_29dof_joint_pos": h1_29dof_joint_pos,
}
```

#### 6e. `config_values/observation.py`

```python
# import：
from holosoma.config_values.loco.h1.observation import h1_29dof_loco_single_wolinvel

# DEFAULTS：
DEFAULTS = {
    # ... 已有项
    "h1_29dof_loco_single_wolinvel": h1_29dof_loco_single_wolinvel,
}
```

#### 6f. `config_values/command.py`

```python
# import：
from holosoma.config_values.loco.h1.command import h1_29dof_command

# DEFAULTS：
DEFAULTS = {
    # ... 已有项
    "h1_29dof": h1_29dof_command,
}
```

#### 6g. `config_values/curriculum.py`

```python
# import：
from holosoma.config_values.loco.h1.curriculum import h1_29dof_curriculum, h1_29dof_curriculum_fast_sac

# DEFAULTS：
DEFAULTS = {
    # ... 已有项
    "h1_29dof": h1_29dof_curriculum,
    "h1_29dof_fast_sac": h1_29dof_curriculum_fast_sac,
}
```

#### 6h. `config_values/randomization.py`

```python
# import：
from holosoma.config_values.loco.h1.randomization import h1_29dof_randomization

# DEFAULTS：
DEFAULTS = {
    # ... 已有项
    "h1_29dof": h1_29dof_randomization,
}
```

---

### 第 7 步：Smoke Test — 验证配置能加载

```bash
cd /home/pjm/Desktop/holosoma
source scripts/source_isaacgym_setup.sh

# 小规模测试，1 个环境，不要 headless 方便看
python src/holosoma/holosoma/train_agent.py \
  exp:h1-29dof \
  simulator:isaacgym \
  terrain:terrain-locomotion-plane \
  --training.num-envs 1 \
  --training.headless False
```

**检查清单**：
- [ ] 无 ImportError（如果报找不到模块，检查 import 路径）
- [ ] 无 `Missing default joint angle` 错误
- [ ] 无 `find_rigid_body_indice` 找不到 body 的错误
- [ ] 日志里 `num_dof` 和你预期一致
- [ ] 日志里 DOF names 正确
- [ ] 日志里 Body names 正确
- [ ] feet_indices 都不是 -1
- [ ] termination_contact_indices 都不是 -1
- [ ] viewer 里能看到 H1 机器人
- [ ] 机器人没有穿地或飞走

**常见错误修复**：
- 如果机器人倒下：调整 `init_state.default_joint_angles`，尤其是腿的角度
- 如果机器人抖：降低 `control.stiffness` 的值
- 如果机器人太软站不住：提高 `control.stiffness` 的值
- 如果机器人飞走：看看是不是 `init_state.pos[2]` 太高

---

## 🎯 阶段 C：站立稳定性

**目标**：让 H1 在没有 policy（zero action）的情况下至少站 1 秒。

### 第 8 步：调默认站姿

你需要在 `robot.py` 的 H1 配置里反复调整以下值：

1. **`init_state.pos[2]`** — base 离地高度。需要让脚正好碰到地面。从 URDF 里找到 base 到脚底的距离。
2. **`init_state.default_joint_angles`** — 关节初始角度。腿关节的角度直接决定了站姿。
3. **`control.stiffness`** — P 增益。太大会抖，太小会软。
4. **`control.damping`** — D 增益。太小会震荡，太大会增加延迟感。

调试方法：

```bash
# 打开 viewer，设置 action 全零（默认初始就是）
python src/holosoma/holosoma/train_agent.py \
  exp:h1-29dof \
  simulator:isaacgym \
  terrain:terrain-locomotion-plane \
  --training.num-envs 1 \
  --training.headless False
```

观察：
- 机器人是否一开始就倒？→ 关节角度不对，或者 PD 太小
- 机器人是否抖得厉害？→ PD 太大
- 脚是否腾空？→ `init_state.pos[2]` 太高
- 脚是否穿地？→ `init_state.pos[2]` 太低
- 膝盖方向是否反了？→ URDF 关节轴方向和你想的不一致

---

## 🎯 阶段 D：平地训练

**目标**：训练出一个能在平地慢走的 H1 策略。

### 第 9 步：先跑小规模训练验证

```bash
cd /home/pjm/Desktop/holosoma
source scripts/source_isaacgym_setup.sh

python src/holosoma/holosoma/train_agent.py \
  exp:h1-29dof-fast-sac \
  simulator:isaacgym \
  terrain:terrain-locomotion-plane \
  --training.num-envs 128 \
  --training.headless True \
  --logger.video.enabled False
```

观察：
- [ ] episode length 是否在增长（说明机器人能站住了）
- [ ] reward 是否为 NaN（如果是 NaN，说明某个地方数值爆炸了）
- [ ] `rew_tracking_lin_vel` 是否在增长
- [ ] loss 是否正常下降

如果 reward 是 NaN，检查：
- `dof_pos_lower_limit_list` / `dof_pos_upper_limit_list` 是否正确
- `action_scale` 是否太大（先试试 0.1）
- PD 增益是否太大导致力太大
- 是否某个 joint limit 设反了（lower > upper）

### 第 10 步：正式训练

等小规模测试通过后，加大规模：

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:h1-29dof-fast-sac \
  simulator:isaacgym \
  terrain:terrain-locomotion-plane \
  logger:wandb \
  --training.num-envs 1024 \
  --training.seed 1 \
  --logger.video.enabled False
```

或者用 MJWarp（如果你的 H1 XML 已就绪）：

```bash
source scripts/source_mujoco_setup.sh

python src/holosoma/holosoma/train_agent.py \
  exp:h1-29dof-fast-sac \
  simulator:mjwarp \
  terrain:terrain-locomotion-plane \
  logger:wandb \
  --training.num-envs 1024 \
  --logger.video.enabled False
```

### 第 11 步：导出 ONNX

训练一段时间后，检查 checkpoint：

```bash
# 看 checkpoint 目录
ls -la logs/hv-h1-manager/
```

用 eval 导出 ONNX：

```bash
python src/holosoma/holosoma/eval_agent.py \
  --checkpoint=logs/hv-h1-manager/h1_29dof_fast_sac_manager/<具体checkpoint文件.pt> \
  --training.export_onnx True
```

---

## 🎯 阶段 E：Inference 侧配置

**目标**：让 `run_policy.py` 能加载 H1 的 ONNX 并推理。

### 第 12 步：在 inference 侧新增 H1 RobotConfig

打开文件：
```
src/holosoma_inference/holosoma_inference/config/config_values/robot.py
```

在 `DEFAULTS` 字典前面添加 H1 配置，然后在 DEFAULTS 里注册。

你需要对照 training 侧的 `robot.py` 来填以下字段（每个字段的**顺序**必须和 training 侧一致）：

```python
h1_29dof = RobotConfig(
    robot_type="h1_29dof",
    robot="h1",
    sdk_type="unitree",
    motor_type="serial",
    message_type="HG",          # 或 GO2，取决于你的 H1
    use_sensor=False,
    num_motors=?,               # 和 training 侧 actions_dim 一致
    num_joints=?,                # 和 training 侧 actions_dim 一致
    num_upper_body_joints=?,    # 上身关节数

    # 默认关节角度：顺序必须和 training 侧的 dof_names 顺序一致！
    default_dof_angles=(...),    # 从 robot.py 的 default_joint_angles 按 dof_names 顺序提取
    default_motor_angles=(...),  # 通常和 default_dof_angles 一样

    motor2joint=tuple(range(?)),  # 如果 motor 和 joint 一一对应就是 identity
    joint2motor=tuple(range(?)),

    # 关节名：必须和 training 侧的 dof_names 完全一致
    dof_names=(...),
    dof_names_upper_body=(...),
    dof_names_lower_body=(...),

    torso_link_name="?",         # 和 training 侧一致
    left_hand_link_name=None,
    right_hand_link_name=None,

    unitree_legged_const={...},  # 从 G1 复制
    weak_motor_joint_index={...}, # 按 H1 填
    motion={"body_name_ref": ["your_torso_name"]},  # 填 H1 的 torso 名
)
```

然后在 DEFAULTS 里添加：
```python
DEFAULTS = {
    "g1-29dof": g1_29dof,
    "t1-29dof": t1_29dof,
    "h1-29dof": h1_29dof,       # ← 注意是连字符
}
```

### 第 13 步：在 inference 侧新增 ObservationConfig

打开：
```
src/holosoma_inference/holosoma_inference/config/config_values/observation.py
```

添加：

```python
loco_h1_29dof = ObservationConfig(
    obs_dict={
        "actor_obs": [
            "base_ang_vel",
            "projected_gravity",
            "command_lin_vel",
            "command_ang_vel",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase",
        ]
    },
    obs_dims={
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "command_lin_vel": 2,
        "command_ang_vel": 1,
        "dof_pos": ?,     # ← H1 的 DOF 数量
        "dof_vel": ?,     # ← H1 的 DOF 数量
        "actions": ?,     # ← H1 的 DOF 数量
        "sin_phase": 2,
        "cos_phase": 2,
    },
    obs_scales={
        "base_lin_vel": 2.0,
        "base_ang_vel": 0.25,
        "projected_gravity": 1.0,
        "command_lin_vel": 1.0,
        "command_ang_vel": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
        "actions": 1.0,
        "sin_phase": 1.0,
        "cos_phase": 1.0,
    },
    history_length_dict={
        "actor_obs": 1,
    },
)
```

在 DEFAULTS 里注册：
```python
DEFAULTS = {
    "loco-g1-29dof": loco_g1_29dof,
    "loco-t1-29dof": loco_t1_29dof,
    "loco-h1-29dof": loco_h1_29dof,   # ← 添加
    "wbt": wbt,
}
```

**最重要的一致性检查**：

Training 侧的 actor_obs 顺序（`loco/h1/observation.py`）：

```
base_ang_vel → projected_gravity → command_lin_vel → command_ang_vel → dof_pos → dof_vel → actions → sin_phase → cos_phase
```

Inference 侧的 actor_obs 顺序（`observation.py`）必须**一模一样**。顺序不同会导致 ONNX 推理结果正确但动作完全错误。

### 第 14 步：在 inference 侧新增 InferenceConfig

打开：
```
src/holosoma_inference/holosoma_inference/config/config_values/inference.py
```

添加：

```python
h1_29dof_loco = InferenceConfig(
    robot=robot.h1_29dof,
    observation=observation.loco_h1_29dof,
    task=task.locomotion,
)
```

在 DEFAULTS 里注册：
```python
DEFAULTS = {
    "g1-29dof-loco": g1_29dof_loco,
    "t1-29dof-loco": t1_29dof_loco,
    "g1-29dof-wbt": g1_29dof_wbt,
    "h1-29dof-loco": h1_29dof_loco,        # ← 添加
}
```

### 第 15 步：调整 TaskConfig（如果需要）

打开：
```
src/holosoma_inference/holosoma_inference/config/config_values/task.py
```

H1 可以复用已有的 `locomotion` task config。但如果你的 H1 base 高度和 G1 不同，需要改 `desired_base_height`：

```python
# 如果 H1 的 base 高度是 0.8（举例），可以创建专用的 task：
h1_locomotion = TaskConfig(
    model_path="",
    rl_rate=50,
    policy_action_scale=0.25,
    use_phase=True,
    gait_period=1.0,
    desired_base_height=0.8,  # ← H1 的实际 base 高度
    residual_upper_body_action=False,
    domain_id=0,
    interface="lo",
    velocity_input="keyboard",
    state_input="keyboard",
    joystick_type="xbox",
    joystick_device=0,
)
```

---

## 🎯 阶段 F：Sim2Sim

**目标**：在 MuJoCo 仿真里跑 H1 的 ONNX 策略。

### 第 16 步：跑 sim2sim

终端 A（仿真端）：

```bash
cd /home/pjm/Desktop/holosoma
source scripts/source_mujoco_setup.sh

python src/holosoma/holosoma/run_sim.py robot:h1-29dof \
  --simulator.config.bridge.enabled True \
  --simulator.config.bridge.interface lo \
  --simulator.config.bridge.domain-id 7
```

终端 B（推理端）：

```bash
cd /home/pjm/Desktop/holosoma
source scripts/source_inference_setup.sh

python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:h1-29dof-loco \
  --task.model-path <你的_h1_policy.onnx> \
  --task.interface lo \
  --task.domain-id 7 \
  --task.no-use-joystick
```

### 第 17 步：排查 sim2sim 问题

如果机器人一动就倒：

| 现象 | 可能原因 | 排查方法 |
|------|----------|----------|
| 完全不动/抽搐 | 推理端没发出命令 | 检查 domain-id 是否一致 |
| 动作幅度太大 | `action_scale` 太大 | 降低 `policy_action_scale` |
| 动作方向反了 | 某个关节符号反了 | 对照 training 侧 `flip_sign_joint_names` |
| 关节顺序错了 | inference `dof_names` 顺序和 training 不一致 | 逐一对照 |
| obs 拼错了 | `obs_dict.actor_obs` 顺序错 | 对照 training `observation.py` |
| PD 不对 | inference 侧 kp/kd 和 training 侧不一致 | 对照 `stiffness`/`damping` |
| base 高度不对 | `desired_base_height` 不对 | 改 task config |

---

## 🎯 阶段 G：真机部署（roadmap，先不做）

真机需要额外注意：
1. 真实的 H1 motor 顺序是否和你的 `dof_names` 一致
2. 安全站立模式和急停
3. 关节限位保护
4. 从站立状态平滑切到 policy 控制
5. 真实硬件的 kp/kd 限制

---

## 📋 完整检查清单

### 资产和 RobotConfig
- [ ] `data/robots/h1/h1.urdf` 存在
- [ ] `data/robots/h1/h1.xml` 存在（MuJoCo 用）
- [ ] `data/robots/h1/meshes/` 有所有文件
- [ ] `robot.py` 里 H1 的 `dof_names` 和 URDF 完全一致
- [ ] `robot.py` 里 H1 的 `body_names` 和 URDF 完全一致
- [ ] `robot.py` 里 `default_joint_angles` 涵盖了所有 dof_names
- [ ] `robot.py` 里 `dof_pos_lower_limit_list` 个数 = len(dof_names)
- [ ] `robot.py` 里 H1 已加入 `DEFAULTS`

### loco/h1/ 配置
- [ ] `loco/h1/` 目录存在且有 9 个文件
- [ ] `action.py` 变量名已改为 `h1_29dof_joint_pos`
- [ ] `observation.py` 变量名已改为 `h1_29dof_loco_single_wolinvel`
- [ ] `reward.py` 里 `pose_weights` 长度 = `actions_dim`
- [ ] `termination.py` 变量名已改为 `h1_29dof_termination`
- [ ] `command.py` 速度范围已调小（适合第一版）
- [ ] `randomization.py` 随机化已关闭/减小（适合第一版）
- [ ] `experiment.py` 用 `terrain_locomotion_plane`（不是 mix）

### 全局注册
- [ ] `experiment.py` 已 import + DEFAULTS
- [ ] `reward.py` 已 import + DEFAULTS
- [ ] `termination.py` 已 import + DEFAULTS
- [ ] `action.py` 已 import + DEFAULTS
- [ ] `observation.py` 已 import + DEFAULTS
- [ ] `command.py` 已 import + DEFAULTS
- [ ] `curriculum.py` 已 import + DEFAULTS
- [ ] `randomization.py` 已 import + DEFAULTS

### 验证
- [ ] `exp:h1-29dof` 能被 tyro 识别
- [ ] IsaacGym 能加载 H1
- [ ] feet_indices 能找到
- [ ] termination_contact_indices 能找到
- [ ] 小规模训练能跑（128 envs）
- [ ] episode length 在增长
- [ ] reward 不为 NaN

### Inference
- [ ] `robot.py`（inference 侧）有 H1 RobotConfig
- [ ] `observation.py`（inference 侧）有 H1 ObservationConfig
- [ ] `inference.py`（inference 侧）有 H1 InferenceConfig
- [ ] `dof_names` 顺序 training 和 inference 完全一致
- [ ] `obs_dict.actor_obs` 顺序 training 和 inference 完全一致
- [ ] ONNX 已导出

### Sim2Sim
- [ ] `domain-id` 两端一致
- [ ] `interface` 两端一致
- [ ] H1 能在 MuJoCo 里被 policy 控制行走
- [ ] 动作方向和幅度正常

---

## 🔧 快速命令参考

```bash
# ====== 环境激活 ======
# IsaacGym
source scripts/source_isaacgym_setup.sh
# MuJoCo/MJWarp
source scripts/source_mujoco_setup.sh
# Inference
source scripts/source_inference_setup.sh

# ====== 只加载仿真（不训练） ======
python src/holosoma/holosoma/run_sim.py robot:h1-29dof \
  simulator:mujoco \
  terrain:terrain-locomotion-plane

# ====== 小规模 smoke test ======
python src/holosoma/holosoma/train_agent.py \
  exp:h1-29dof \
  simulator:isaacgym \
  terrain:terrain-locomotion-plane \
  --training.num-envs 1 \
  --training.headless False

# ====== 正式训练 ======
python src/holosoma/holosoma/train_agent.py \
  exp:h1-29dof-fast-sac \
  simulator:isaacgym \
  terrain:terrain-locomotion-plane \
  logger:wandb \
  --training.num-envs 1024 \
  --training.seed 1

# ====== Sim2Sim 仿真端 ======
python src/holosoma/holosoma/run_sim.py robot:h1-29dof \
  --simulator.config.bridge.enabled True \
  --simulator.config.bridge.interface lo \
  --simulator.config.bridge.domain-id 7

# ====== Sim2Sim 推理端 ======
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:h1-29dof-loco \
  --task.model-path <你的.onnx> \
  --task.interface lo \
  --task.domain-id 7 \
  --task.no-use-joystick
```

---

## ❓ 常见问题

**Q: 什么时候可以加混合地形？**
A: 等 `terrain:terrain-locomotion-plane` 训练出能稳定行走的策略后，再切到 `terrain:terrain-locomotion-mix`。

**Q: 什么时候可以打开 domain randomization？**
A: 等平地能走了，再逐步打开 randominzation 里的各个开关。

**Q: inference 侧启动报 `KeyError: 'h1-29dof'`？**
A: 确认你在 `robot.py`（inference 侧）和 `observation.py` 的 `DEFAULTS` 里都注册了。注意 inference 侧用连字符 `h1-29dof` 不是下划线。

**Q: 训练时 reward 一直是 NaN？**
A: 通常是一个数值问题。检查 joint limit 是否正确、action_scale 是否太大。试试点 `action_scale=0.1`。

**Q: 机器人一直倒怎么办？**
A: 先确认 zero action 能不能站住（阶段 C）。如果 zero action 就倒，那 RL 几乎不可能训练出来。
