# Franka Modular Controller

一个面向 Isaac Sim 中 Franka / Panda / FR3 机械臂的 PyQt6 + ROS 2 外部控制实验项目。

项目由两部分组成：

- 根目录 `main.py`：启动 Isaac Sim，加载 Franka，启用 ROS2 Bridge，并创建 `/joint_states` 与 `/joint_command` 的 Action Graph。
- `franka_modular_controller/main.py`：启动 PyQt6 控制界面，订阅关节状态，计算运动学与控制命令，并向 Isaac Sim 发布控制消息。

当前主界面支持两种末端点动控制：

- `Pose IK Jog`：修改目标末端位姿，通过 IK 求解 `q_goal`，发布关节位置命令。
- `Twist Velocity Jog`：生成末端速度 `twist`，通过几何 Jacobian 求解 `dq_cmd`，发布关节速度命令。

本项目主要用于仿真联调和控制算法验证，不建议直接连接真实机械臂。真实机械臂需要额外加入速度、加速度、jerk、碰撞检测、软硬限位、急停、watchdog、控制器状态检查等安全机制。

## 目录结构

```text
.
├── main.py
├── README.md
├── ros2_topic_monitor_ui.py
├── ros2_force_control/
│   ├── cartesian_nullspace_pyqt6.py
│   └── joint_pd_effort.py
└── franka_modular_controller/
    ├── main.py
    ├── config/
    │   ├── controller_config.py
    │   ├── fr3.urdf
    │   ├── fr3_robot_description.yaml
    │   └── meshes/
    ├── core/
    │   ├── kinematics.py
    │   ├── math_utils.py
    │   └── robot_state.py
    ├── controllers/
    │   ├── base.py
    │   ├── cartesian_pose_ik.py
    │   ├── cartesian_twist_controller.py
    │   ├── cartesian_nullspace_effort.py
    │   ├── joint_position.py
    │   └── admittance_placeholder.py
    ├── ros_interface/
    │   └── franka_ros_node.py
    ├── ui/
    │   ├── main_window.py
    │   └── styles.py
    └── utils/
```

## 整体架构

```text
Isaac Sim Franka
    │
    │ publish /joint_states
    ↓
FrankaRosNode
    ↓
RobotStateBuffer
    ├── UI 状态显示
    ├── CartesianPoseIKController
    │       ↓
    │   q_goal
    │       ↓
    │   publish_position(q_goal)
    │
    └── CartesianTwistController
            ↓
        dq_cmd
            ↓
        publish_velocity(dq_cmd)

/joint_command
    ↓
Isaac Sim ArticulationController
```

`franka_modular_controller/main.py` 的启动链路：

```text
AppConfig
    ↓
rclpy.init()
    ↓
RobotStateBuffer
    ↓
FrankaRosNode
    ↓
PandaKinematics(FR3 URDF)
    ↓
CartesianPoseIKController
    ↓
CartesianNullspaceWindow
    ↓
Qt event loop
```

## 核心模块

### `config/controller_config.py`

集中管理 ROS 话题、关节名、安全限幅、阻抗参数、零空间参数和 UI 刷新参数。

当前默认 ROS 配置：

```python
node_name = "franka_modular_pyqt6_controller"
joint_state_topic = "/joint_states"
joint_command_topic = "/joint_command"
joint_names = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
]
```

注意：运动学模块当前读取的是 FR3 URDF，默认 active joints 是 `fr3_joint1` 到 `fr3_joint7`。如果 Isaac Sim 发布的 `/joint_states.name` 是 `fr3_joint*`，需要把 `RobotConfig.joint_names` 改成同样的名字和顺序。关节名顺序必须同时匹配 `/joint_states`、`/joint_command`、URDF 和 Isaac Sim ArticulationController。

### `core/robot_state.py`

`RobotStateBuffer` 缓存最新的 `sensor_msgs/JointState`。

它把 ROS 消息中的：

```text
msg.name
msg.position
msg.velocity
```

转换成按 `RobotConfig.joint_names` 排列的 `q` 和 `dq` 数组，供 UI 和控制器读取。

### `core/kinematics.py`

`PandaKinematics` 是当前运动学核心。虽然类名保留为 Panda，但实际默认加载 FR3 相关文件：

```python
urdf_path="config/fr3.urdf"
robot_description_yaml_path="config/fr3_robot_description.yaml"
base_link="fr3_link0"
end_link="fr3_hand_tcp"
```

主要能力：

- `fk(q)`：末端位置。
- `fk_transform(q)`：末端 4x4 齐次变换。
- `fk_pose6(q)`：末端 `[x, y, z, rx, ry, rz]`。
- `numerical_jacobian(q)`：3x7 位置数值 Jacobian。
- `numerical_pose_jacobian(q)`：6x7 位姿数值 Jacobian，用于 IK。
- `geometric_jacobian(q)`：6x7 几何 Jacobian，用于 Twist 速度控制。
- `clamp_to_joint_limits(q)`：根据 URDF 限制关节角。

### `ros_interface/franka_ros_node.py`

`FrankaRosNode` 是 ROS 2 通信层，不负责控制算法。

订阅：

```text
/joint_states
```

发布：

```text
/joint_command
```

当前提供的发布接口：

```python
publish_position(position)
publish_position_with_velocity(position, velocity)
publish_velocity(velocity)
publish_effort(effort)
publish_zero_velocity()
publish_zero_effort()
```

其中 `publish_velocity()` 会发布 `JointState.velocity`，用于当前 UI 的 `Twist Velocity Jog`。

### `controllers/cartesian_pose_ik.py`

`CartesianPoseIKController` 负责末端 6D 位姿 IK。

输入：

```python
target_pose6 = [x, y, z, rx, ry, rz]
```

输出：

```python
q_goal
```

求解方法是阻尼最小二乘：

```text
err = target_pose - current_pose
J = numerical_pose_jacobian(q)
dq = J.T @ solve(J @ J.T + lambda^2 I, err)
q = q + clip(dq)
```

该控制器只求解 IK，不直接发布 ROS 命令。

### `controllers/cartesian_twist_controller.py`

`CartesianTwistController` 负责末端速度到关节速度的转换。

输入：

```python
twist = [vx, vy, vz, wx, wy, wz]
```

控制关系：

```text
twist = J(q) @ dq
```

求解方法：

```text
dq_cmd = J.T @ solve(J @ J.T + lambda^2 I, twist)
```

随后进行关节速度限幅，并计算可选的积分目标 `q_target`。当前 UI 的 Twist 模式使用的是 `dq_cmd`，直接调用：

```python
ros_node.publish_velocity(dq_cmd)
```

### `controllers/cartesian_nullspace_effort.py`

预留的 effort 控制器，用于末端位置保持、零空间姿态恢复和扰动力矩实验。当前主界面没有把它作为主要控制链路使用。

### `ui/main_window.py`

`CartesianNullspaceWindow` 是当前 PyQt6 主窗口。

主要功能：

- 显示当前关节角、目标关节角和误差。
- 显示当前末端 pose6。
- 输入目标末端 pose6。
- 执行 `Move To Target Pose`。
- 支持 X/Y/Z/Rx/Ry/Rz 点动按钮。
- 支持 `Pose IK Jog` 和 `Twist Velocity Jog`。
- 支持重复发布最后一次 `q_goal`。
- 支持发布零速度和零 effort。

UI 内部使用多个 `QTimer`：

```text
ros_timer   : 周期 spin_once，处理 ROS 回调
ui_timer    : 周期刷新界面显示
hold_timer  : 可选重复发布 q_goal
jog_timer   : 按住 Jog 按钮时周期执行点动控制
```

## 控制模式

### Pose IK Jog

流程：

```text
按住 X+/Y-/Rz+
    ↓
修改目标 pose6
    ↓
调用 IK 求解 q_goal
    ↓
发布 JointState.position
```

适合慢速位姿调试、目标姿态设置、IK 验证。

### Twist Velocity Jog

流程：

```text
按住 X+/Y-/Rz+
    ↓
生成末端速度 twist
    ↓
用 geometric_jacobian(q_current) 求 dq_cmd
    ↓
发布 JointState.velocity
```

当前代码中，Twist Jog 已经是速度命令发布链路。松开 Jog 按钮时，UI 会调用 `publish_zero_velocity()`，避免底层继续执行上一帧速度命令。

## Isaac Sim 启动脚本

根目录 `main.py` 的职责：

```text
创建 SimulationApp
    ↓
启用 isaacsim.ros2.bridge
    ↓
创建 World 和地面
    ↓
加载 Franka
    ↓
设置关节 drive 参数
    ↓
创建 ROS2 Action Graph
    ↓
发布 /joint_states
    ↓
订阅 /joint_command
    ↓
ArticulationController 执行 position / velocity / effort 命令
```

Action Graph 连接了三类命令：

```text
ROS2SubscribeJointState.outputs:positionCommand
    → IsaacArticulationController.inputs:positionCommand

ROS2SubscribeJointState.outputs:velocityCommand
    → IsaacArticulationController.inputs:velocityCommand

ROS2SubscribeJointState.outputs:effortCommand
    → IsaacArticulationController.inputs:effortCommand
```

当前根目录脚本默认调用 `set_franka_drive_to_velocity_mode()`，用于配合速度命令测试。

## 运行方式

### 1. 启动 Isaac Sim 仿真

在 Isaac Sim 环境中运行根目录 `main.py`：

```bash
source /opt/ros/humble/setup.bash
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/exts/isaacsim.ros2.bridge/humble/lib

./python.sh ~/Project/VS_Project/Franka/main.py
```

启动后检查话题：

```bash
ros2 topic list
```

期望包含：

```text
/joint_states
/joint_command
```

查看关节状态：

```bash
ros2 topic echo /joint_states
```

### 2. 启动 PyQt6 控制界面

进入控制器目录：

```bash
cd ~/Project/VS_Project/Franka/franka_modular_controller
source /opt/ros/humble/setup.bash
python3 main.py
```

需要确保当前 Python 环境中可用：

```text
PyQt6
numpy
rclpy
sensor_msgs
```

## 使用流程

1. 先启动 Isaac Sim 仿真。
2. 确认 `/joint_states` 正常发布。
3. 启动 PyQt6 控制界面。
4. 点击 `Read Current Joint Angles` 检查关节状态读取是否正常。
5. 点击 `Fill Target From Current Pose` 把当前末端位姿填入目标输入框。
6. 使用 `Move To Target Pose` 执行一次 IK 位姿移动。
7. 切换 `Control Mode`，使用 X/Y/Z/Rx/Ry/Rz 按钮进行点动控制。

## UI 控件

| 控件 | 作用 |
|---|---|
| Read Current Joint Angles | 读取当前关节角 |
| Fill Target From Current Pose | 把当前末端位姿填入目标输入框 |
| Move To Target Pose | 对目标末端位姿求 IK，并发布 `q_goal` |
| Publish Last q_goal Again | 再次发布上一次 `q_goal` |
| Zero Effort | 发布全 0 effort |
| Control Mode | 切换 `Pose IK Jog` / `Twist Velocity Jog` |
| Hold target by republishing q_goal | 在 Pose IK 模式下周期性重复发布 `q_goal` |
| Pose step / m | Pose IK Jog 每次位置增量 |
| Pose rot step / rad | Pose IK Jog 每次角度增量 |
| Twist linear / m/s | Twist Jog 线速度 |
| Twist angular / rad/s | Twist Jog 角速度 |
| Jog rate / Hz | 按住 Jog 按钮时的控制频率 |

## 当前能力状态

| 能力 | 状态 |
|---|---|
| Isaac Sim 加载 Franka | 已实现 |
| ROS2 Bridge Action Graph | 已实现 |
| 发布 `/joint_states` | 已实现 |
| 订阅 `/joint_command` | 已实现 |
| PyQt6 控制界面 | 已实现 |
| `/joint_states` 状态缓存 | 已实现 |
| 末端正运动学 | 已实现 |
| 末端 6D IK | 已实现 |
| 数值位姿 Jacobian | 已实现 |
| 几何 Jacobian | 已实现 |
| 关节位置命令 | 已实现 |
| 关节速度命令 | 已实现 |
| 关节 effort 命令接口 | 已实现 |
| Twist Velocity Jog | 已实现，发布 `JointState.velocity` |
| Cartesian nullspace effort | 已有控制器，未接入主 UI 控制流程 |
| Admittance control | 占位接口 |
| MoveIt Servo | 未接入 |
| 真实机械臂安全保护 | 未完善 |

## 注意事项

- `RobotConfig.joint_names` 必须和 Isaac Sim 发布的 `/joint_states.name` 完全一致。
- 当前运动学模型默认使用 FR3 URDF，但根目录 Isaac Sim 脚本加载的是 Isaac Sim 示例 Franka。实际运行时需要确认两侧关节命名、关节顺序、末端 frame 是否一致。
- `Pose IK Jog` 的姿态误差当前基于 RPY 差值，适合小角度调试；大角度姿态控制建议改为 SO(3) log map 或 rotation vector error。
- `Twist Velocity Jog` 依赖 Isaac Sim 侧 `velocityCommand` 是否被正确接收和执行。如果按钮松开后仍运动，应检查 `publish_zero_velocity()` 是否到达 `/joint_command`。
- `Zero Effort` 只清零 effort 字段；在 position 或 velocity drive 模式下，它不等价于停止所有运动。

## 后续改进方向

- 统一 Panda / FR3 命名，避免 `panda_joint*` 与 `fr3_joint*` 混用。
- 将 UI 中的控制模式逻辑拆成独立 control manager，减少 `main_window.py` 的职责。
- 为 IK 使用 rotation vector 或 SO(3) log map 姿态误差。
- 增加奇异点检测、速度缩放和关节限位裕度。
- 增加命令 watchdog，超时自动发布零速度。
- 将 nullspace effort / admittance 控制器接入主界面。
- 接入 MoveIt Servo，用于更成熟的 Twist Jog 控制链路。

## 项目描述

中文：

```text
一个面向 Isaac Sim 中 Franka / Panda / FR3 机械臂的模块化 PyQt6 + ROS 2 控制实验项目，支持末端位姿 IK 点动和基于几何 Jacobian 的末端速度点动，并通过 ROS2 Bridge 向 Isaac Sim 发布关节位置、速度或 effort 命令。
```

English:

```text
A modular PyQt6 + ROS 2 controller for Franka / Panda / FR3 simulation in Isaac Sim. It supports Cartesian pose IK jog and geometric-Jacobian-based twist velocity jog, publishing joint position, velocity, or effort commands through the Isaac Sim ROS2 Bridge.
```

推荐标签：

```text
franka, panda, fr3, ros2, pyqt6, isaac-sim, robot-control, inverse-kinematics, jacobian, cartesian-control
```
