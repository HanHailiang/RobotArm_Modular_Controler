# Franka Modular Controller

一个面向 Franka / FR3 / Panda 机械臂仿真控制的模块化 PyQt6 + ROS 2 控制程序。项目提供图形化界面，用于读取机械臂关节状态、显示末端位姿、输入目标末端位姿，并通过两种方式控制机械臂运动：

1. **Pose IK Jog**：末端位姿增量控制，通过 IK 求解目标关节角，然后发布关节位置命令。
2. **Twist Jog**：末端速度式点动控制，通过 Jacobian 伪逆把末端 Twist 转换为关节速度，再积分为关节位置目标，最后发布关节位置命令。

> 当前版本的 Twist Jog 是“末端速度计算 + 关节位置目标流式执行”，并不是严格意义上的底层 velocity controller。代码中会计算 `dq_cmd`，但最终仍然通过 `/joint_command` 发布 `JointState.position`。

---

## 1. 项目定位

本项目适合用于：

- Isaac Sim 中 Franka / Panda / FR3 机械臂的外部控制测试；
- ROS 2 与 PyQt6 图形界面联调；
- 末端 6D 位姿 IK 控制实验；
- 基于 Jacobian 的末端 Twist Jog 控制实验；
- 后续扩展关节速度控制、关节力矩控制、阻抗控制、零空间控制等高级控制算法。

当前项目更偏向**仿真控制与算法验证**，不建议直接连接真实机械臂使用。若用于真实机械臂，需要额外加入速度、加速度、jerk、碰撞检测、软硬限位、急停、watchdog 等安全机制。

---

## 2. 代码目录结构

```text
franka_modular_controller/
├── main.py
├── config/
│   ├── controller_config.py
│   ├── fr3.urdf
│   ├── fr3_robot_description.yaml
│   └── meshes/
├── core/
│   ├── kinematics.py
│   ├── robot_state.py
│   └── math_utils.py
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

根目录下还有一个 `main.py`，用于启动 Isaac Sim、加载 Franka、创建 ROS 2 Action Graph，并建立 `/joint_states` 与 `/joint_command` 话题桥接。

---

## 3. 整体架构

```text
Isaac Sim / ROS 2
        │
        │ /joint_states
        ↓
FrankaRosNode
        │
        ↓
RobotStateBuffer
        │
        ├──────────────→ UI 状态显示
        │
        ├──────────────→ CartesianPoseIKController
        │                       │
        │                       ↓
        │                   q_goal
        │
        └──────────────→ CartesianTwistController
                                │
                                ↓
                         dq_cmd → q_target
                                │
                                ↓
FrankaRosNode.publish_position(q_goal / q_target)
        │
        │ /joint_command
        ↓
Isaac Sim ArticulationController
```

程序主入口 `franka_modular_controller/main.py` 的启动流程：

```text
AppConfig
   ↓
rclpy.init()
   ↓
RobotStateBuffer
   ↓
FrankaRosNode
   ↓
PandaKinematics / FR3 URDF
   ↓
CartesianPoseIKController
   ↓
CartesianNullspaceWindow
   ↓
Qt event loop
```

---

## 4. 核心模块说明

### 4.1 `config/controller_config.py`

负责集中管理机器人、ROS 话题、安全参数和 UI 参数。

主要配置包括：

```python
node_name = "franka_modular_pyqt6_controller"
joint_state_topic = "/joint_states"
joint_command_topic = "/joint_command"
joint_names = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
]
```

注意：当前默认关节名是 `panda_joint1` 到 `panda_joint7`。如果你的 Isaac Sim 或 URDF 使用的是 `fr3_joint1` 到 `fr3_joint7`，需要把这里统一修改为：

```python
joint_names = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]
```

关节名称顺序必须与 `/joint_states`、URDF、Isaac Sim Articulation Controller 完全一致。

---

### 4.2 `core/robot_state.py`

`RobotStateBuffer` 用于缓存最新的机器人关节状态。

ROS 回调收到 `/joint_states` 后，会将：

```text
msg.name
msg.position
msg.velocity
msg.effort
```

解析并存入状态缓存。UI 和控制器再从该缓存读取当前关节角 `q_current` 和关节速度 `dq_current`。

---

### 4.3 `core/kinematics.py`

`PandaKinematics` 负责运动学计算。虽然类名叫 `PandaKinematics`，但当前加载的是 FR3 相关文件：

```python
urdf_path="config/fr3.urdf"
robot_description_yaml_path="config/fr3_robot_description.yaml"
base_link="fr3_link0"
end_link="fr3_hand_tcp"
```

主要能力包括：

- 正运动学 `fk(q)`；
- 齐次变换 `fk_transform(q)`；
- 末端 6D 位姿 `fk_pose6(q)`；
- 数值 Jacobian `numerical_pose_jacobian(q)`；
- 几何 Jacobian `geometric_jacobian(q)`；
- 关节限位裁剪 `clamp_to_joint_limits(q)`。

其中 `geometric_jacobian(q)` 是 Twist Jog 控制的关键函数。

---

### 4.4 `ros_interface/franka_ros_node.py`

`FrankaRosNode` 是 ROS 2 通信层，只负责订阅和发布，不负责控制算法。

订阅：

```text
/joint_states
```

发布：

```text
/joint_command
```

当前提供 4 类发布接口：

```python
publish_position(position)
publish_position_with_velocity(position, velocity)
publish_effort(effort)
publish_zero_effort()
```

当前 UI 主流程主要调用的是：

```python
publish_position(q_goal)
```

因此当前主控制链路是**关节位置命令控制**。

---

### 4.5 `controllers/cartesian_pose_ik.py`

`CartesianPoseIKController` 用于末端 6D 位姿 IK 求解。

输入：

```python
target_pose6 = [x, y, z, rx, ry, rz]
```

其中：

```text
x/y/z 单位：m
rx/ry/rz 单位：rad
```

输出：

```python
q_goal
```

求解流程：

```text
current_pose = fk_pose6(q)
error = target_pose - current_pose
J = numerical_pose_jacobian(q)
dq = J.T @ inv(J @ J.T + lambda² I) @ error
q = q + dq
```

该控制器只负责求解 IK，不直接发布 ROS 命令。

---

### 4.6 `controllers/cartesian_twist_controller.py`

`CartesianTwistController` 是末端 Twist Jog 控制器。

输入：

```python
twist = [vx, vy, vz, wx, wy, wz]
```

其中：

```text
vx/vy/vz 单位：m/s
wx/wy/wz 单位：rad/s
```

核心控制关系：

```text
twist = J(q) · dq
```

使用阻尼最小二乘伪逆：

```text
dq = Jᵀ · inv(J · Jᵀ + λ²I) · twist
```

然后限速并积分：

```text
q_target = q_integrator + dq_cmd · dt
```

注意：这个控制器会计算 `dq_cmd`，但当前 UI 没有把 `dq_cmd` 作为 velocity command 发布，而是把积分后的 `q_target` 作为 position command 发布。

---

### 4.7 `ui/main_window.py`

`CartesianNullspaceWindow` 是 PyQt6 主界面。

主要功能：

- 显示当前关节角；
- 显示目标关节角；
- 显示关节误差；
- 显示当前末端位姿；
- 输入目标末端位姿；
- 一键读取当前关节角；
- 一键填充当前末端位姿；
- 执行 Move To Target Pose；
- 支持 X/Y/Z/Rx/Ry/Rz 点动按钮；
- 支持 Pose IK Jog 和 Twist Jog 两种模式；
- 支持重复发布最后一次 `q_goal`；
- 支持 Zero Effort。

UI 内部使用多个定时器：

```text
ros_timer   ：周期 spin_once，处理 ROS 回调
ui_timer    ：周期刷新 UI 显示
hold_timer  ：可选重复发布 q_goal
jog_timer   ：按住 Jog 按钮时周期执行点动控制
```

---

## 5. 两种控制模式

### 5.1 Pose IK Jog

Pose IK Jog 的流程是：

```text
点击 / 按住 X+/Y-/Rz+
        ↓
修改 target_pose6
        ↓
调用 IK 求解 q_goal
        ↓
publish_position(q_goal)
```

特点：

- 控制目标是末端位姿；
- 每次点动都会修改目标 pose；
- 通过 IK 得到目标关节角；
- 最终发布的是关节位置命令。

适合：

- 慢速位姿调试；
- 设置目标末端位置；
- 验证 IK 是否正常。

---

### 5.2 Twist Jog

Twist Jog 的流程是：

```text
点击 / 按住 X+/Y-/Rz+
        ↓
生成末端速度 twist
        ↓
使用 geometric_jacobian(q_current)
        ↓
阻尼最小二乘求 dq_cmd
        ↓
q_target = q_integrator + dq_cmd * dt
        ↓
publish_position(q_target)
```

特点：

- 上层输入是末端速度；
- 内部计算关节速度 `dq_cmd`；
- 使用积分目标 `twist_q_target` 保证运动连续；
- 最终仍然发布关节位置命令。

因此当前 Twist Jog 更准确的定义是：

> 基于 Jacobian 的末端速度点动控制，底层通过关节位置命令执行。

不是严格意义上的：

> 真正的关节速度控制器。

---

## 6. 当前是否是真正的速度控制？

不是。

当前代码确实计算了末端速度对应的关节速度：

```python
dq_cmd = J.T @ solve(J @ J.T + damping² I, twist)
```

但最终执行时调用的是：

```python
self.ros_node.publish_position(self.q_goal.tolist())
```

也就是发布：

```python
cmd.position = list(position)
cmd.velocity = []
cmd.effort = []
```

所以当前执行链路是：

```text
末端 Twist
   ↓
关节速度 dq_cmd
   ↓
积分成关节位置 q_target
   ↓
发布 JointState.position
```

如果要改成真正的速度控制，应该变成：

```text
末端 Twist
   ↓
关节速度 dq_cmd
   ↓
发布 velocity command
```

例如增加：

```python
def publish_velocity(self, velocity):
    cmd = JointState()
    cmd.header.stamp = self.get_clock().now().to_msg()
    cmd.name = self.robot_cfg.joint_names
    cmd.position = []
    cmd.velocity = list(velocity)
    cmd.effort = []
    self.publisher.publish(cmd)
```

然后在 Twist Jog 中改为：

```python
self.ros_node.publish_velocity(twist_result.dq_cmd.tolist())
```

但前提是 Isaac Sim 或 ros2_control 底层真的接收并执行 `velocity` 字段。

---

## 7. Isaac Sim 启动脚本

根目录下的 `main.py` 用于启动 Isaac Sim 仿真。主要流程：

```text
SimulationApp
   ↓
启用 isaacsim.ros2.bridge
   ↓
创建 World
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
ArticulationController 执行命令
```

Action Graph 中的核心连接：

```text
ROS2SubscribeJointState.outputs:positionCommand
        ↓
IsaacArticulationController.inputs:positionCommand

ROS2SubscribeJointState.outputs:velocityCommand
        ↓
IsaacArticulationController.inputs:velocityCommand

ROS2SubscribeJointState.outputs:effortCommand
        ↓
IsaacArticulationController.inputs:effortCommand
```

这说明 Isaac Sim 侧理论上同时连接了 position / velocity / effort command，但当前 PyQt 控制器主路径主要发布的是 position command。

---

## 8. 运行方法

### 8.1 启动 Isaac Sim 仿真

在 Isaac Sim 环境中运行根目录 `main.py`：

```bash
source /opt/ros/humble/setup.bash
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/exts/isaacsim.ros2.bridge/humble/lib

./python.sh ~/Project/VS_Project/Franka/main.py
```

启动成功后，应能看到话题：

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

---

### 8.2 启动 PyQt6 控制界面

进入控制器目录：

```bash
cd franka_modular_controller
```

运行：

```bash
source /opt/ros/humble/setup.bash
python3 main.py
```

如果使用 conda / mamba 环境，需要确保 PyQt6、numpy、rclpy 等依赖可用。

---

## 9. 使用说明

### 9.1 基础操作

1. 先启动 Isaac Sim 仿真；
2. 确认 `/joint_states` 正常发布；
3. 启动 PyQt6 控制界面；
4. 点击 `Read Current Joint Angles` 查看当前关节角；
5. 点击 `Fill Target From Current Pose` 将当前末端位姿填入目标输入框；
6. 修改目标位姿后点击 `Move To Target Pose`；
7. 或使用 X/Y/Z/Rx/Ry/Rz 按钮进行 Jog 控制。

---

### 9.2 UI 控件说明

| 控件 | 作用 |
|---|---|
| Read Current Joint Angles | 读取当前关节角 |
| Fill Target From Current Pose | 把当前末端位姿填入目标输入框 |
| Move To Target Pose | 对目标末端位姿求 IK 并发布 q_goal |
| Publish Last q_goal Again | 再次发布上一次 q_goal |
| Zero Effort | 发布全 0 effort |
| Control Mode | 切换 Pose IK Jog / Twist Jog |
| Hold target by republishing q_goal | 周期性重复发布 q_goal |
| Pose step / m | Pose IK Jog 每次位置增量 |
| Pose rot step / rad | Pose IK Jog 每次角度增量 |
| Twist linear / m/s | Twist Jog 线速度 |
| Twist angular / rad/s | Twist Jog 角速度 |
| Jog rate / Hz | 按住 Jog 按钮时的控制频率 |

---

## 10. 重要注意事项

### 10.1 关节名必须一致

如果 `/joint_states` 中的关节名是：

```text
fr3_joint1 ... fr3_joint7
```

则配置文件也必须使用：

```text
fr3_joint1 ... fr3_joint7
```

如果 Isaac Sim 加载的是 Panda，并发布：

```text
panda_joint1 ... panda_joint7
```

则配置文件也必须保持 Panda 命名。

---

### 10.2 当前不是标准 ros2_control 架构

当前 `/joint_command` 使用的是 `sensor_msgs/JointState`。这适合 Isaac Sim ROS2 Bridge 的 `ROS2SubscribeJointState` 节点，但不是标准 ros2_control 的常见控制接口。

标准 ros2_control 常见接口包括：

```text
trajectory_msgs/JointTrajectory
control_msgs/action/FollowJointTrajectory
std_msgs/Float64MultiArray
```

如果后续接真实机械臂或标准 ros2_control，需要改造 ROS 通信层。

---

### 10.3 Zero Effort 不等于保持位置

`Zero Effort` 会发布：

```text
effort = [0, 0, 0, 0, 0, 0, 0]
```

如果底层是 effort 模式，机械臂可能受重力下坠。如果底层是 position drive，这个命令未必能覆盖 position command 的效果。

---

### 10.4 Twist Jog 的 achieved_twist 是理论值

当前 UI 中显示的 `achieved_twist` 来自：

```python
achieved_twist = J @ dq_cmd
```

这是理论计算值，不是从实际关节速度反馈出来的真实末端速度。

如需显示真实末端速度，应使用：

```python
actual_twist = J(q_current) @ dq_current
```

---

## 11. 后续改进方向

### 11.1 增加真正的 velocity command

新增：

```python
publish_velocity(dq_cmd)
```

然后 Twist Jog 直接发布 `dq_cmd`，不再积分成 `q_target`。

---

### 11.2 接入 ros2_control velocity controller

可考虑使用：

```text
forward_velocity_controller
joint_group_velocity_controller
```

控制链路变成：

```text
Twist → dq_cmd → velocity controller → robot
```

---

### 11.3 接入 MoveIt Servo

如果目标是成熟稳定的 Jog 控制，可以考虑使用 MoveIt Servo，由其处理：

- 奇异点；
- 速度缩放；
- 碰撞检测；
- 关节限位；
- 低通滤波；
- Twist 到关节命令转换。

---

### 11.4 改进 IK 姿态误差

当前 IK 的姿态误差使用 RPY 差值。后续可改为：

```text
SO(3) log map
rotation vector error
quaternion error
```

这样在大角度姿态控制时更稳定。

---

### 11.5 增加真实速度反馈显示

建议 UI 中同时显示：

```text
desired_twist
solved_twist = J @ dq_cmd
actual_twist = J @ dq_current
```

这样可以判断仿真底层是否真正跟随速度命令。

---

## 12. 当前版本控制能力总结

| 能力 | 当前状态 |
|---|---|
| 读取 `/joint_states` | 已实现 |
| 发布 `/joint_command` | 已实现 |
| 关节位置命令 | 已实现 |
| 关节 effort 命令 | 已预留 |
| 末端正运动学 | 已实现 |
| 末端位姿 IK | 已实现 |
| 数值 Jacobian | 已实现 |
| 几何 Jacobian | 已实现 |
| 末端 Twist Jog | 已实现 |
| 真正 velocity command | 未作为主流程使用 |
| 真正 velocity controller | 未接入 |
| MoveIt Servo | 未接入 |
| 真实机械臂安全保护 | 未完善 |

---

## 13. 项目一句话说明

中文：

> 一个面向 Franka / FR3 / Panda 仿真的模块化 PyQt6 + ROS 2 末端点动控制系统，支持末端位姿 IK 控制和基于 Jacobian 的 Twist Jog 控制；当前版本通过关节位置命令执行运动，后续可扩展为真正的关节速度控制、力矩控制和 MoveIt Servo 控制链路。

English:

> A modular PyQt6 + ROS 2 Cartesian jog controller for Franka / FR3 / Panda simulation. It supports pose IK jog and Jacobian-based twist jog. The current implementation converts Cartesian twist commands into integrated joint position targets for Isaac Sim position-command execution, and can be extended to real joint velocity, torque, or MoveIt Servo control pipelines.

---

## 14. 推荐 GitHub 描述

```text
A modular PyQt6 + ROS 2 controller for Franka / FR3 / Panda simulation, supporting Cartesian pose IK jog and Jacobian-based twist jog with Isaac Sim ROS2 Bridge integration.
```

推荐标签：

```text
franka, fr3, panda, ros2, pyqt6, isaac-sim, robot-control, inverse-kinematics, jacobian, cartesian-control
```