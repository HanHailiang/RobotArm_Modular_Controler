# Franka Modular PyQt6 Controller

这是从原始单文件代码拆分出来的多文件版本，目标是让后续扩展位置控制、力控、导纳控制、阻抗控制更方便。

## 目录结构

```text
franka_modular_controller/
├── main.py
├── config/
│   └── controller_config.py
├── core/
│   ├── kinematics.py
│   ├── math_utils.py
│   └── robot_state.py
├── controllers/
│   ├── base.py
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

## 运行方式

进入 `franka_modular_controller` 目录后运行：

```bash
python3 main.py
```

前提：当前 ROS 2 环境已经 source，并且 Python 环境中有 `rclpy`、`sensor_msgs`、`numpy`、`PyQt6`。

## 当前保留的功能

1. 订阅 `/joint_states`。
2. 发布 `/joint_command` 的 `JointState.effort`。
3. 末端位置保持。
4. 零空间姿态恢复。
5. joint6 扰动力矩。
6. 可选择是否把扰动力矩投影到零空间。
7. PyQt6 参数调节与关节监控。

## 后续扩展建议

### 1. 更准确的运动学
当前 `core/kinematics.py` 仍然是近似 DH + 数值雅可比。工程化建议替换为：

- Pinocchio；
- KDL；
- Isaac Sim Articulation Jacobian；
- MoveIt 的 RobotState Jacobian。

### 2. 位置控制
现在的 `controllers/joint_position.py` 是用 effort 模拟关节位置 PD。若使用 ROS 2 标准控制器，建议新增：

- `ros_interface/trajectory_command_publisher.py`
- 发布 `trajectory_msgs/JointTrajectory`
- 对接 `/fr3_arm_controller/follow_joint_trajectory`

### 3. 力控 / 阻抗控制
当前主控制器本质是笛卡尔位置阻抗：

```text
F = Kx * (p_des - p) - Dx * p_dot
τ = JᵀF + Nτ_null
```

后续可以补充：

- 姿态控制：roll/pitch/yaw 或 quaternion error；
- 6D wrench 控制：Fx,Fy,Fz,Mx,My,Mz；
- 重力补偿项；
- 摩擦补偿项；
- 外部力估计；
- 接触状态检测；
- 力限幅、速度限幅、能量罐 passivity 安全机制。

### 4. 导纳控制
已预留 `controllers/admittance_placeholder.py`。典型结构：

```text
M x_ddot + D x_dot + K x = F_ext
```

导纳控制通常输出目标位置/速度，再交给位置控制器或轨迹控制器执行。

## 注意

如果机械臂在 Isaac Sim 中仍然趴下，说明底层 articulation drive / effort mode / 重力补偿之间还没有配合好。这个工程把控制程序模块化，不会自动修复物理仿真参数不匹配问题。
