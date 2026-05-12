#!/usr/bin/env python3
"""
main.py

程序入口文件。

当前版本功能：
1. 加载系统配置；
2. 初始化 ROS 2；
3. 创建机器人状态缓存；
4. 创建 ROS 通信节点；
5. 创建 Panda/Franka 运动学模型；
6. 创建末端 6D 位姿 IK 控制器；
7. 创建 PyQt6 图形界面；
8. UI 输入 x/y/z/rx/ry/rz 后，通过 IK 求解 q_goal；
9. 通过 ros_node.publish_position(q_goal) 发布关节位置命令；
10. 程序退出时安全释放 ROS 资源。
"""

import sys
from pathlib import Path
import rclpy
from PyQt6.QtWidgets import QApplication

# =========================
# 配置模块
# =========================
from config.controller_config import AppConfig

# =========================
# 核心模块
# =========================
from core.kinematics import PandaKinematics
from core.robot_state import RobotStateBuffer

# =========================
# ROS 通信模块
# =========================
from ros_interface.franka_ros_node import FrankaRosNode

# =========================
# IK 控制器模块
# =========================
from controllers.cartesian_pose_ik import CartesianPoseIKController

# =========================
# UI 模块
# =========================
from ui.main_window import CartesianNullspaceWindow


def main() -> None:
    """
    主函数。

    当前启动流程：

        AppConfig
            ↓
        rclpy.init()
            ↓
        RobotStateBuffer
            ↓
        FrankaRosNode
            ↓
        PandaKinematics
            ↓
        CartesianPoseIKController
            ↓
        CartesianNullspaceWindow
            ↓
        app.exec()
    """

    # ------------------------------------------------------------
    # 1. 加载配置
    # ------------------------------------------------------------
    # cfg.robot 里一般包含：
    # - joint_names
    # - joint_state_topic
    # - joint_command_topic
    # - node_name
    cfg = AppConfig()

    # ------------------------------------------------------------
    # 2. 初始化 ROS 2
    # ------------------------------------------------------------
    rclpy.init()

    # ------------------------------------------------------------
    # 3. 创建机器人状态缓存
    # ------------------------------------------------------------
    # 用于缓存 /joint_states 中的当前关节角 q 和速度 dq。
    state_buffer = RobotStateBuffer(cfg.robot.joint_names)

    # ------------------------------------------------------------
    # 4. 创建 ROS 通信节点
    # ------------------------------------------------------------
    # FrankaRosNode 负责：
    # - 订阅 joint_state_topic；
    # - 更新 state_buffer；
    # - 发布 position / effort 命令。
    ros_node = FrankaRosNode(cfg.robot, state_buffer)

    # ------------------------------------------------------------
    # 5. 创建运动学模型
    # ------------------------------------------------------------
    # kin 负责：
    # - fk(q)
    # - fk_transform(q)
    # - fk_pose6(q)
    # - numerical_pose_jacobian(q)
    PROJECT_ROOT = Path(__file__).resolve().parent

    kin = PandaKinematics(
        urdf_path=str(PROJECT_ROOT / "config" / "fr3.urdf"),
        robot_description_yaml_path=str(PROJECT_ROOT / "config" / "fr3_robot_description.yaml"),
        base_link="fr3_link0",
        end_link="fr3_hand_tcp",
    )

    # ------------------------------------------------------------
    # 6. 创建末端 6D 位姿 IK 控制器
    # ------------------------------------------------------------
    # 这个控制器负责：
    # - 输入目标 pose6 = [x, y, z, rx, ry, rz]
    # - 从当前 q_current 开始迭代求解 IK
    # - 输出 q_goal
    #
    # 注意：
    # 它不输出 effort。
    # 它的结果 q_goal 会通过 ros_node.publish_position(q_goal) 发布。
    pose_ik_controller = CartesianPoseIKController(
        kin=kin,
        max_iterations=100,
        tolerance=1e-4,
        damping=0.05,
        max_delta_q=0.08,
    )

    # ------------------------------------------------------------
    # 7. 创建 Qt 应用
    # ------------------------------------------------------------
    app = QApplication(sys.argv)

    # ------------------------------------------------------------
    # 8. 创建主窗口
    # ------------------------------------------------------------
    # 这里要和新版 ui/main_window.py 的 __init__ 参数保持一致：
    #
    # def __init__(
    #     self,
    #     cfg,
    #     ros_node,
    #     state_buffer,
    #     pose_ik_controller,
    # )
    window = CartesianNullspaceWindow(
        cfg=cfg,
        ros_node=ros_node,
        state_buffer=state_buffer,
        pose_ik_controller=pose_ik_controller,
    )

    window.show()

    try:
        # --------------------------------------------------------
        # 9. 进入 Qt 主事件循环
        # --------------------------------------------------------
        # ROS spin 会在 UI 里面通过 QTimer 调用 spin_once_safe()。
        exit_code = app.exec()

    finally:
        # --------------------------------------------------------
        # 10. 程序退出时释放资源
        # --------------------------------------------------------
        try:
            # 如果底层当前是 effort 模式，这个可以清零力矩。
            # 如果底层是 position drive，这个不一定有实际作用，但保留也没问题。
            ros_node.publish_zero_effort()

        finally:
            ros_node.destroy_node()
            rclpy.shutdown()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()