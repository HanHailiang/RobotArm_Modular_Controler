from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from config.controller_config import RobotConfig
from core.robot_state import RobotStateBuffer
from typing import List, Optional

class FrankaRosNode(Node):
    """
    Franka ROS2 通信节点。

    职责：
    1. 订阅机器人当前关节状态；
    2. 将关节状态写入 RobotStateBuffer；
    3. 发布 effort 控制命令到指定控制话题；
    4. 提供发布零力矩的安全接口。

    注意：
    这个类只负责 ROS 通信，不负责控制算法。
    控制算法应放在 controllers/ 目录下。
    UI 显示逻辑应放在 ui/ 目录下。
    """

    def __init__(self, robot_cfg: RobotConfig, state_buffer: RobotStateBuffer):
        """
        初始化 ROS2 节点。

        参数：
        robot_cfg:
            机器人相关配置，例如：
            - 节点名
            - 关节名列表
            - 关节状态订阅话题
            - 控制命令发布话题

        state_buffer:
            机器人状态缓存对象。
            ROS 回调收到 /joint_states 后，会把数据写入该缓存。
            控制器和 UI 再从这个缓存中读取最新状态。
        """

        # 使用配置文件中的 node_name 创建 ROS2 节点
        super().__init__(robot_cfg.node_name)

        # 保存机器人配置，后续发布命令时需要关节名和话题名
        self.robot_cfg = robot_cfg

        # 保存状态缓存对象
        # 该对象用于在 ROS 回调、控制器、UI 之间共享机器人状态
        self.state_buffer = state_buffer

        # ------------------------------------------------------------
        # 创建 effort 命令发布器
        # ------------------------------------------------------------
        # 发布类型使用 sensor_msgs/JointState。
        #
        # 当前你的 Isaac Sim 控制接口是通过 /joint_command 接收 effort。
        # 因此这里发布 JointState，并只填充 effort 字段。
        #
        # 如果后续切换到 ros2_control 的标准控制器，
        # 这里可能需要改成：
        # - trajectory_msgs/JointTrajectory
        # - control_msgs/action/FollowJointTrajectory
        # - std_msgs/Float64MultiArray
        # - 自定义 effort command 消息
        self.publisher = self.create_publisher(
            JointState,
            robot_cfg.joint_command_topic,
            10,
        )

        # ------------------------------------------------------------
        # 创建关节状态订阅器
        # ------------------------------------------------------------
        # 订阅 /joint_states 或配置文件中指定的话题。
        #
        # Isaac Sim、ros2_control、joint_state_broadcaster
        # 通常都会发布 sensor_msgs/JointState。
        self.subscription = self.create_subscription(
            JointState,
            robot_cfg.joint_state_topic,
            self.joint_state_callback,
            10,
        )

        # 打印启动信息，方便确认订阅和发布的话题是否正确
        self.get_logger().info(
            f"Started. sub={robot_cfg.joint_state_topic}, "
            f"pub={robot_cfg.joint_command_topic}"
        )

    def publish_position(self, position: List[float]) -> None:
        """
        发布关节位置命令。

        适用于 Isaac Sim 中 position drive / position command 模式。
        """
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = self.robot_cfg.joint_names
        cmd.position = list(position)
        cmd.velocity = []
        cmd.effort = []
        self.publisher.publish(cmd)


    def publish_position_with_velocity(
        self,
        position: List[float],
        velocity: Optional[List[float]] = None,
    ) -> None:
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = self.robot_cfg.joint_names
        cmd.position = list(position)
        cmd.velocity = list(velocity) if velocity is not None else []
        cmd.effort = []
        self.publisher.publish(cmd)


    def publish_zero_velocity(self) -> None:
        """
        发布零速度命令。

        速度控制模式下，松开 Jog 按钮时必须调用这个函数，
        否则底层可能继续保持上一帧速度命令。
        """
        self.publish_velocity([0.0] * len(self.robot_cfg.joint_names))
    # ============================================================
    # Velocity Command
    # ============================================================

    def publish_velocity(self, velocity: List[float]) -> None:
        """
        发布关节速度命令。

        这是这次真正速度控制的关键接口。

        发布内容：

            name     = 关节名
            position = []
            velocity = dq_cmd
            effort   = []

        对应 Isaac Sim ROS2 Graph：

            ROS2SubscribeJointState.outputs:velocityCommand
                    ↓
            IsaacArticulationController.inputs:velocityCommand
        """

        if len(velocity) != len(self.robot_cfg.joint_names):
            self.get_logger().error(
                f"Velocity command length mismatch: "
                f"got {len(velocity)}, expected {len(self.robot_cfg.joint_names)}"
            )
            return

        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = self.robot_cfg.joint_names

        cmd.position = []
        cmd.velocity = [float(v) for v in velocity]
        cmd.effort = []

        self.publisher.publish(cmd)
    def joint_state_callback(self, msg: JointState) -> None:
        """
        /joint_states 回调函数。

        每次收到新的 JointState 消息后，将其写入状态缓存。

        JointState 通常包含：
        - msg.name：关节名称列表
        - msg.position：关节角度
        - msg.velocity：关节速度
        - msg.effort：关节力矩或力

        这里不直接解析 q、dq，而是交给 RobotStateBuffer 统一处理。
        这样可以避免 ROS 通信层和状态解析逻辑耦合太深。
        """

        self.state_buffer.update_from_msg(msg)

    def publish_effort(self, effort: List[float]) -> None:
        """
        发布关节 effort 控制命令。

        参数：
        effort:
            长度应与 robot_cfg.joint_names 一致。
            对 Franka/Panda 来说通常是 7 个关节力矩。

        当前发布的 JointState 中：
        - name 填写关节名；
        - effort 填写控制器计算出的力矩；
        - position 和 velocity 留空。

        这适用于 Isaac Sim 中某些通过 JointState 接收 effort 的控制方式。
        如果后续使用标准 ros2_control，发布格式可能需要调整。
        """

        # 创建 JointState 命令消息
        cmd = JointState()

        # 写入当前 ROS 时间戳
        # 有些控制接口会检查时间戳，保留这个字段更稳妥
        cmd.header.stamp = self.get_clock().now().to_msg()

        # 写入关节名，顺序必须和 effort 数组严格对应
        cmd.name = self.robot_cfg.joint_names

        # 当前是 effort 控制，所以 position 和 velocity 置空
        cmd.position = []
        cmd.velocity = []

        # 写入力矩命令
        cmd.effort = list(effort)

        # 发布到 robot_cfg.joint_command_topic
        self.publisher.publish(cmd)

    def publish_zero_effort(self) -> None:
        """
        发布零力矩命令。

        用途：
        1. 控制器关闭时停止输出；
        2. 程序退出前安全释放；
        3. UI 中点击 Zero effort；
        4. 控制异常时避免保留上一帧力矩命令。

        注意：
        如果底层控制器是 effort 模式，发送零力矩后，
        机械臂可能会受重力影响下坠。
        如果希望机械臂保持姿态，需要底层有重力补偿或位置保持控制器。
        """

        self.publish_effort([0.0] * len(self.robot_cfg.joint_names))


def spin_once_safe(node: FrankaRosNode) -> str:
    """
    安全执行一次 ROS spin。

    在 PyQt6 程序中，通常不直接调用 rclpy.spin(node)，
    因为 rclpy.spin(node) 会阻塞当前线程，导致 Qt 界面卡住。

    更常见的做法是：
    1. 在 QTimer 中周期调用 spin_once_safe；
    2. 每隔几毫秒处理一次 ROS 回调；
    3. Qt 主事件循环继续负责 UI 响应。

    返回值：
    - 如果正常，返回空字符串；
    - 如果异常，返回错误信息，方便 UI 显示到状态栏。

    示例：
        err = spin_once_safe(ros_node)
        if err:
            status_label.setText(err)
    """

    try:
        # timeout_sec=0.0 表示非阻塞执行
        # 有 ROS 消息就处理，没有消息就立即返回
        rclpy.spin_once(node, timeout_sec=0.0)
        return ""

    except Exception as exc:
        # 不在这里直接抛异常，避免 UI 主循环因为 ROS 异常退出
        # 返回字符串，让 UI 层决定如何显示或处理
        return f"ROS spin error: {exc}"