from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class RobotConfig:
    """机器人与 ROS 话题配置。"""
    node_name: str = "franka_modular_pyqt6_controller"
    joint_state_topic: str = "/joint_states"
    joint_command_topic: str = "/joint_command"
    joint_names: List[str] = field(default_factory=lambda: [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ])


@dataclass
class SafetyConfig:
    """安全限幅。实际机械臂上还需要速度、加速度、jerk、碰撞检测等更完整保护。"""
    max_tau: float = 40.0
    max_position_step: float = 0.02
    max_force: float = 80.0
    publish_zero_when_disabled: bool = True


@dataclass
class CartesianImpedanceConfig:
    """末端笛卡尔阻抗参数。"""
    kx: float = 200.0
    dx: float = 30.0
    damping_lambda: float = 0.03


@dataclass
class NullspaceConfig:
    """零空间姿态恢复和扰动参数。"""
    kq: float = 5.0
    dq: float = 1.0
    joint6_disturbance_tau: float = 0.0
    project_disturbance_to_nullspace: bool = True


@dataclass
class UiConfig:
    control_rate_hz: int = 50
    ros_spin_interval_ms: int = 5
    ui_refresh_interval_ms: int = 100


@dataclass
class AppConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    cartesian: CartesianImpedanceConfig = field(default_factory=CartesianImpedanceConfig)
    nullspace: NullspaceConfig = field(default_factory=NullspaceConfig)
    ui: UiConfig = field(default_factory=UiConfig)


def np7_zero() -> np.ndarray:
    return np.zeros(7, dtype=float)


def np3_zero() -> np.ndarray:
    return np.zeros(3, dtype=float)
