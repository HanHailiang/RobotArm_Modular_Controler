from typing import Optional
import numpy as np

from config.controller_config import SafetyConfig
from core.math_utils import clamp_vector
from controllers.base import BaseController, ControlDebugData


class JointPositionAsEffortController(BaseController):
    """
    预留：关节位置保持控制。

    注意：这里仍然用 effort 形式模拟 PD 位置控制。
    如果你后续接 ros2_control 的 JointTrajectoryController，应改成发布 trajectory_msgs/JointTrajectory。
    """

    def __init__(self, safety_cfg: SafetyConfig, kp: float = 20.0, kd: float = 2.0):
        self.safety_cfg = safety_cfg
        self.kp = kp
        self.kd = kd
        self.q_des: Optional[np.ndarray] = None
        self.debug = ControlDebugData(mode="joint_position_effort")

    def capture_target(self, q: np.ndarray) -> None:
        self.q_des = q.copy()
        self.debug.message = "Captured current joint positions as q_des."

    def compute(self, q: np.ndarray, dq: np.ndarray) -> Optional[np.ndarray]:
        if self.q_des is None:
            self.debug.message = "Please capture q_des first."
            return None
        tau = self.kp * (self.q_des - q) - self.kd * dq
        tau = clamp_vector(tau, self.safety_cfg.max_tau)
        self.debug.tau_total = tau.copy()
        self.debug.mode = "joint_position_effort"
        self.debug.message = "Publishing joint position PD effort command."
        return tau

    def get_debug_data(self) -> ControlDebugData:
        return self.debug
