from typing import Optional
import numpy as np

from controllers.base import BaseController, ControlDebugData


class CartesianAdmittanceControllerPlaceholder(BaseController):
    """
    预留：导纳控制接口。

    典型需要补充：
    1. 外部力/力矩估计：tau_ext 或 F/T sensor；
    2. M-D-K 导纳模型：M*x_ddot + D*x_dot + K*x = F_ext；
    3. 目标位姿积分；
    4. IK / 轨迹控制器 / 速度控制器输出。
    """

    def __init__(self):
        self.debug = ControlDebugData(mode="cartesian_admittance_placeholder")

    def capture_target(self, q: np.ndarray) -> None:
        self.debug.message = "Admittance target capture placeholder."

    def compute(self, q: np.ndarray, dq: np.ndarray) -> Optional[np.ndarray]:
        self.debug.message = "Admittance controller is only a placeholder now."
        return None

    def get_debug_data(self) -> ControlDebugData:
        return self.debug
