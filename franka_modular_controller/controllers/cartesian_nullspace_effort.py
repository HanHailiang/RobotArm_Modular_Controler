from typing import Optional
import numpy as np

from config.controller_config import CartesianImpedanceConfig, NullspaceConfig, SafetyConfig
from core.kinematics import PandaKinematics
from core.math_utils import clamp_vector, nullspace_projector
from controllers.base import BaseController, ControlDebugData


class CartesianNullspaceEffortController(BaseController):
    """
    当前主控制器：末端位置保持 + 零空间姿态恢复 + joint6 扰动力矩。

    输出：7 维 effort/torque。
    """

    def __init__(self, kin: PandaKinematics, cart_cfg: CartesianImpedanceConfig,
                 null_cfg: NullspaceConfig, safety_cfg: SafetyConfig):
        self.kin = kin
        self.cart_cfg = cart_cfg
        self.null_cfg = null_cfg
        self.safety_cfg = safety_cfg
        self.q_home: Optional[np.ndarray] = None
        self.p_des: Optional[np.ndarray] = None
        self.debug = ControlDebugData(mode="cartesian_nullspace_effort")

    def capture_target(self, q: np.ndarray) -> None:
        self.q_home = q.copy()
        self.p_des = self.kin.fk(q)
        self.debug.p_des = self.p_des.copy()
        self.debug.message = "Captured current q as home and current end-effector position as target."

    def compute(self, q: np.ndarray, dq: np.ndarray) -> Optional[np.ndarray]:
        if self.q_home is None or self.p_des is None:
            self.debug.message = "Please capture home / end-effector target first."
            return None

        p = self.kin.fk(q)
        J = self.kin.numerical_jacobian(q)
        p_dot = J @ dq
        p_err = self.p_des - p

        F_pos = self.cart_cfg.kx * p_err - self.cart_cfg.dx * p_dot
        F_pos = np.clip(F_pos, -abs(self.safety_cfg.max_force), abs(self.safety_cfg.max_force))
        tau_cart = J.T @ F_pos

        N = nullspace_projector(J, self.cart_cfg.damping_lambda)
        tau_null_raw = self.null_cfg.kq * (self.q_home - q) - self.null_cfg.dq * dq

        tau_dist_raw = np.zeros(7, dtype=float)
        tau_dist_raw[5] = self.null_cfg.joint6_disturbance_tau

        if self.null_cfg.project_disturbance_to_nullspace:
            tau_extra = N @ (tau_null_raw + tau_dist_raw)
            tau_null_projected = N @ tau_null_raw
            tau_dist_projected = N @ tau_dist_raw
        else:
            tau_extra = N @ tau_null_raw + tau_dist_raw
            tau_null_projected = N @ tau_null_raw
            tau_dist_projected = tau_dist_raw

        tau_total = clamp_vector(tau_cart + tau_extra, self.safety_cfg.max_tau)

        self.debug = ControlDebugData(
            tau_total=tau_total.copy(),
            tau_cart=tau_cart.copy(),
            tau_null=tau_null_projected.copy(),
            tau_dist=tau_dist_projected.copy(),
            p=p.copy(),
            p_des=self.p_des.copy(),
            p_err=p_err.copy(),
            mode="cartesian_nullspace_effort",
            message="Publishing Cartesian hold + nullspace effort command.",
        )
        return tau_total

    def get_debug_data(self) -> ControlDebugData:
        return self.debug
