from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CartesianTwistResult:
    success: bool
    q_target: Optional[np.ndarray]
    dq_cmd: Optional[np.ndarray]
    achieved_twist: Optional[np.ndarray]
    message: str


class CartesianTwistController:
    """
    末端笛卡尔速度 / Twist 控制器。

    控制关系：

        twist = J(q) @ dq

    其中：

        twist = [vx, vy, vz, wx, wy, wz]

    使用阻尼最小二乘伪逆：

        dq = J.T @ inv(J @ J.T + lambda^2 I) @ twist

    推荐使用方式：

        1. Jacobian 使用当前真实关节角 q_current 计算；
        2. q_target 使用内部积分目标 q_integrator 积分；
        3. 不要每次都 q_current + dq * dt。

    正确方式：

        q_target = q_integrator + dq * dt

    这样可以避免 Isaac Sim position drive 跟踪滞后、重力下坠等因素
    导致每次目标都被实际关节状态重新拉回。
    """

    def __init__(
        self,
        kin,
        dt: float = 0.02,
        damping: float = 0.08,
        max_joint_speed: float = 0.35,
        joint_lower: Optional[np.ndarray] = None,
        joint_upper: Optional[np.ndarray] = None,
        allow_numerical_jacobian_fallback: bool = False,
        debug: bool = False,
    ):
        self.kin = kin
        self.dt = float(dt)
        self.damping = float(damping)
        self.max_joint_speed = float(max_joint_speed)
        self.allow_numerical_jacobian_fallback = bool(allow_numerical_jacobian_fallback)
        self.debug = bool(debug)

        self.joint_lower = None
        self.joint_upper = None

        if joint_lower is not None:
            self.joint_lower = np.asarray(joint_lower, dtype=float).reshape(7)

        if joint_upper is not None:
            self.joint_upper = np.asarray(joint_upper, dtype=float).reshape(7)

    def solve(
        self,
        q_current: np.ndarray,
        twist: np.ndarray,
        q_integrator: Optional[np.ndarray] = None,
    ) -> CartesianTwistResult:
        """
        根据当前 q 和期望末端 twist 求 dq，并积分得到 q_target。

        参数：
            q_current:
                当前真实关节角，用来计算 Jacobian。

            twist:
                期望末端速度：
                    [vx, vy, vz, wx, wy, wz]

            q_integrator:
                UI 或控制器内部维护的积分目标。
                如果传入，则：
                    q_target = q_integrator + dq * dt

                如果不传，则退化为：
                    q_target = q_current + dq * dt

                但是 Twist Jog 强烈建议传入 q_integrator。
        """

        q_current = np.asarray(q_current, dtype=float).reshape(7)
        twist = np.asarray(twist, dtype=float).reshape(6)

        if q_integrator is None:
            q_base = q_current.copy()
        else:
            q_base = np.asarray(q_integrator, dtype=float).reshape(7).copy()

        if np.linalg.norm(twist) < 1e-12:
            return CartesianTwistResult(
                success=True,
                q_target=q_base.copy(),
                dq_cmd=np.zeros(7, dtype=float),
                achieved_twist=np.zeros(6, dtype=float),
                message="Zero twist.",
            )

        try:
            J = self._get_jacobian(q_current)

            if J.shape != (6, 7):
                return CartesianTwistResult(
                    success=False,
                    q_target=None,
                    dq_cmd=None,
                    achieved_twist=None,
                    message=f"Invalid Jacobian shape: {J.shape}, expected (6, 7).",
                )

            dq_cmd = self._solve_damped_least_squares(J, twist)
            dq_cmd = self._limit_joint_speed(dq_cmd)

            q_target = q_base + dq_cmd * self.dt
            q_target = self._clamp_joint_limits(q_target)

            achieved_twist = J @ dq_cmd

            if self.debug:
                self._print_debug(
                    q_current=q_current,
                    q_base=q_base,
                    twist=twist,
                    dq_cmd=dq_cmd,
                    achieved_twist=achieved_twist,
                    q_target=q_target,
                    J=J,
                )

            return CartesianTwistResult(
                success=True,
                q_target=q_target,
                dq_cmd=dq_cmd,
                achieved_twist=achieved_twist,
                message="Twist solved.",
            )

        except Exception as e:
            return CartesianTwistResult(
                success=False,
                q_target=None,
                dq_cmd=None,
                achieved_twist=None,
                message=f"Twist solve failed: {e}",
            )

    def _get_jacobian(self, q_current: np.ndarray) -> np.ndarray:
        """
        获取 6x7 Jacobian。

        强烈建议使用 geometric_jacobian。
        numerical_pose_jacobian 对 IK 勉强可用，
        但不适合直接做 Twist 速度控制。
        """

        if hasattr(self.kin, "geometric_jacobian"):
            J = self.kin.geometric_jacobian(q_current)
            return np.asarray(J, dtype=float)

        if self.allow_numerical_jacobian_fallback and hasattr(
            self.kin,
            "numerical_pose_jacobian",
        ):
            J = self.kin.numerical_pose_jacobian(q_current)
            return np.asarray(J, dtype=float)

        raise RuntimeError(
            "kin does not provide geometric_jacobian(q). "
            "Please add geometric_jacobian() in kinematics.py. "
            "Do not use numerical_pose_jacobian() for Twist Jog unless debugging."
        )

    def _solve_damped_least_squares(
        self,
        J: np.ndarray,
        twist: np.ndarray,
    ) -> np.ndarray:
        """
        阻尼最小二乘：

            dq = J.T @ inv(J @ J.T + lambda^2 I) @ twist

        这里用 np.linalg.solve，避免直接 inv。
        """

        J = np.asarray(J, dtype=float).reshape(6, 7)
        twist = np.asarray(twist, dtype=float).reshape(6)

        lambda2 = self.damping ** 2
        A = J @ J.T + lambda2 * np.eye(6, dtype=float)

        y = np.linalg.solve(A, twist)
        dq = J.T @ y

        return dq

    def _limit_joint_speed(self, dq: np.ndarray) -> np.ndarray:
        """
        限制关节速度，避免动作太猛。

        max_joint_speed 单位：
            rad/s
        """

        dq = np.asarray(dq, dtype=float).reshape(7)

        max_abs = float(np.max(np.abs(dq)))

        if max_abs < 1e-12:
            return dq

        if max_abs <= self.max_joint_speed:
            return dq

        scale = self.max_joint_speed / max_abs
        return dq * scale

    def _clamp_joint_limits(self, q: np.ndarray) -> np.ndarray:
        """
        限制关节角。
        """

        q = np.asarray(q, dtype=float).reshape(7)

        if self.joint_lower is not None and self.joint_upper is not None:
            return np.clip(q, self.joint_lower, self.joint_upper)

        if hasattr(self.kin, "clamp_to_joint_limits"):
            return self.kin.clamp_to_joint_limits(q)

        return q

    def _print_debug(
        self,
        q_current: np.ndarray,
        q_base: np.ndarray,
        twist: np.ndarray,
        dq_cmd: np.ndarray,
        achieved_twist: np.ndarray,
        q_target: np.ndarray,
        J: np.ndarray,
    ) -> None:
        print("\n========== CartesianTwistController Debug ==========")
        print("q_current      =", np.array2string(q_current, precision=4))
        print("q_base         =", np.array2string(q_base, precision=4))
        print("desired twist  =", np.array2string(twist, precision=4))
        print("dq_cmd         =", np.array2string(dq_cmd, precision=4))
        print("achieved twist =", np.array2string(achieved_twist, precision=4))
        print("q_target       =", np.array2string(q_target, precision=4))
        print("J linear rows  =\n", np.array2string(J[0:3, :], precision=4))
        print("J angular rows =\n", np.array2string(J[3:6, :], precision=4))
        print("====================================================\n")