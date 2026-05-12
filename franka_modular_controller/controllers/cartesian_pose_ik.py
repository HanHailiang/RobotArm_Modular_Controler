from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from core.kinematics import PandaKinematics


@dataclass
class IKResult:
    success: bool
    q_goal: Optional[np.ndarray]
    message: str
    iterations: int
    final_error_norm: float


class CartesianPoseIKController:
    """
    末端 6D 位姿 IK 控制器。

    作用：
    1. 输入目标末端位姿 [x, y, z, rx, ry, rz]
    2. 从当前关节角 q_current 开始迭代
    3. 求解目标关节角 q_goal
    4. q_goal 再交给 position / trajectory 控制器执行

    注意：
    这个类本身不发布 ROS 命令，也不输出 effort。
    它只负责 IK 求解。
    """

    def __init__(
        self,
        kin: PandaKinematics,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        damping: float = 0.05,
        max_delta_q: float = 0.08,
    ):
        self.kin = kin
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping
        self.max_delta_q = max_delta_q

        self.target_pose: Optional[np.ndarray] = None
        self.last_result: Optional[IKResult] = None

    def set_target_pose(self, pose6: np.ndarray) -> None:
        """
        设置目标末端位姿。

        pose6:
            [x, y, z, rx, ry, rz]

        其中：
            x/y/z 单位 m
            rx/ry/rz 单位 rad
        """
        self.target_pose = np.asarray(pose6, dtype=float).copy()

    def solve(self, q_start: np.ndarray) -> IKResult:
        """
        从当前关节角 q_start 开始求解 IK。

        返回：
            IKResult，其中包含 q_goal。
        """

        if self.target_pose is None:
            result = IKResult(
                success=False,
                q_goal=None,
                message="Target pose is not set.",
                iterations=0,
                final_error_norm=float("inf"),
            )
            self.last_result = result
            return result

        q = np.asarray(q_start, dtype=float).copy()

        for it in range(self.max_iterations):
            current_pose = self.kin.fk_pose6(q)
            err = self.pose_error(self.target_pose, current_pose)

            err_norm = float(np.linalg.norm(err))
            if err_norm < self.tolerance:
                result = IKResult(
                    success=True,
                    q_goal=q.copy(),
                    message="IK solved.",
                    iterations=it,
                    final_error_norm=err_norm,
                )
                self.last_result = result
                return result

            J = self.kin.numerical_pose_jacobian(q)

            # 阻尼最小二乘 IK：
            # dq = J.T @ inv(J @ J.T + lambda^2 I) @ err
            lam2 = self.damping ** 2
            dq = J.T @ np.linalg.solve(
                J @ J.T + lam2 * np.eye(6),
                err,
            )

            # 单步关节增量限幅，避免跳动过大
            dq = np.clip(dq, -self.max_delta_q, self.max_delta_q)

            q = q + dq

            # 可选：这里后续可以加入关节限位
            # q = np.clip(q, q_min, q_max)

        final_pose = self.kin.fk_pose6(q)
        final_err = self.pose_error(self.target_pose, final_pose)
        final_error_norm = float(np.linalg.norm(final_err))

        result = IKResult(
            success=False,
            q_goal=q.copy(),
            message="IK reached max iterations. Result may be approximate.",
            iterations=self.max_iterations,
            final_error_norm=final_error_norm,
        )
        self.last_result = result
        return result

    @staticmethod
    def pose_error(target_pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
        """
        计算 6D 位姿误差。

        这里假设 pose6 是：
            [x, y, z, rx, ry, rz]

        简化处理：
            直接用 RPY 差值作为姿态误差。

        注意：
            工程上更推荐用 rotation vector / SO(3) log map。
            但对于小角度相对移动和调试，这个版本更容易理解。
        """

        err = np.asarray(target_pose, dtype=float) - np.asarray(current_pose, dtype=float)

        # 将角度误差归一化到 [-pi, pi]
        err[3:6] = (err[3:6] + np.pi) % (2.0 * np.pi) - np.pi

        return err