import numpy as np


def clamp_vector(v: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(v, -abs(limit), abs(limit))


def damped_pinv_jacobian(J: np.ndarray, damping_lambda: float) -> np.ndarray:
    """阻尼伪逆。J 为 3x7 时返回 7x3。"""
    lam2 = damping_lambda ** 2
    JJt = J @ J.T
    return J.T @ np.linalg.inv(JJt + lam2 * np.eye(J.shape[0]))


def nullspace_projector(J: np.ndarray, damping_lambda: float) -> np.ndarray:
    """N = I - J^T * J_pinv^T，用于把力矩投影到位置任务的零空间。"""
    J_pinv = damped_pinv_jacobian(J, damping_lambda)
    n = J.shape[1]
    return np.eye(n) - J.T @ J_pinv.T
