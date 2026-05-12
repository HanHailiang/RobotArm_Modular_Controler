#!/usr/bin/env python3
import sys
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class PandaKinematics:
    """
    一个用于学习和调试的 Panda 近似运动学模型。

    说明：
    1. 这里使用硬编码 DH 参数做 FK 和数值雅可比。
    2. 末端位置保持效果取决于这个模型和 Isaac Sim 中 USD 模型的一致性。
    3. 后面要做高精度控制时，建议换成 Pinocchio / KDL / Isaac Sim 真实 Jacobian。
    """

    def __init__(self):
        # 近似 Panda DH 参数，单位 m
        # 用于学习控制结构足够；精确工程控制建议用 URDF/Pinocchio。
        self.a = np.array([0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088], dtype=float)
        self.d = np.array([0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107], dtype=float)
        self.alpha = np.array(
            [0.0, -math.pi / 2.0, math.pi / 2.0, math.pi / 2.0,
             -math.pi / 2.0, math.pi / 2.0, math.pi / 2.0],
            dtype=float,
        )

        # 末端工具偏移，可根据 Isaac Sim 中末端实际位置微调
        self.tool_offset = np.array([0.0, 0.0, 0.103], dtype=float)

    @staticmethod
    def _dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        ct = math.cos(theta)
        st = math.sin(theta)
        ca = math.cos(alpha)
        sa = math.sin(alpha)

        return np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def fk(self, q: np.ndarray) -> np.ndarray:
        """
        返回近似末端位置 p = [x, y, z]
        """
        T = np.eye(4)

        for i in range(7):
            T = T @ self._dh_transform(
                a=float(self.a[i]),
                alpha=float(self.alpha[i]),
                d=float(self.d[i]),
                theta=float(q[i]),
            )

        p = T[:3, 3] + T[:3, :3] @ self.tool_offset
        return p

    def numerical_jacobian(self, q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        数值计算末端位置雅可比 J_pos，大小 3x7。
        """
        J = np.zeros((3, 7), dtype=float)

        for i in range(7):
            dq = np.zeros(7, dtype=float)
            dq[i] = eps

            p_plus = self.fk(q + dq)
            p_minus = self.fk(q - dq)

            J[:, i] = (p_plus - p_minus) / (2.0 * eps)

        return J


class FrankaRosNode(Node):
    def __init__(self):
        super().__init__("franka_cartesian_nullspace_pyqt6")

        self.joint_names: List[str] = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]

        self.latest_joint_state: Optional[JointState] = None

        self.publisher = self.create_publisher(
            JointState,
            "/joint_command",
            10,
        )

        self.subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )

        self.get_logger().info("Cartesian nullspace PyQt6 controller started.")

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def publish_effort(self, effort: List[float]):
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = self.joint_names
        cmd.position = []
        cmd.velocity = []
        cmd.effort = effort
        self.publisher.publish(cmd)


class CartesianNullspaceWindow(QMainWindow):
    def __init__(self, ros_node: FrankaRosNode):
        super().__init__()

        self.ros_node = ros_node
        self.joint_names = ros_node.joint_names
        self.kin = PandaKinematics()

        self.q_map: Dict[str, float] = {}
        self.dq_map: Dict[str, float] = {}

        self.q_home: Optional[np.ndarray] = None
        self.p_des: Optional[np.ndarray] = None

        self.last_tau = np.zeros(7, dtype=float)
        self.last_p = np.zeros(3, dtype=float)
        self.last_p_err = np.zeros(3, dtype=float)
        self.last_tau_cart = np.zeros(7, dtype=float)
        self.last_tau_null = np.zeros(7, dtype=float)
        self.last_tau_dist = np.zeros(7, dtype=float)

        self.setWindowTitle("Franka Cartesian Position Hold + Nullspace Compliance")
        self.resize(1280, 760)

        self.build_ui()
        self.setup_timers()

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ---------------- Global Control ----------------
        global_group = QGroupBox("Global Control")
        global_layout = QGridLayout(global_group)

        self.enable_control_box = QCheckBox("Enable controller")
        self.enable_control_box.setChecked(False)

        self.publish_zero_box = QCheckBox("Publish zero effort when disabled")
        self.publish_zero_box.setChecked(True)

        self.rate_spin = QSpinBox()
        self.rate_spin.setRange(1, 500)
        self.rate_spin.setValue(50)

        self.max_tau_spin = QDoubleSpinBox()
        self.max_tau_spin.setRange(0.1, 300.0)
        self.max_tau_spin.setDecimals(2)
        self.max_tau_spin.setSingleStep(1.0)
        self.max_tau_spin.setValue(40.0)

        self.damping_lambda_spin = QDoubleSpinBox()
        self.damping_lambda_spin.setRange(0.0001, 1.0)
        self.damping_lambda_spin.setDecimals(4)
        self.damping_lambda_spin.setSingleStep(0.001)
        self.damping_lambda_spin.setValue(0.03)

        self.capture_home_button = QPushButton("Capture current as home / end-effector target")
        self.zero_button = QPushButton("Zero effort now")

        global_layout.addWidget(self.enable_control_box, 0, 0)
        global_layout.addWidget(self.publish_zero_box, 0, 1)
        global_layout.addWidget(QLabel("rate Hz"), 0, 2)
        global_layout.addWidget(self.rate_spin, 0, 3)
        global_layout.addWidget(QLabel("max_tau"), 0, 4)
        global_layout.addWidget(self.max_tau_spin, 0, 5)
        global_layout.addWidget(QLabel("pinv damping λ"), 0, 6)
        global_layout.addWidget(self.damping_lambda_spin, 0, 7)

        global_layout.addWidget(self.capture_home_button, 1, 0, 1, 4)
        global_layout.addWidget(self.zero_button, 1, 4, 1, 2)

        main_layout.addWidget(global_group)

        # ---------------- Cartesian impedance ----------------
        cart_group = QGroupBox("End-effector Position Hold")
        cart_layout = QGridLayout(cart_group)

        self.kx_spin = QDoubleSpinBox()
        self.kx_spin.setRange(0.0, 5000.0)
        self.kx_spin.setDecimals(2)
        self.kx_spin.setSingleStep(10.0)
        self.kx_spin.setValue(200.0)

        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.0, 1000.0)
        self.dx_spin.setDecimals(2)
        self.dx_spin.setSingleStep(1.0)
        self.dx_spin.setValue(30.0)

        self.p_current_label = QLabel("p current: [0.000, 0.000, 0.000]")
        self.p_des_label = QLabel("p target: not captured")
        self.p_err_label = QLabel("p error: [0.000, 0.000, 0.000], norm=0.000")

        cart_layout.addWidget(QLabel("Kx position stiffness"), 0, 0)
        cart_layout.addWidget(self.kx_spin, 0, 1)
        cart_layout.addWidget(QLabel("Dx position damping"), 0, 2)
        cart_layout.addWidget(self.dx_spin, 0, 3)

        cart_layout.addWidget(self.p_current_label, 1, 0, 1, 4)
        cart_layout.addWidget(self.p_des_label, 2, 0, 1, 4)
        cart_layout.addWidget(self.p_err_label, 3, 0, 1, 4)

        main_layout.addWidget(cart_group)

        # ---------------- Nullspace ----------------
        null_group = QGroupBox("Null-space Posture Recovery")
        null_layout = QGridLayout(null_group)

        self.kq_spin = QDoubleSpinBox()
        self.kq_spin.setRange(0.0, 500.0)
        self.kq_spin.setDecimals(2)
        self.kq_spin.setSingleStep(1.0)
        self.kq_spin.setValue(5.0)

        self.dq_spin = QDoubleSpinBox()
        self.dq_spin.setRange(0.0, 100.0)
        self.dq_spin.setDecimals(2)
        self.dq_spin.setSingleStep(0.1)
        self.dq_spin.setValue(1.0)

        self.dist_joint6_spin = QDoubleSpinBox()
        self.dist_joint6_spin.setRange(-200.0, 200.0)
        self.dist_joint6_spin.setDecimals(2)
        self.dist_joint6_spin.setSingleStep(1.0)
        self.dist_joint6_spin.setValue(0.0)

        self.project_disturbance_box = QCheckBox("Project joint6 disturbance into null-space")
        self.project_disturbance_box.setChecked(True)

        null_layout.addWidget(QLabel("Kq posture stiffness"), 0, 0)
        null_layout.addWidget(self.kq_spin, 0, 1)
        null_layout.addWidget(QLabel("Dq posture damping"), 0, 2)
        null_layout.addWidget(self.dq_spin, 0, 3)

        null_layout.addWidget(QLabel("joint6 disturbance tau"), 1, 0)
        null_layout.addWidget(self.dist_joint6_spin, 1, 1)
        null_layout.addWidget(self.project_disturbance_box, 1, 2, 1, 2)

        main_layout.addWidget(null_group)

        # ---------------- Joint monitor ----------------
        joint_group = QGroupBox("Joint Monitor")
        joint_layout = QGridLayout(joint_group)

        headers = ["Joint", "q", "dq", "tau_total", "tau_cart", "tau_null", "tau_dist"]
        for col, text in enumerate(headers):
            label = QLabel(text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("font-weight: bold;")
            joint_layout.addWidget(label, 0, col)

        self.q_labels = {}
        self.dq_labels = {}
        self.tau_labels = {}
        self.tau_cart_labels = {}
        self.tau_null_labels = {}
        self.tau_dist_labels = {}

        for row, name in enumerate(self.joint_names, start=1):
            name_label = QLabel(name)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            self.q_labels[name] = QLabel("0.000")
            self.dq_labels[name] = QLabel("0.000")
            self.tau_labels[name] = QLabel("0.000")
            self.tau_cart_labels[name] = QLabel("0.000")
            self.tau_null_labels[name] = QLabel("0.000")
            self.tau_dist_labels[name] = QLabel("0.000")

            labels = [
                name_label,
                self.q_labels[name],
                self.dq_labels[name],
                self.tau_labels[name],
                self.tau_cart_labels[name],
                self.tau_null_labels[name],
                self.tau_dist_labels[name],
            ]

            for col, label in enumerate(labels):
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                joint_layout.addWidget(label, row, col)

        main_layout.addWidget(joint_group)

        self.status_label = QLabel("Waiting for /joint_states ...")
        self.status_label.setStyleSheet("font-size: 14px; padding: 6px;")
        main_layout.addWidget(self.status_label)

        self.capture_home_button.clicked.connect(self.capture_home)
        self.zero_button.clicked.connect(self.publish_zero_effort)
        self.rate_spin.valueChanged.connect(self.update_control_timer)

        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f4f6f8;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #c6cbd1;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QLabel {
                font-size: 13px;
            }
            QDoubleSpinBox, QSpinBox {
                min-height: 28px;
                font-size: 13px;
                min-width: 90px;
            }
            QPushButton {
                min-height: 30px;
                padding: 4px 10px;
                border-radius: 6px;
                background-color: #e8edf3;
            }
            QPushButton:hover {
                background-color: #dce6f2;
            }
            QCheckBox {
                font-size: 13px;
            }
            """
        )

    def setup_timers(self):
        self.ros_timer = QTimer(self)
        self.ros_timer.timeout.connect(self.spin_ros_once)
        self.ros_timer.start(5)

        self.control_timer = QTimer(self)
        self.control_timer.timeout.connect(self.control_loop)
        self.update_control_timer()

        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.refresh_ui)
        self.ui_timer.start(100)

    def update_control_timer(self):
        rate = max(1, int(self.rate_spin.value()))
        interval_ms = max(2, int(1000.0 / rate))
        self.control_timer.start(interval_ms)

    def spin_ros_once(self):
        try:
            rclpy.spin_once(self.ros_node, timeout_sec=0.0)
        except Exception as e:
            self.status_label.setText(f"ROS spin error: {e}")

    def parse_joint_state(self) -> bool:
        msg = self.ros_node.latest_joint_state
        if msg is None:
            return False

        q_map: Dict[str, float] = {}
        dq_map: Dict[str, float] = {}

        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                q_map[name] = msg.position[i]

            if i < len(msg.velocity):
                dq_map[name] = msg.velocity[i]
            else:
                dq_map[name] = 0.0

        self.q_map = q_map
        self.dq_map = dq_map
        return True

    def get_q_dq_arrays(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.parse_joint_state():
            return None

        q = np.zeros(7, dtype=float)
        dq = np.zeros(7, dtype=float)

        for i, name in enumerate(self.joint_names):
            if name not in self.q_map:
                self.status_label.setText(f"Joint name mismatch: {name} not found.")
                return None

            q[i] = self.q_map[name]
            dq[i] = self.dq_map.get(name, 0.0)

        return q, dq

    @staticmethod
    def clamp_vector(v: np.ndarray, limit: float) -> np.ndarray:
        return np.clip(v, -abs(limit), abs(limit))

    @staticmethod
    def damped_pinv_jacobian(J: np.ndarray, damping_lambda: float) -> np.ndarray:
        """
        J: 3x7
        返回 J_pinv: 7x3
        """
        lam2 = damping_lambda ** 2
        JJt = J @ J.T
        return J.T @ np.linalg.inv(JJt + lam2 * np.eye(3))

    def capture_home(self):
        result = self.get_q_dq_arrays()
        if result is None:
            self.status_label.setText("Cannot capture home: no valid /joint_states.")
            return

        q, _ = result
        self.q_home = q.copy()
        self.p_des = self.kin.fk(q)

        self.status_label.setText(
            f"Captured home q and target p = [{self.p_des[0]:.3f}, {self.p_des[1]:.3f}, {self.p_des[2]:.3f}]"
        )

    def compute_control_tau(self) -> Optional[np.ndarray]:
        result = self.get_q_dq_arrays()
        if result is None:
            self.status_label.setText("Waiting for valid /joint_states ...")
            return None

        q, dq = result

        if self.q_home is None or self.p_des is None:
            self.status_label.setText("Please click 'Capture current as home / end-effector target' first.")
            return None

        # 当前末端位置和位置雅可比
        p = self.kin.fk(q)
        J = self.kin.numerical_jacobian(q)

        p_dot = J @ dq
        p_err = self.p_des - p

        Kx = float(self.kx_spin.value())
        Dx = float(self.dx_spin.value())

        # 末端位置保持力
        F_pos = Kx * p_err - Dx * p_dot

        # 任务空间力转关节力矩
        tau_cart = J.T @ F_pos

        # 零空间投影
        damping_lambda = float(self.damping_lambda_spin.value())
        J_pinv = self.damped_pinv_jacobian(J, damping_lambda)
        N = np.eye(7) - J.T @ J_pinv.T

        # 零空间姿态恢复
        Kq = float(self.kq_spin.value())
        Dq = float(self.dq_spin.value())

        tau_null_raw = Kq * (self.q_home - q) - Dq * dq

        # joint6 外部扰动力矩
        tau_dist_raw = np.zeros(7, dtype=float)
        tau_dist_raw[5] = float(self.dist_joint6_spin.value())

        if self.project_disturbance_box.isChecked():
            tau_extra = N @ (tau_null_raw + tau_dist_raw)
            tau_null_projected = N @ tau_null_raw
            tau_dist_projected = N @ tau_dist_raw
        else:
            tau_extra = N @ tau_null_raw + tau_dist_raw
            tau_null_projected = N @ tau_null_raw
            tau_dist_projected = tau_dist_raw

        tau_total = tau_cart + tau_extra

        max_tau = float(self.max_tau_spin.value())
        tau_total = self.clamp_vector(tau_total, max_tau)

        self.last_tau = tau_total.copy()
        self.last_tau_cart = tau_cart.copy()
        self.last_tau_null = tau_null_projected.copy()
        self.last_tau_dist = tau_dist_projected.copy()
        self.last_p = p.copy()
        self.last_p_err = p_err.copy()

        return tau_total

    def control_loop(self):
        if self.enable_control_box.isChecked():
            tau = self.compute_control_tau()
            if tau is not None:
                self.ros_node.publish_effort(tau.tolist())
                self.status_label.setText("Publishing Cartesian hold + nullspace effort command.")
        else:
            if self.publish_zero_box.isChecked():
                self.publish_zero_effort()
            self.status_label.setText("Control disabled.")

    def publish_zero_effort(self):
        zero_tau = np.zeros(7, dtype=float)
        self.last_tau = zero_tau.copy()
        self.last_tau_cart = zero_tau.copy()
        self.last_tau_null = zero_tau.copy()
        self.last_tau_dist = zero_tau.copy()
        self.ros_node.publish_effort(zero_tau.tolist())

    def refresh_ui(self):
        result = self.get_q_dq_arrays()

        if result is not None:
            q, dq = result
            p = self.kin.fk(q)
            self.last_p = p.copy()

            if self.p_des is not None:
                p_err = self.p_des - p
            else:
                p_err = np.zeros(3, dtype=float)

            self.last_p_err = p_err.copy()

            self.p_current_label.setText(
                f"p current: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]"
            )

            if self.p_des is not None:
                self.p_des_label.setText(
                    f"p target:  [{self.p_des[0]:.4f}, {self.p_des[1]:.4f}, {self.p_des[2]:.4f}]"
                )
            else:
                self.p_des_label.setText("p target: not captured")

            self.p_err_label.setText(
                f"p error: [{p_err[0]:.4f}, {p_err[1]:.4f}, {p_err[2]:.4f}], norm={np.linalg.norm(p_err):.4f}"
            )

            for i, name in enumerate(self.joint_names):
                self.q_labels[name].setText(f"{q[i]:.4f}")
                self.dq_labels[name].setText(f"{dq[i]:.4f}")
                self.tau_labels[name].setText(f"{self.last_tau[i]:.4f}")
                self.tau_cart_labels[name].setText(f"{self.last_tau_cart[i]:.4f}")
                self.tau_null_labels[name].setText(f"{self.last_tau_null[i]:.4f}")
                self.tau_dist_labels[name].setText(f"{self.last_tau_dist[i]:.4f}")

    def closeEvent(self, event):
        self.publish_zero_effort()
        event.accept()


def main():
    rclpy.init()

    ros_node = FrankaRosNode()

    app = QApplication(sys.argv)
    window = CartesianNullspaceWindow(ros_node)
    window.show()

    try:
        exit_code = app.exec()
    finally:
        window.publish_zero_effort()
        ros_node.destroy_node()
        rclpy.shutdown()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()