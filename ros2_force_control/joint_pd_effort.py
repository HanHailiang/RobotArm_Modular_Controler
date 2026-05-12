#!/usr/bin/env python3
import sys
from typing import Dict, List, Optional

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


class FrankaRosNode(Node):
    def __init__(self):
        super().__init__("franka_joint_pd_effort_pyqt6")

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

        self.get_logger().info("Franka PyQt6 PD effort node started.")

    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg

    def publish_effort(self, joint_names: List[str], effort: List[float]):
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = joint_names
        cmd.position = []
        cmd.velocity = []
        cmd.effort = effort
        self.publisher.publish(cmd)


class JointRow:
    def __init__(
        self,
        name: str,
        default_q_des: float,
        default_kp: float,
        default_kd: float,
    ):
        self.name = name

        self.enable_box = QCheckBox()
        self.enable_box.setChecked(True)

        self.name_label = QLabel(name)
        self.q_label = QLabel("0.000")
        self.dq_label = QLabel("0.000")
        self.err_label = QLabel("0.000")
        self.tau_label = QLabel("0.000")

        self.q_des_spin = QDoubleSpinBox()
        self.q_des_spin.setRange(-10.0, 10.0)
        self.q_des_spin.setDecimals(4)
        self.q_des_spin.setSingleStep(0.05)
        self.q_des_spin.setValue(default_q_des)

        self.kp_spin = QDoubleSpinBox()
        self.kp_spin.setRange(0.0, 1000.0)
        self.kp_spin.setDecimals(3)
        self.kp_spin.setSingleStep(1.0)
        self.kp_spin.setValue(default_kp)

        self.kd_spin = QDoubleSpinBox()
        self.kd_spin.setRange(0.0, 200.0)
        self.kd_spin.setDecimals(3)
        self.kd_spin.setSingleStep(0.1)
        self.kd_spin.setValue(default_kd)

        for label in [
            self.name_label,
            self.q_label,
            self.dq_label,
            self.err_label,
            self.tau_label,
        ]:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)


class FrankaPDWindow(QMainWindow):
    def __init__(self, ros_node: FrankaRosNode):
        super().__init__()

        self.ros_node = ros_node
        self.joint_names = ros_node.joint_names

        self.q_map: Dict[str, float] = {}
        self.dq_map: Dict[str, float] = {}
        self.last_tau: Dict[str, float] = {name: 0.0 for name in self.joint_names}

        self.default_q_des = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.5,
            "panda_joint3": 0.0,
            "panda_joint4": -2.0,
            "panda_joint5": 0.0,
            "panda_joint6": 1.5,
            "panda_joint7": 0.7,
        }

        self.default_kp = {
            "panda_joint1": 20.0,
            "panda_joint2": 30.0,
            "panda_joint3": 20.0,
            "panda_joint4": 20.0,
            "panda_joint5": 10.0,
            "panda_joint6": 10.0,
            "panda_joint7": 5.0,
        }

        self.default_kd = {
            "panda_joint1": 2.0,
            "panda_joint2": 3.0,
            "panda_joint3": 2.0,
            "panda_joint4": 2.0,
            "panda_joint5": 1.0,
            "panda_joint6": 1.0,
            "panda_joint7": 0.5,
        }

        self.rows: Dict[str, JointRow] = {}

        self.setWindowTitle("Franka PyQt6 Joint PD Effort Controller")
        self.resize(1280, 680)

        self.build_ui()
        self.setup_timers()

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        control_group = QGroupBox("Global Control")
        control_layout = QHBoxLayout(control_group)

        self.enable_control_box = QCheckBox("Enable effort control")
        self.enable_control_box.setChecked(False)

        self.publish_zero_box = QCheckBox("Publish zero effort when disabled")
        self.publish_zero_box.setChecked(True)

        self.max_tau_spin = QDoubleSpinBox()
        self.max_tau_spin.setRange(0.1, 300.0)
        self.max_tau_spin.setDecimals(2)
        self.max_tau_spin.setSingleStep(1.0)
        self.max_tau_spin.setValue(30.0)

        self.rate_spin = QSpinBox()
        self.rate_spin.setRange(1, 500)
        self.rate_spin.setValue(50)

        self.set_current_button = QPushButton("Set q_des = current q")
        self.zero_button = QPushButton("Zero effort now")
        self.reset_button = QPushButton("Reset defaults")
        self.single_joint6_button = QPushButton("Only enable joint6")
        self.enable_all_button = QPushButton("Enable all joints")
        self.disable_all_button = QPushButton("Disable all joints")

        control_layout.addWidget(self.enable_control_box)
        control_layout.addWidget(self.publish_zero_box)
        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("max_tau"))
        control_layout.addWidget(self.max_tau_spin)
        control_layout.addWidget(QLabel("rate Hz"))
        control_layout.addWidget(self.rate_spin)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.set_current_button)
        control_layout.addWidget(self.zero_button)
        control_layout.addWidget(self.reset_button)
        control_layout.addWidget(self.single_joint6_button)
        control_layout.addWidget(self.enable_all_button)
        control_layout.addWidget(self.disable_all_button)

        main_layout.addWidget(control_group)

        table_group = QGroupBox("Joint Parameters")
        grid = QGridLayout(table_group)

        headers = [
            "Joint",
            "Enable",
            "q current",
            "dq current",
            "q_des",
            "Kp",
            "Kd",
            "error",
            "tau",
        ]

        for col, text in enumerate(headers):
            label = QLabel(text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("font-weight: bold;")
            grid.addWidget(label, 0, col)

        for row_index, name in enumerate(self.joint_names, start=1):
            row = JointRow(
                name=name,
                default_q_des=self.default_q_des[name],
                default_kp=self.default_kp[name],
                default_kd=self.default_kd[name],
            )
            self.rows[name] = row

            grid.addWidget(row.name_label, row_index, 0)
            grid.addWidget(row.enable_box, row_index, 1)
            grid.addWidget(row.q_label, row_index, 2)
            grid.addWidget(row.dq_label, row_index, 3)
            grid.addWidget(row.q_des_spin, row_index, 4)
            grid.addWidget(row.kp_spin, row_index, 5)
            grid.addWidget(row.kd_spin, row_index, 6)
            grid.addWidget(row.err_label, row_index, 7)
            grid.addWidget(row.tau_label, row_index, 8)

        main_layout.addWidget(table_group)

        self.status_label = QLabel("Waiting for /joint_states ...")
        self.status_label.setStyleSheet("font-size: 14px; padding: 6px;")
        main_layout.addWidget(self.status_label)

        self.set_current_button.clicked.connect(self.set_q_des_to_current)
        self.zero_button.clicked.connect(self.publish_zero_effort)
        self.reset_button.clicked.connect(self.reset_defaults)
        self.single_joint6_button.clicked.connect(self.only_enable_joint6)
        self.enable_all_button.clicked.connect(self.enable_all_joints)
        self.disable_all_button.clicked.connect(self.disable_all_joints)

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

        self.rate_spin.valueChanged.connect(self.update_control_timer)

    def update_control_timer(self):
        rate_hz = max(1, int(self.rate_spin.value()))
        interval_ms = max(2, int(1000.0 / rate_hz))
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

    @staticmethod
    def clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def compute_tau(self) -> Optional[List[float]]:
        if not self.parse_joint_state():
            self.status_label.setText("Waiting for /joint_states ...")
            return None

        max_tau = abs(float(self.max_tau_spin.value()))
        tau_cmd: List[float] = []

        for name in self.joint_names:
            if name not in self.q_map:
                self.status_label.setText(f"Joint name mismatch: {name} not found.")
                return None

            row = self.rows[name]

            q = self.q_map.get(name, 0.0)
            dq = self.dq_map.get(name, 0.0)
            q_des = row.q_des_spin.value()
            kp = row.kp_spin.value()
            kd = row.kd_spin.value()

            if row.enable_box.isChecked():
                err = q_des - q
                tau = kp * err - kd * dq
                tau = self.clamp(tau, max_tau)
            else:
                tau = 0.0

            tau_cmd.append(float(tau))
            self.last_tau[name] = float(tau)

        return tau_cmd

    def control_loop(self):
        if self.enable_control_box.isChecked():
            tau_cmd = self.compute_tau()
            if tau_cmd is not None:
                self.ros_node.publish_effort(self.joint_names, tau_cmd)
                self.status_label.setText("Publishing effort command.")
        else:
            if self.publish_zero_box.isChecked():
                self.publish_zero_effort()
            self.status_label.setText("Control disabled.")

    def publish_zero_effort(self):
        zero_tau = [0.0] * len(self.joint_names)
        self.ros_node.publish_effort(self.joint_names, zero_tau)

        for name in self.joint_names:
            self.last_tau[name] = 0.0

    def refresh_ui(self):
        self.parse_joint_state()

        for name in self.joint_names:
            row = self.rows[name]

            q = self.q_map.get(name, 0.0)
            dq = self.dq_map.get(name, 0.0)
            q_des = row.q_des_spin.value()
            err = q_des - q
            tau = self.last_tau.get(name, 0.0)

            row.q_label.setText(f"{q:.4f}")
            row.dq_label.setText(f"{dq:.4f}")
            row.err_label.setText(f"{err:.4f}")
            row.tau_label.setText(f"{tau:.4f}")

    def set_q_des_to_current(self):
        if not self.parse_joint_state():
            self.status_label.setText("No /joint_states received.")
            return

        for name in self.joint_names:
            if name in self.q_map:
                self.rows[name].q_des_spin.setValue(self.q_map[name])

        self.status_label.setText("Set q_des to current q.")

    def reset_defaults(self):
        for name in self.joint_names:
            row = self.rows[name]
            row.q_des_spin.setValue(self.default_q_des[name])
            row.kp_spin.setValue(self.default_kp[name])
            row.kd_spin.setValue(self.default_kd[name])
            row.enable_box.setChecked(True)

        self.max_tau_spin.setValue(30.0)
        self.rate_spin.setValue(50)
        self.status_label.setText("Defaults reset.")

    def only_enable_joint6(self):
        for name in self.joint_names:
            self.rows[name].enable_box.setChecked(name == "panda_joint6")

        if "panda_joint6" in self.q_map:
            self.rows["panda_joint6"].q_des_spin.setValue(self.q_map["panda_joint6"] + 0.3)

        self.max_tau_spin.setValue(10.0)
        self.rows["panda_joint6"].kp_spin.setValue(20.0)
        self.rows["panda_joint6"].kd_spin.setValue(2.0)

        self.status_label.setText("Only joint6 enabled. q_des = current + 0.3 if available.")

    def enable_all_joints(self):
        for name in self.joint_names:
            self.rows[name].enable_box.setChecked(True)
        self.status_label.setText("All joints enabled.")

    def disable_all_joints(self):
        for name in self.joint_names:
            self.rows[name].enable_box.setChecked(False)
        self.status_label.setText("All joints disabled.")

    def closeEvent(self, event):
        self.publish_zero_effort()
        event.accept()


def main():
    rclpy.init()

    ros_node = FrankaRosNode()

    app = QApplication(sys.argv)
    window = FrankaPDWindow(ros_node)
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