from typing import Optional, Tuple

import numpy as np

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ros_interface.franka_ros_node import FrankaRosNode, spin_once_safe
from core.robot_state import RobotStateBuffer
from controllers.cartesian_pose_ik import CartesianPoseIKController
from controllers.cartesian_twist_controller import CartesianTwistController


class CartesianNullspaceWindow(QMainWindow):
    """
    PyQt6 主窗口。

    支持两种末端控制模式：

    1. Pose IK Jog
       连续小步目标位姿 IK 控制。
       按住 X+/Y-/Rz+ 时，持续修改 target_pose6，然后调用 IK，再发布 q_goal。

    2. Twist Jog
       末端笛卡尔速度控制雏形。
       按住 X+/Y-/Rz+ 时，生成 twist = [vx, vy, vz, wx, wy, wz]，
       然后通过 Jacobian 反解 dq，再积分得到 q_target，最后 publish_position(q_target)。

    目前 Twist Jog 仍然通过 position command 实现速度效果。
    如果后续接入真正的 velocity controller，可以直接发布 dq_cmd。
    """

    MODE_POSE_IK = "Pose IK Jog"
    MODE_TWIST = "Twist Jog"

    def __init__(
        self,
        cfg,
        ros_node: FrankaRosNode,
        state_buffer: RobotStateBuffer,
        pose_ik_controller: CartesianPoseIKController,
    ):
        super().__init__()

        self.cfg = cfg
        self.ros_node = ros_node
        self.state_buffer = state_buffer
        self.pose_ik_controller = pose_ik_controller

        self.joint_names = self.cfg.robot.joint_names

        self.q_goal: Optional[np.ndarray] = None
        self.target_pose6: Optional[np.ndarray] = None

        # Twist Jog 内部积分目标
        # 用于保存上一帧的 q_target，避免每次都用 q_current + dq * dt
        self.twist_q_target: Optional[np.ndarray] = None

        # 连续 Jog 状态
        self.active_jog_axis: Optional[str] = None
        self.active_jog_direction: float = 0.0

        # Twist 控制器
        self.twist_controller = self._create_twist_controller()

        self.setWindowTitle("Franka Cartesian Pose IK / Twist Jog Control")
        self.resize(1320, 900)

        self.build_ui()
        self.setup_timers()

    # ============================================================
    # Controller 构建
    # ============================================================

    def _create_twist_controller(self) -> CartesianTwistController:
        joint_lower = self._try_get_cfg_array(
            [
                "joint_lower",
                "joint_lower_limits",
                "lower_limits",
            ]
        )
        joint_upper = self._try_get_cfg_array(
            [
                "joint_upper",
                "joint_upper_limits",
                "upper_limits",
            ]
        )

        return CartesianTwistController(
            kin=self.pose_ik_controller.kin,
            dt=0.02,
            damping=0.10,
            max_joint_speed=0.25,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            allow_numerical_jacobian_fallback=False,
            debug=True,
        )

    def _try_get_cfg_array(self, names) -> Optional[np.ndarray]:
        for name in names:
            if hasattr(self.cfg.robot, name):
                value = getattr(self.cfg.robot, name)
                arr = np.asarray(value, dtype=float)
                if arr.shape[0] == 7:
                    return arr
        return None

    # ============================================================
    # UI 构建
    # ============================================================

    def build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self._build_global_control(main_layout)
        self._build_joint_status(main_layout)
        self._build_pose_status(main_layout)
        self._build_target_pose_input(main_layout)
        self._build_jog_control(main_layout)
        self._build_ik_result(main_layout)
        self._build_status_bar(main_layout)

        self._connect_signals()
        self._apply_style()

    def _build_global_control(self, main_layout: QVBoxLayout) -> None:
        global_group = QGroupBox("Global Control")
        global_layout = QGridLayout(global_group)

        self.read_joint_button = QPushButton("Read Current Joint Angles")
        self.fill_pose_button = QPushButton("Fill Target From Current Pose")
        self.move_to_pose_button = QPushButton("Move To Target Pose")
        self.publish_goal_button = QPushButton("Publish Last q_goal Again")
        self.zero_effort_button = QPushButton("Zero Effort")

        self.hold_position_box = QCheckBox("Hold target by republishing q_goal")
        self.hold_position_box.setChecked(False)

        self.hold_rate_spin = QDoubleSpinBox()
        self.hold_rate_spin.setRange(1.0, 200.0)
        self.hold_rate_spin.setDecimals(1)
        self.hold_rate_spin.setSingleStep(1.0)
        self.hold_rate_spin.setValue(30.0)

        self.control_mode_box = QComboBox()
        self.control_mode_box.addItems(
            [
                self.MODE_POSE_IK,
                self.MODE_TWIST,
            ]
        )
        self.control_mode_box.setCurrentText(self.MODE_TWIST)

        global_layout.addWidget(self.read_joint_button, 0, 0)
        global_layout.addWidget(self.fill_pose_button, 0, 1)
        global_layout.addWidget(self.move_to_pose_button, 0, 2)
        global_layout.addWidget(self.publish_goal_button, 0, 3)
        global_layout.addWidget(self.zero_effort_button, 0, 4)

        global_layout.addWidget(QLabel("Control Mode"), 1, 0)
        global_layout.addWidget(self.control_mode_box, 1, 1)

        global_layout.addWidget(self.hold_position_box, 1, 2, 1, 2)
        global_layout.addWidget(QLabel("hold publish rate Hz"), 1, 4)
        global_layout.addWidget(self.hold_rate_spin, 1, 5)

        main_layout.addWidget(global_group)

    def _build_joint_status(self, main_layout: QVBoxLayout) -> None:
        joint_group = QGroupBox("Current Joint Angles")
        joint_layout = QGridLayout(joint_group)

        joint_layout.addWidget(self._header_label("Joint"), 0, 0)
        joint_layout.addWidget(self._header_label("q current / rad"), 0, 1)
        joint_layout.addWidget(self._header_label("q goal / rad"), 0, 2)
        joint_layout.addWidget(self._header_label("error / rad"), 0, 3)

        self.q_current_labels = {}
        self.q_goal_labels = {}
        self.q_error_labels = {}

        for row, name in enumerate(self.joint_names, start=1):
            joint_layout.addWidget(self._center_label(name), row, 0)

            self.q_current_labels[name] = self._center_label("0.0000")
            self.q_goal_labels[name] = self._center_label("-")
            self.q_error_labels[name] = self._center_label("-")

            joint_layout.addWidget(self.q_current_labels[name], row, 1)
            joint_layout.addWidget(self.q_goal_labels[name], row, 2)
            joint_layout.addWidget(self.q_error_labels[name], row, 3)

        main_layout.addWidget(joint_group)

    def _build_pose_status(self, main_layout: QVBoxLayout) -> None:
        pose_status_group = QGroupBox("Current End-effector Pose")
        pose_status_layout = QGridLayout(pose_status_group)

        self.current_pose_labels = {}

        pose_names = ["x", "y", "z", "rx", "ry", "rz"]
        pose_units = ["m", "m", "m", "rad", "rad", "rad"]

        for col, name in enumerate(pose_names):
            pose_status_layout.addWidget(
                self._header_label(f"{name} / {pose_units[col]}"),
                0,
                col,
            )
            self.current_pose_labels[name] = self._center_label("0.0000")
            pose_status_layout.addWidget(self.current_pose_labels[name], 1, col)

        main_layout.addWidget(pose_status_group)

    def _build_target_pose_input(self, main_layout: QVBoxLayout) -> None:
        target_group = QGroupBox("Target End-effector Pose Input")
        target_layout = QGridLayout(target_group)

        self.pose_spins = {}

        pose_names = ["x", "y", "z", "rx", "ry", "rz"]

        for col, name in enumerate(pose_names):
            target_layout.addWidget(self._header_label(name), 0, col)

            spin = QDoubleSpinBox()
            spin.setDecimals(5)
            spin.setSingleStep(0.01)
            spin.setRange(-10.0, 10.0)

            if name in ["rx", "ry", "rz"]:
                spin.setSingleStep(0.05)
                spin.setRange(-np.pi * 2.0, np.pi * 2.0)

            self.pose_spins[name] = spin
            target_layout.addWidget(spin, 1, col)

        main_layout.addWidget(target_group)

    def _build_jog_control(self, main_layout: QVBoxLayout) -> None:
        jog_group = QGroupBox("Joystick-like Jog Control")
        jog_layout = QGridLayout(jog_group)

        # Pose IK Jog 使用：每个 tick 的位移增量
        self.pos_step_spin = QDoubleSpinBox()
        self.pos_step_spin.setRange(0.0001, 0.5)
        self.pos_step_spin.setDecimals(4)
        self.pos_step_spin.setSingleStep(0.005)
        self.pos_step_spin.setValue(0.005)

        self.rot_step_spin = QDoubleSpinBox()
        self.rot_step_spin.setRange(0.0001, 1.0)
        self.rot_step_spin.setDecimals(4)
        self.rot_step_spin.setSingleStep(0.01)
        self.rot_step_spin.setValue(0.02)

        # Twist Jog 使用：末端速度
        self.linear_speed_spin = QDoubleSpinBox()
        self.linear_speed_spin.setRange(0.001, 0.3)
        self.linear_speed_spin.setDecimals(4)
        self.linear_speed_spin.setSingleStep(0.005)
        self.linear_speed_spin.setValue(0.03)

        self.angular_speed_spin = QDoubleSpinBox()
        self.angular_speed_spin.setRange(0.001, 2.0)
        self.angular_speed_spin.setDecimals(4)
        self.angular_speed_spin.setSingleStep(0.02)
        self.angular_speed_spin.setValue(0.15)

        # 两种模式都用的 Jog 频率
        self.jog_rate_spin = QDoubleSpinBox()
        self.jog_rate_spin.setRange(1.0, 100.0)
        self.jog_rate_spin.setDecimals(1)
        self.jog_rate_spin.setSingleStep(1.0)
        self.jog_rate_spin.setValue(50.0)

        jog_layout.addWidget(QLabel("Pose step / m"), 0, 0)
        jog_layout.addWidget(self.pos_step_spin, 0, 1)

        jog_layout.addWidget(QLabel("Pose rot step / rad"), 0, 2)
        jog_layout.addWidget(self.rot_step_spin, 0, 3)

        jog_layout.addWidget(QLabel("Twist linear / m/s"), 1, 0)
        jog_layout.addWidget(self.linear_speed_spin, 1, 1)

        jog_layout.addWidget(QLabel("Twist angular / rad/s"), 1, 2)
        jog_layout.addWidget(self.angular_speed_spin, 1, 3)

        jog_layout.addWidget(QLabel("Jog rate / Hz"), 1, 4)
        jog_layout.addWidget(self.jog_rate_spin, 1, 5)

        self.x_minus_button, self.x_plus_button = self._add_jog_axis_control(
            jog_layout, row=2, col=0, axis_name="X", axis_key="x"
        )
        self.y_minus_button, self.y_plus_button = self._add_jog_axis_control(
            jog_layout, row=2, col=3, axis_name="Y", axis_key="y"
        )
        self.z_minus_button, self.z_plus_button = self._add_jog_axis_control(
            jog_layout, row=2, col=6, axis_name="Z", axis_key="z"
        )

        self.rx_minus_button, self.rx_plus_button = self._add_jog_axis_control(
            jog_layout, row=3, col=0, axis_name="Rx", axis_key="rx"
        )
        self.ry_minus_button, self.ry_plus_button = self._add_jog_axis_control(
            jog_layout, row=3, col=3, axis_name="Ry", axis_key="ry"
        )
        self.rz_minus_button, self.rz_plus_button = self._add_jog_axis_control(
            jog_layout, row=3, col=6, axis_name="Rz", axis_key="rz"
        )

        main_layout.addWidget(jog_group)

    def _build_ik_result(self, main_layout: QVBoxLayout) -> None:
        ik_group = QGroupBox("IK / Twist Result")
        ik_layout = QVBoxLayout(ik_group)

        self.ik_result_label = QLabel("Result: -")
        self.ik_result_label.setStyleSheet("font-size: 13px; padding: 6px;")

        ik_layout.addWidget(self.ik_result_label)
        main_layout.addWidget(ik_group)

    def _build_status_bar(self, main_layout: QVBoxLayout) -> None:
        self.status_label = QLabel("Waiting for /joint_states ...")
        self.status_label.setStyleSheet("font-size: 14px; padding: 6px;")
        main_layout.addWidget(self.status_label)

    def _add_jog_axis_control(
        self,
        layout: QGridLayout,
        row: int,
        col: int,
        axis_name: str,
        axis_key: str,
    ) -> Tuple[QPushButton, QPushButton]:
        minus_btn = QPushButton("−")
        plus_btn = QPushButton("+")
        axis_label = QLabel(axis_name)

        minus_btn.setObjectName("JogMinusButton")
        plus_btn.setObjectName("JogPlusButton")
        axis_label.setObjectName("JogAxisLabel")

        minus_btn.setMinimumSize(70, 56)
        plus_btn.setMinimumSize(70, 56)
        axis_label.setMinimumSize(70, 56)

        minus_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        plus_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        axis_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        minus_btn.pressed.connect(lambda: self.start_jog(axis_key, -1.0))
        plus_btn.pressed.connect(lambda: self.start_jog(axis_key, 1.0))

        minus_btn.released.connect(self.stop_jog)
        plus_btn.released.connect(self.stop_jog)

        layout.addWidget(minus_btn, row, col)
        layout.addWidget(axis_label, row, col + 1)
        layout.addWidget(plus_btn, row, col + 2)

        return minus_btn, plus_btn

    def _connect_signals(self) -> None:
        self.read_joint_button.clicked.connect(self.on_read_current_joints)
        self.fill_pose_button.clicked.connect(self.on_fill_target_from_current_pose)
        self.move_to_pose_button.clicked.connect(self.on_move_to_target_pose)
        self.publish_goal_button.clicked.connect(self.on_publish_last_q_goal)
        self.zero_effort_button.clicked.connect(self.ros_node.publish_zero_effort)

        self.hold_rate_spin.valueChanged.connect(self.update_hold_timer)
        self.jog_rate_spin.valueChanged.connect(self.update_jog_timer_interval)

        self.control_mode_box.currentTextChanged.connect(self.on_control_mode_changed)

    def _apply_style(self) -> None:
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

            QDoubleSpinBox,
            QComboBox {
                min-height: 28px;
                font-size: 13px;
                min-width: 100px;
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

            QPushButton:pressed {
                background-color: #cbdcf0;
            }

            QCheckBox {
                font-size: 13px;
            }

            QPushButton#JogMinusButton,
            QPushButton#JogPlusButton {
                min-height: 54px;
                min-width: 68px;
                font-size: 24px;
                font-weight: bold;
                border-radius: 12px;
                border: 1px solid #b9c3cf;
                background-color: #eef3f8;
            }

            QPushButton#JogMinusButton:hover,
            QPushButton#JogPlusButton:hover {
                background-color: #dce9f6;
            }

            QPushButton#JogMinusButton:pressed,
            QPushButton#JogPlusButton:pressed {
                background-color: #b8d7f3;
                border: 2px solid #5c9ed6;
            }

            QLabel#JogAxisLabel {
                font-size: 18px;
                font-weight: bold;
                border: 1px solid #c6cbd1;
                border-radius: 8px;
                background-color: #f7f9fc;
            }
            """
        )

    # ============================================================
    # Timer
    # ============================================================

    def setup_timers(self) -> None:
        self.ros_timer = QTimer(self)
        self.ros_timer.timeout.connect(self.spin_ros_once)
        self.ros_timer.start(5)

        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.refresh_ui)
        self.ui_timer.start(100)

        self.hold_timer = QTimer(self)
        self.hold_timer.timeout.connect(self.publish_hold_target)
        self.update_hold_timer()

        self.jog_timer = QTimer(self)
        self.jog_timer.timeout.connect(self.on_jog_timer_tick)
        self.update_jog_timer_interval()

    def update_hold_timer(self) -> None:
        rate = max(1.0, float(self.hold_rate_spin.value()))
        interval_ms = max(2, int(1000.0 / rate))
        self.hold_timer.start(interval_ms)

    def update_jog_timer_interval(self) -> None:
        rate = max(1.0, float(self.jog_rate_spin.value()))
        interval_ms = max(5, int(1000.0 / rate))
        self.jog_timer.setInterval(interval_ms)

        if hasattr(self, "twist_controller"):
            self.twist_controller.dt = interval_ms / 1000.0

    def spin_ros_once(self) -> None:
        err = spin_once_safe(self.ros_node)
        if err:
            self.status_label.setText(err)

    # ============================================================
    # 当前状态读取
    # ============================================================

    def get_current_q_dq(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if hasattr(self.state_buffer, "get_q_dq_arrays"):
            result = self.state_buffer.get_q_dq_arrays()
            if result is None:
                return None
            q, dq = result
            return np.asarray(q, dtype=float), np.asarray(dq, dtype=float)

        if hasattr(self.state_buffer, "get_q_dq"):
            result = self.state_buffer.get_q_dq()
            if result is None:
                return None
            q, dq = result
            return np.asarray(q, dtype=float), np.asarray(dq, dtype=float)

        if hasattr(self.state_buffer, "q") and hasattr(self.state_buffer, "dq"):
            q = np.asarray(self.state_buffer.q, dtype=float)
            dq = np.asarray(self.state_buffer.dq, dtype=float)
            if q.shape[0] == 7:
                return q, dq

        self.status_label.setText(
            "RobotStateBuffer does not provide get_q_dq_arrays() / get_q_dq() / q,dq."
        )
        return None

    def get_current_pose6(self) -> Optional[np.ndarray]:
        result = self.get_current_q_dq()
        if result is None:
            return None

        q, _ = result
        return self.pose_ik_controller.kin.fk_pose6(q)

    # ============================================================
    # 全局按钮回调
    # ============================================================

    def on_read_current_joints(self) -> None:
        result = self.get_current_q_dq()
        if result is None:
            self.status_label.setText("Cannot read current joints: no valid /joint_states.")
            return

        q, _ = result
        self.status_label.setText(
            "Current q = [" + ", ".join([f"{v:.4f}" for v in q]) + "]"
        )
        self.refresh_joint_labels(q)

    def on_fill_target_from_current_pose(self) -> None:
        pose6 = self.get_current_pose6()
        if pose6 is None:
            self.status_label.setText("Cannot fill target: no valid current pose.")
            return

        self.set_pose_spins(pose6)
        self.target_pose6 = pose6.copy()

        self.status_label.setText("Filled target pose from current end-effector pose.")

    def on_move_to_target_pose(self) -> None:
        result = self.get_current_q_dq()
        if result is None:
            self.status_label.setText("Cannot move: no valid current joint state.")
            return

        q_current, _ = result

        target_pose6 = self.get_pose_from_spins()
        self.target_pose6 = target_pose6.copy()

        self.pose_ik_controller.set_target_pose(target_pose6)
        ik_result = self.pose_ik_controller.solve(q_current)

        if ik_result.q_goal is not None:
            self.q_goal = ik_result.q_goal.copy()

        self.ik_result_label.setText(
            f"IK success={ik_result.success}, "
            f"iter={ik_result.iterations}, "
            f"err={ik_result.final_error_norm:.6f}, "
            f"message={ik_result.message}"
        )

        if not ik_result.success:
            self.status_label.setText(
                "IK did not fully converge. Approximate q_goal was saved if available."
            )
            return

        self.ros_node.publish_position(ik_result.q_goal.tolist())
        self.status_label.setText("Published q_goal position command.")

    def on_publish_last_q_goal(self) -> None:
        if self.q_goal is None:
            self.status_label.setText("No q_goal available. Please solve IK first.")
            return

        self.ros_node.publish_position(self.q_goal.tolist())
        self.status_label.setText("Published last q_goal again.")

    def on_control_mode_changed(self, mode: str) -> None:
        self.stop_jog()
        self.status_label.setText(f"Control mode changed to: {mode}")

    # ============================================================
    # Jog 控制入口
    # ============================================================

    def start_jog(self, axis: str, direction: float) -> None:
        self.active_jog_axis = axis
        self.active_jog_direction = float(direction)

        # 按下瞬间先执行一次，避免手感延迟
        self.on_jog_timer_tick()

        if not self.jog_timer.isActive():
            self.jog_timer.start()

    def stop_jog(self) -> None:
        self.active_jog_axis = None
        self.active_jog_direction = 0.0

        if self.jog_timer.isActive():
            self.jog_timer.stop()

    def on_jog_timer_tick(self) -> None:
        if self.active_jog_axis is None:
            return

        if abs(self.active_jog_direction) < 1e-12:
            return

        mode = self.control_mode_box.currentText()

        if mode == self.MODE_TWIST:
            self._do_twist_jog(self.active_jog_axis, self.active_jog_direction)
        else:
            self._do_pose_ik_jog(self.active_jog_axis, self.active_jog_direction)

    # ============================================================
    # Pose IK Jog
    # ============================================================

    def _do_pose_ik_jog(self, axis: str, direction: float) -> None:
        current_target = self.get_pose_from_spins()

        if axis == "x":
            current_target[0] += direction * float(self.pos_step_spin.value())
        elif axis == "y":
            current_target[1] += direction * float(self.pos_step_spin.value())
        elif axis == "z":
            current_target[2] += direction * float(self.pos_step_spin.value())
        elif axis == "rx":
            current_target[3] += direction * float(self.rot_step_spin.value())
        elif axis == "ry":
            current_target[4] += direction * float(self.rot_step_spin.value())
        elif axis == "rz":
            current_target[5] += direction * float(self.rot_step_spin.value())
        else:
            self.status_label.setText(f"Unknown pose jog axis: {axis}")
            return

        current_target[3:6] = self.pose_ik_controller.kin.normalize_angles(
            current_target[3:6]
        )

        self.set_pose_spins(current_target)
        self.target_pose6 = current_target.copy()

        self.on_move_to_target_pose()

    # ============================================================
    # Twist Jog
    # ============================================================

    # ============================================================
    # Twist Jog
    # ============================================================

    def _do_twist_jog(self, axis: str, direction: float) -> None:
        result = self.get_current_q_dq()
        if result is None:
            self.status_label.setText("Cannot twist jog: no valid current joint state.")
            return

        q_current, _ = result

        if self.twist_q_target is None:
            self.twist_q_target = q_current.copy()

        tracking_err = float(np.linalg.norm(self.twist_q_target - q_current))

        if tracking_err > 0.5:
            self.twist_q_target = q_current.copy()
            tracking_err = 0.0

        twist = self._make_twist(axis, direction)
        if twist is None:
            self.status_label.setText(f"Unknown twist jog axis: {axis}")
            return

        twist_result = self.twist_controller.solve(
            q_current=q_current,
            twist=twist,
            q_integrator=self.twist_q_target,
        )

        if not twist_result.success:
            self.ik_result_label.setText(f"Twist failed: {twist_result.message}")
            self.status_label.setText(twist_result.message)
            return

        if twist_result.q_target is None:
            self.status_label.setText("Twist result has no q_target.")
            return

        self.twist_q_target = twist_result.q_target.copy()
        self.q_goal = self.twist_q_target.copy()

        self.ros_node.publish_position(self.q_goal.tolist())

        twist_text = ", ".join([f"{v:.3f}" for v in twist])

        if twist_result.dq_cmd is not None:
            dq_text = ", ".join([f"{v:.3f}" for v in twist_result.dq_cmd])
        else:
            dq_text = "-"

        if twist_result.achieved_twist is not None:
            achieved_text = ", ".join([f"{v:.3f}" for v in twist_result.achieved_twist])
        else:
            achieved_text = "-"

        self.ik_result_label.setText(
            f"Twist success=True, "
            f"desired=[{twist_text}], "
            f"achieved=[{achieved_text}], "
            f"dq=[{dq_text}], "
            f"tracking_err={tracking_err:.4f}"
        )

        self.status_label.setText(
            f"Twist Jog {axis} {'+' if direction > 0 else '-'} published q_target."
        )


    def _make_twist(self, axis: str, direction: float) -> Optional[np.ndarray]:
        """
        根据按钮轴向生成末端笛卡尔速度 twist。

        twist = [vx, vy, vz, wx, wy, wz]
        """

        twist = np.zeros(6, dtype=float)

        linear_speed = float(self.linear_speed_spin.value())
        angular_speed = float(self.angular_speed_spin.value())

        if axis == "x":
            twist[0] = direction * linear_speed
        elif axis == "y":
            twist[1] = direction * linear_speed
        elif axis == "z":
            twist[2] = direction * linear_speed
        elif axis == "rx":
            twist[3] = direction * angular_speed
        elif axis == "ry":
            twist[4] = direction * angular_speed
        elif axis == "rz":
            twist[5] = direction * angular_speed
        else:
            return None

        return twist

    # ============================================================
    # Hold
    # ============================================================

    def publish_hold_target(self) -> None:
        if not self.hold_position_box.isChecked():
            return

        if self.q_goal is None:
            return

        self.ros_node.publish_position(self.q_goal.tolist())

    # ============================================================
    # UI 数据读写
    # ============================================================

    def get_pose_from_spins(self) -> np.ndarray:
        return np.array(
            [
                self.pose_spins["x"].value(),
                self.pose_spins["y"].value(),
                self.pose_spins["z"].value(),
                self.pose_spins["rx"].value(),
                self.pose_spins["ry"].value(),
                self.pose_spins["rz"].value(),
            ],
            dtype=float,
        )

    def set_pose_spins(self, pose6: np.ndarray) -> None:
        pose6 = np.asarray(pose6, dtype=float).reshape(6)

        self.pose_spins["x"].setValue(float(pose6[0]))
        self.pose_spins["y"].setValue(float(pose6[1]))
        self.pose_spins["z"].setValue(float(pose6[2]))
        self.pose_spins["rx"].setValue(float(pose6[3]))
        self.pose_spins["ry"].setValue(float(pose6[4]))
        self.pose_spins["rz"].setValue(float(pose6[5]))

    def refresh_ui(self) -> None:
        result = self.get_current_q_dq()
        if result is None:
            return

        q, _ = result

        self.refresh_joint_labels(q)

        pose6 = self.pose_ik_controller.kin.fk_pose6(q)
        self.refresh_pose_labels(pose6)

    def refresh_joint_labels(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=float).reshape(7)

        for i, name in enumerate(self.joint_names):
            self.q_current_labels[name].setText(f"{q[i]:.4f}")

            if self.q_goal is not None:
                self.q_goal_labels[name].setText(f"{self.q_goal[i]:.4f}")
                self.q_error_labels[name].setText(f"{self.q_goal[i] - q[i]:.4f}")
            else:
                self.q_goal_labels[name].setText("-")
                self.q_error_labels[name].setText("-")

    def refresh_pose_labels(self, pose6: np.ndarray) -> None:
        pose6 = np.asarray(pose6, dtype=float).reshape(6)

        names = ["x", "y", "z", "rx", "ry", "rz"]

        for i, name in enumerate(names):
            self.current_pose_labels[name].setText(f"{pose6[i]:.5f}")

    # ============================================================
    # QLabel helper
    # ============================================================

    @staticmethod
    def _center_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    @staticmethod
    def _header_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-weight: bold;")
        return label

    # ============================================================
    # 关闭窗口
    # ============================================================

    def closeEvent(self, event) -> None:
        try:
            self.stop_jog()
            self.ros_node.publish_zero_effort()
        except Exception:
            pass

        event.accept()