#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ros2_topic_monitor_ui.py

独立 ROS2 Topic 实时监控 UI。

功能：
1. 自动扫描 ROS2 当前所有 topic；
2. 动态订阅所有 topic；
3. 实时显示 topic 名称、类型、接收次数、频率、最后更新时间；
4. 点击某个 topic，可查看最新消息内容；
5. 支持暂停、清空、过滤、手动刷新。

运行前：
    source /opt/ros/humble/setup.bash
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

运行：
    python3 ros2_topic_monitor_ui.py
"""

import sys
import time
import json
from collections import deque
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosidl_runtime_py.utilities import get_message

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ============================================================
# ROS Message 安全摘要工具
# ============================================================

def summarize_ros_value(value: Any, depth: int = 0, max_depth: int = 2) -> Any:
    """
    将 ROS message 字段转换成安全可显示的摘要。

    重点：
    - 不直接展开 image.data / pointcloud.data 这种超大数组；
    - 避免 UI 卡死；
    - 保留主要字段结构。
    """

    if depth > max_depth:
        return "<max depth reached>"

    if value is None:
        return None

    if isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"

    if isinstance(value, bytearray):
        return f"<bytearray len={len(value)}>"

    if isinstance(value, (list, tuple)):
        n = len(value)

        if n == 0:
            return []

        # 大数组不要完整展开
        if n > 20:
            head = []
            for item in list(value[:5]):
                head.append(summarize_ros_value(item, depth + 1, max_depth))
            return {
                "__sequence__": True,
                "len": n,
                "head": head,
                "note": "large sequence truncated",
            }

        return [
            summarize_ros_value(item, depth + 1, max_depth)
            for item in value
        ]

    # ROS message 对象通常有 get_fields_and_field_types()
    if hasattr(value, "get_fields_and_field_types"):
        result = {}
        fields = value.get_fields_and_field_types()

        for field_name in fields.keys():
            try:
                field_value = getattr(value, field_name)
                result[field_name] = summarize_ros_value(
                    field_value,
                    depth + 1,
                    max_depth,
                )
            except Exception as e:
                result[field_name] = f"<read field failed: {e}>"

        return result

    return str(value)


def summarize_ros_message(msg: Any) -> str:
    """
    把 ROS message 转成 JSON 字符串。
    """
    try:
        data = summarize_ros_value(msg, depth=0, max_depth=3)
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"<message summarize failed: {e}>\n\nraw:\n{msg}"


def make_preview(text: str, max_len: int = 160) -> str:
    """
    表格里显示用的短预览。
    """
    text = text.replace("\n", " ").replace("\r", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len] + " ..."


# ============================================================
# ROS2 Topic Monitor Node
# ============================================================

class TopicMonitorNode(Node):
    """
    ROS2 topic 监控节点。

    动态订阅当前系统里的 topic。
    每个 topic 保存：
        - type
        - count
        - hz
        - last_time
        - last_msg_text
    """

    def __init__(self):
        super().__init__("ros2_topic_monitor_ui")

        self.topic_types: Dict[str, str] = {}
        self.subscriptions_by_topic: Dict[str, Any] = {}

        self.counts: Dict[str, int] = {}
        self.last_receive_time: Dict[str, float] = {}
        self.time_history: Dict[str, deque] = {}
        self.latest_msg_text: Dict[str, str] = {}
        self.latest_preview: Dict[str, str] = {}

        self.subscribe_all: bool = True
        self.topic_filter: str = ""

        self.qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

    def refresh_topic_list(self) -> None:
        """
        扫描当前 ROS2 topic。
        """
        topic_names_and_types = self.get_topic_names_and_types()

        new_topic_types = {}

        for topic_name, type_names in topic_names_and_types:
            if not type_names:
                continue

            # 同一个 topic 理论上可能有多个 type，这里取第一个
            type_name = type_names[0]
            new_topic_types[topic_name] = type_name

        self.topic_types = new_topic_types

    def update_subscriptions(self) -> None:
        """
        根据当前 topic 列表动态创建订阅。
        """

        if not self.subscribe_all:
            return

        for topic_name, type_name in self.topic_types.items():
            if topic_name in self.subscriptions_by_topic:
                continue

            if self.topic_filter and self.topic_filter not in topic_name:
                continue

            try:
                msg_type = get_message(type_name)
            except Exception as e:
                self.get_logger().warning(
                    f"Cannot import message type {type_name} for {topic_name}: {e}"
                )
                continue

            try:
                callback = self._make_callback(topic_name)
                sub = self.create_subscription(
                    msg_type,
                    topic_name,
                    callback,
                    self.qos,
                )

                self.subscriptions_by_topic[topic_name] = sub
                self.counts.setdefault(topic_name, 0)
                self.time_history.setdefault(topic_name, deque(maxlen=50))
                self.latest_msg_text.setdefault(topic_name, "")
                self.latest_preview.setdefault(topic_name, "")

                self.get_logger().info(
                    f"Subscribed: {topic_name} [{type_name}]"
                )

            except Exception as e:
                self.get_logger().warning(
                    f"Failed to subscribe {topic_name} [{type_name}]: {e}"
                )

    def _make_callback(self, topic_name: str):
        def callback(msg):
            now = time.time()

            self.counts[topic_name] = self.counts.get(topic_name, 0) + 1
            self.last_receive_time[topic_name] = now

            if topic_name not in self.time_history:
                self.time_history[topic_name] = deque(maxlen=50)

            self.time_history[topic_name].append(now)

            text = summarize_ros_message(msg)
            self.latest_msg_text[topic_name] = text
            self.latest_preview[topic_name] = make_preview(text)

        return callback

    def get_hz(self, topic_name: str) -> float:
        """
        根据最近接收时间估算频率。
        """
        hist = self.time_history.get(topic_name)

        if not hist or len(hist) < 2:
            return 0.0

        duration = hist[-1] - hist[0]
        if duration <= 1e-9:
            return 0.0

        return (len(hist) - 1) / duration

    def clear_data(self) -> None:
        self.counts.clear()
        self.last_receive_time.clear()
        self.time_history.clear()
        self.latest_msg_text.clear()
        self.latest_preview.clear()

        for topic_name in self.topic_types.keys():
            self.counts[topic_name] = 0
            self.time_history[topic_name] = deque(maxlen=50)
            self.latest_msg_text[topic_name] = ""
            self.latest_preview[topic_name] = ""


# ============================================================
# PyQt6 UI
# ============================================================

class Ros2TopicMonitorWindow(QMainWindow):
    def __init__(self, node: TopicMonitorNode):
        super().__init__()

        self.node = node
        self.selected_topic: Optional[str] = None

        self.setWindowTitle("ROS2 Topic Monitor")
        self.resize(1400, 850)

        self.build_ui()
        self.setup_timers()

    def build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # ------------------------------------------------------------
        # 顶部控制栏
        # ------------------------------------------------------------
        control_layout = QHBoxLayout()

        self.refresh_button = QPushButton("Refresh Topics")
        self.clear_button = QPushButton("Clear Data")
        self.pause_box = QCheckBox("Pause UI")
        self.subscribe_all_box = QCheckBox("Subscribe All")
        self.subscribe_all_box.setChecked(True)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter topic name, e.g. joint / tf / camera")

        self.topic_count_label = QLabel("Topics: 0")
        self.sub_count_label = QLabel("Subscribed: 0")

        control_layout.addWidget(self.refresh_button)
        control_layout.addWidget(self.clear_button)
        control_layout.addWidget(self.subscribe_all_box)
        control_layout.addWidget(self.pause_box)
        control_layout.addWidget(QLabel("Filter:"))
        control_layout.addWidget(self.filter_edit)
        control_layout.addWidget(self.topic_count_label)
        control_layout.addWidget(self.sub_count_label)

        main_layout.addLayout(control_layout)

        # ------------------------------------------------------------
        # 中间：topic 表格 + 消息详情
        # ------------------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Vertical)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            [
                "Topic",
                "Type",
                "Count",
                "Hz",
                "Last Update",
                "Preview",
            ]
        )

        self.table.setColumnWidth(0, 320)
        self.table.setColumnWidth(1, 300)
        self.table.setColumnWidth(2, 80)
        self.table.setColumnWidth(3, 80)
        self.table.setColumnWidth(4, 130)
        self.table.setColumnWidth(5, 600)

        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setPlaceholderText("Select a topic to see latest message data.")

        splitter.addWidget(self.table)
        splitter.addWidget(self.detail_text)
        splitter.setSizes([520, 300])

        main_layout.addWidget(splitter)

        # ------------------------------------------------------------
        # 信号连接
        # ------------------------------------------------------------
        self.refresh_button.clicked.connect(self.on_refresh_topics)
        self.clear_button.clicked.connect(self.on_clear_data)
        self.subscribe_all_box.stateChanged.connect(self.on_subscribe_all_changed)
        self.filter_edit.textChanged.connect(self.on_filter_changed)
        self.table.itemSelectionChanged.connect(self.on_table_selection_changed)

        self.apply_style()

    def setup_timers(self) -> None:
        # ROS spin 定时器
        self.ros_timer = QTimer(self)
        self.ros_timer.timeout.connect(self.spin_ros_once)
        self.ros_timer.start(5)

        # topic 列表刷新定时器
        self.topic_timer = QTimer(self)
        self.topic_timer.timeout.connect(self.periodic_topic_refresh)
        self.topic_timer.start(1000)

        # UI 刷新定时器
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.refresh_ui)
        self.ui_timer.start(200)

    def spin_ros_once(self) -> None:
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception as e:
            self.detail_text.setPlainText(f"ROS spin error:\n{e}")

    def periodic_topic_refresh(self) -> None:
        self.node.refresh_topic_list()
        self.node.update_subscriptions()

    def on_refresh_topics(self) -> None:
        self.node.refresh_topic_list()
        self.node.update_subscriptions()
        self.refresh_ui(force=True)

    def on_clear_data(self) -> None:
        self.node.clear_data()
        self.detail_text.clear()
        self.refresh_ui(force=True)

    def on_subscribe_all_changed(self) -> None:
        self.node.subscribe_all = self.subscribe_all_box.isChecked()
        self.node.update_subscriptions()

    def on_filter_changed(self, text: str) -> None:
        self.node.topic_filter = text.strip()
        self.refresh_ui(force=True)

    def on_table_selection_changed(self) -> None:
        items = self.table.selectedItems()
        if not items:
            self.selected_topic = None
            return

        row = items[0].row()
        topic_item = self.table.item(row, 0)

        if topic_item is None:
            self.selected_topic = None
            return

        self.selected_topic = topic_item.text()
        self.update_detail_text()

    def refresh_ui(self, force: bool = False) -> None:
        if self.pause_box.isChecked() and not force:
            return

        topic_items = []

        topic_filter = self.filter_edit.text().strip()

        for topic_name, type_name in sorted(self.node.topic_types.items()):
            if topic_filter and topic_filter not in topic_name:
                continue

            count = self.node.counts.get(topic_name, 0)
            hz = self.node.get_hz(topic_name)
            last_t = self.node.last_receive_time.get(topic_name)
            preview = self.node.latest_preview.get(topic_name, "")

            if last_t is None:
                last_update = "-"
            else:
                last_update = time.strftime("%H:%M:%S", time.localtime(last_t))

            topic_items.append(
                (
                    topic_name,
                    type_name,
                    count,
                    hz,
                    last_update,
                    preview,
                )
            )

        self.table.setRowCount(len(topic_items))

        for row, item in enumerate(topic_items):
            topic_name, type_name, count, hz, last_update, preview = item

            self.table.setItem(row, 0, QTableWidgetItem(topic_name))
            self.table.setItem(row, 1, QTableWidgetItem(type_name))
            self.table.setItem(row, 2, QTableWidgetItem(str(count)))
            self.table.setItem(row, 3, QTableWidgetItem(f"{hz:.2f}"))
            self.table.setItem(row, 4, QTableWidgetItem(last_update))
            self.table.setItem(row, 5, QTableWidgetItem(preview))

        self.topic_count_label.setText(f"Topics: {len(self.node.topic_types)}")
        self.sub_count_label.setText(
            f"Subscribed: {len(self.node.subscriptions_by_topic)}"
        )

        self.update_detail_text()

    def update_detail_text(self) -> None:
        if not self.selected_topic:
            return

        topic_name = self.selected_topic
        type_name = self.node.topic_types.get(topic_name, "-")
        count = self.node.counts.get(topic_name, 0)
        hz = self.node.get_hz(topic_name)
        msg_text = self.node.latest_msg_text.get(topic_name, "")

        header = (
            f"Topic: {topic_name}\n"
            f"Type : {type_name}\n"
            f"Count: {count}\n"
            f"Hz   : {hz:.2f}\n"
            f"{'-' * 80}\n"
        )

        if not msg_text:
            msg_text = "<no message received yet>"

        self.detail_text.setPlainText(header + msg_text)

    def apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f4f6f8;
            }

            QLabel {
                font-size: 13px;
            }

            QPushButton {
                min-height: 30px;
                padding: 4px 12px;
                border-radius: 6px;
                background-color: #e8edf3;
            }

            QPushButton:hover {
                background-color: #dce6f2;
            }

            QLineEdit {
                min-height: 30px;
                font-size: 13px;
                padding: 2px 8px;
                border: 1px solid #c6cbd1;
                border-radius: 6px;
                background-color: white;
            }

            QTableWidget {
                background-color: white;
                gridline-color: #d8dde4;
                font-size: 12px;
            }

            QHeaderView::section {
                background-color: #eef3f8;
                padding: 4px;
                border: 1px solid #d8dde4;
                font-weight: bold;
            }

            QTextEdit {
                background-color: white;
                font-family: Consolas, Menlo, monospace;
                font-size: 12px;
                border: 1px solid #c6cbd1;
                border-radius: 6px;
            }

            QCheckBox {
                font-size: 13px;
            }
            """
        )


# ============================================================
# main
# ============================================================

def main() -> None:
    rclpy.init()

    node = TopicMonitorNode()

    app = QApplication(sys.argv)
    window = Ros2TopicMonitorWindow(node)
    window.show()

    exit_code = 0

    try:
        exit_code = app.exec()

    finally:
        try:
            node.destroy_node()
        except Exception:
            pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    sys.exit(exit_code)


if __name__ == "__main__":
    main()