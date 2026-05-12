import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class UrdfJoint:
    """
    URDF 关节信息。

    这里主要用于运动学计算，不处理惯量、碰撞、视觉 mesh。
    """

    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: Optional[float] = None
    upper: Optional[float] = None


class PandaKinematics:
    """
    基于 Isaac Sim FR3 URDF 的运动学模型。

    注意：
    这里虽然类名仍然叫 PandaKinematics，
    是为了兼容你原来工程里已有的 import：

        from core.kinematics import PandaKinematics

    但内部已经不再使用硬编码 DH 参数，
    而是根据 Isaac Sim 中导出的 FR3 URDF 进行链式 FK 计算。

    默认控制链路：

        fr3_link0
            ↓
        fr3_joint1 ~ fr3_joint7
            ↓
        fr3_link7
            ↓
        fr3_joint8 fixed
            ↓
        fr3_link8
            ↓
        fr3_hand_joint fixed
            ↓
        fr3_hand
            ↓
        fr3_hand_tcp_joint fixed
            ↓
        fr3_hand_tcp

    默认末端 frame：

        fr3_hand_tcp

    返回 pose6 格式：

        [x, y, z, rx, ry, rz]

    单位：
        x/y/z: m
        rx/ry/rz: rad
    """

    DEFAULT_ACTIVE_JOINT_NAMES = [
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
    ]

    DEFAULT_FIXED_JOINT_VALUES = {
        "fr3_finger_joint1": 0.025,
        "fr3_finger_joint2": 0.025,
    }

    def __init__(
        self,
        urdf_path: str = "./config/fr3.urdf",
        robot_description_yaml_path: Optional[str] = None,
        base_link: str = "fr3_link0",
        end_link: str = "fr3_hand_tcp",
        active_joint_names: Optional[List[str]] = None,
    ):
        """
        初始化 FR3 URDF 运动学模型。

        参数：
            urdf_path:
                你的 Isaac Sim / Franka 导出的 fr3.urdf 路径。

            robot_description_yaml_path:
                可选。Lula 的 robot_description.yaml 路径。
                如果提供，会尝试从里面读取 cspace 和 fixed joint rule。
                如果不提供，则默认使用 fr3_joint1 ~ fr3_joint7。

            base_link:
                基坐标系 link。默认 fr3_link0。

            end_link:
                末端 link。默认 fr3_hand_tcp。
                也可以改成：
                    - fr3_link8
                    - fr3_hand
                    - gripper_center

            active_joint_names:
                主动关节顺序。
                如果不传，默认使用 fr3_joint1 ~ fr3_joint7。
        """

        self.urdf_path = urdf_path
        self.robot_description_yaml_path = robot_description_yaml_path

        self.base_link = base_link
        self.end_link = end_link

        self.active_joint_names = (
            list(active_joint_names)
            if active_joint_names is not None
            else self.DEFAULT_ACTIVE_JOINT_NAMES.copy()
        )

        self.fixed_joint_values = self.DEFAULT_FIXED_JOINT_VALUES.copy()

        if robot_description_yaml_path is not None:
            self._load_robot_description_yaml(robot_description_yaml_path)

        self.joints_by_name: Dict[str, UrdfJoint] = {}
        self.child_to_joint: Dict[str, UrdfJoint] = {}
        self.parent_to_joints: Dict[str, List[UrdfJoint]] = {}

        self._load_urdf(urdf_path)

        self.chain_joints: List[UrdfJoint] = self._find_chain(
            base_link=self.base_link,
            end_link=self.end_link,
        )

        self._validate_active_joints_in_chain()

    # ============================================================
    # 文件加载
    # ============================================================

    def _load_robot_description_yaml(self, yaml_path: str) -> None:
        """
        轻量读取 robot_description.yaml。

        不依赖 PyYAML，只解析当前需要的：
        1. cspace
        2. fixed joint rules

        如果你的 YAML 格式变化较大，建议改用 yaml.safe_load。
        """

        with open(yaml_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 解析 cspace:
        # cspace:
        #     - fr3_joint1
        #     - fr3_joint2
        cspace_match = re.search(
            r"cspace:\s*(.*?)\n\S",
            text + "\nEND:",
            flags=re.DOTALL,
        )

        if cspace_match:
            block = cspace_match.group(1)
            names = re.findall(r"-\s*([A-Za-z0-9_]+)", block)
            if names:
                self.active_joint_names = names

        # 解析 fixed rules:
        # - {name: fr3_finger_joint1, rule: fixed, value: 0.025}
        fixed_rules = re.findall(
            r"\{\s*name:\s*([A-Za-z0-9_]+)\s*,\s*rule:\s*fixed\s*,\s*value:\s*([-+0-9.eE]+)\s*\}",
            text,
        )

        for name, value in fixed_rules:
            self.fixed_joint_values[name] = float(value)

    def _load_urdf(self, urdf_path: str) -> None:
        """
        读取 URDF 文件，并解析所有 joint。
        """

        tree = ET.parse(urdf_path)
        root = tree.getroot()

        for joint_elem in root.findall("joint"):
            joint = self._parse_joint(joint_elem)

            self.joints_by_name[joint.name] = joint
            self.child_to_joint[joint.child] = joint

            if joint.parent not in self.parent_to_joints:
                self.parent_to_joints[joint.parent] = []
            self.parent_to_joints[joint.parent].append(joint)

    def _parse_joint(self, joint_elem: ET.Element) -> UrdfJoint:
        """
        解析单个 URDF joint。
        """

        name = joint_elem.attrib["name"]
        joint_type = joint_elem.attrib.get("type", "fixed")

        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")

        if parent_elem is None or child_elem is None:
            raise ValueError(f"Joint {name} missing parent or child.")

        parent = parent_elem.attrib["link"]
        child = child_elem.attrib["link"]

        origin_elem = joint_elem.find("origin")

        if origin_elem is not None:
            xyz = self._parse_vec3(origin_elem.attrib.get("xyz", "0 0 0"))
            rpy = self._parse_vec3(origin_elem.attrib.get("rpy", "0 0 0"))
        else:
            xyz = np.zeros(3, dtype=float)
            rpy = np.zeros(3, dtype=float)

        axis_elem = joint_elem.find("axis")
        if axis_elem is not None:
            axis = self._parse_vec3(axis_elem.attrib.get("xyz", "0 0 1"))
        else:
            axis = np.array([0.0, 0.0, 1.0], dtype=float)

        norm = np.linalg.norm(axis)
        if norm > 1e-12:
            axis = axis / norm

        lower = None
        upper = None

        limit_elem = joint_elem.find("limit")
        if limit_elem is not None:
            if "lower" in limit_elem.attrib:
                lower = float(limit_elem.attrib["lower"])
            if "upper" in limit_elem.attrib:
                upper = float(limit_elem.attrib["upper"])

        return UrdfJoint(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child,
            xyz=xyz,
            rpy=rpy,
            axis=axis,
            lower=lower,
            upper=upper,
        )

    @staticmethod
    def _parse_vec3(text: str) -> np.ndarray:
        values = [float(v) for v in text.strip().split()]
        if len(values) != 3:
            raise ValueError(f"Expected 3 values, got: {text}")
        return np.array(values, dtype=float)

    # ============================================================
    # 链路查找
    # ============================================================

    def _find_chain(self, base_link: str, end_link: str) -> List[UrdfJoint]:
        """
        从 URDF 里查找 base_link 到 end_link 的 joint 链。

        返回：
            [joint1, joint2, ..., jointN]
        """

        path: List[UrdfJoint] = []

        visited = set()

        def dfs(current_link: str) -> bool:
            if current_link == end_link:
                return True

            if current_link in visited:
                return False

            visited.add(current_link)

            for joint in self.parent_to_joints.get(current_link, []):
                path.append(joint)

                if dfs(joint.child):
                    return True

                path.pop()

            return False

        found = dfs(base_link)

        if not found:
            raise ValueError(
                f"Cannot find kinematic chain from {base_link} to {end_link}."
            )

        return path.copy()

    def _validate_active_joints_in_chain(self) -> None:
        """
        检查 active_joint_names 是否都在 URDF 里。
        """

        for name in self.active_joint_names:
            if name not in self.joints_by_name:
                raise ValueError(f"Active joint {name} not found in URDF.")

    # ============================================================
    # 基础数学工具
    # ============================================================

    def _check_q(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)

        expected = len(self.active_joint_names)
        if q.shape[0] != expected:
            raise ValueError(
                f"Expected q with {expected} elements for {self.active_joint_names}, "
                f"got shape {q.shape}"
            )

        return q

    def _q_to_map(self, q: np.ndarray) -> Dict[str, float]:
        """
        将 q 数组转换成 {joint_name: value}。
        """

        q = self._check_q(q)
        return {
            name: float(q[i])
            for i, name in enumerate(self.active_joint_names)
        }

    def _get_joint_value(self, joint: UrdfJoint, q_map: Dict[str, float]) -> float:
        """
        获取某个 joint 的当前值。

        主动关节：
            使用 q_map 里的值。

        fixed joint：
            0。

        其他非主动关节：
            如果 fixed_joint_values 中有配置，则使用配置值。
            否则默认 0。
        """

        if joint.name in q_map:
            return q_map[joint.name]

        if joint.name in self.fixed_joint_values:
            return float(self.fixed_joint_values[joint.name])

        return 0.0

    @staticmethod
    def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
        T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
        return T

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def normalize_angles(angles: np.ndarray) -> np.ndarray:
        angles = np.asarray(angles, dtype=float)
        return (angles + math.pi) % (2.0 * math.pi) - math.pi

    # ============================================================
    # 旋转相关
    # ============================================================

    @staticmethod
    def rpy_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
        """
        URDF origin rpy 使用 roll/pitch/yaw。

        这里采用：

            R = Rz(rz) @ Ry(ry) @ Rx(rx)
        """

        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)

        Rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cx, -sx],
                [0.0, sx, cx],
            ],
            dtype=float,
        )

        Ry = np.array(
            [
                [cy, 0.0, sy],
                [0.0, 1.0, 0.0],
                [-sy, 0.0, cy],
            ],
            dtype=float,
        )

        Rz = np.array(
            [
                [cz, -sz, 0.0],
                [sz, cz, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        return Rz @ Ry @ Rx

    @staticmethod
    def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=float).reshape(3, 3)

        sy = float(np.clip(-R[2, 0], -1.0, 1.0))
        ry = math.asin(sy)
        cy = math.cos(ry)

        if abs(cy) > 1e-8:
            rx = math.atan2(R[2, 1], R[2, 2])
            rz = math.atan2(R[1, 0], R[0, 0])
        else:
            rx = math.atan2(-R[1, 2], R[1, 1])
            rz = 0.0

        return np.array([rx, ry, rz], dtype=float)

    @staticmethod
    def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
        """
        任意轴旋转矩阵 Rodrigues 公式。
        """

        axis = np.asarray(axis, dtype=float).reshape(3)
        norm = np.linalg.norm(axis)

        if norm < 1e-12:
            return np.eye(3, dtype=float)

        axis = axis / norm
        x, y, z = axis

        K = np.array(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ],
            dtype=float,
        )

        R = (
            np.eye(3, dtype=float)
            + math.sin(angle) * K
            + (1.0 - math.cos(angle)) * (K @ K)
        )

        return R

    @staticmethod
    def rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=float).reshape(3, 3)

        trace = float(np.trace(R))
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))

        theta = math.acos(cos_theta)

        if theta < 1e-9:
            return np.array(
                [
                    0.5 * (R[2, 1] - R[1, 2]),
                    0.5 * (R[0, 2] - R[2, 0]),
                    0.5 * (R[1, 0] - R[0, 1]),
                ],
                dtype=float,
            )

        if abs(math.pi - theta) < 1e-6:
            axis = np.empty(3, dtype=float)
            axis[0] = math.sqrt(max(0.0, (R[0, 0] + 1.0) / 2.0))
            axis[1] = math.sqrt(max(0.0, (R[1, 1] + 1.0) / 2.0))
            axis[2] = math.sqrt(max(0.0, (R[2, 2] + 1.0) / 2.0))

            if R[0, 1] < 0.0:
                axis[1] = -axis[1]
            if R[0, 2] < 0.0:
                axis[2] = -axis[2]

            norm_axis = np.linalg.norm(axis)
            if norm_axis < 1e-9:
                return np.zeros(3, dtype=float)

            return theta * axis / norm_axis

        axis = np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ],
            dtype=float,
        ) / (2.0 * math.sin(theta))

        return theta * axis

    @staticmethod
    def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
        rotvec = np.asarray(rotvec, dtype=float).reshape(3)
        theta = float(np.linalg.norm(rotvec))

        if theta < 1e-12:
            return np.eye(3, dtype=float)

        axis = rotvec / theta
        return PandaKinematics.axis_angle_to_matrix(axis, theta)

    # ============================================================
    # URDF joint transform
    # ============================================================

    def _joint_origin_transform(self, joint: UrdfJoint) -> np.ndarray:
        """
        URDF joint origin transform:

            T_origin = Trans(xyz) * Rot(rpy)
        """

        R = self.rpy_to_matrix(
            float(joint.rpy[0]),
            float(joint.rpy[1]),
            float(joint.rpy[2]),
        )

        return self.make_transform(R, joint.xyz)

    def _joint_motion_transform(self, joint: UrdfJoint, value: float) -> np.ndarray:
        """
        根据 joint type 和 value 计算关节运动变换。
        """

        if joint.joint_type in ["fixed"]:
            return np.eye(4, dtype=float)

        if joint.joint_type in ["revolute", "continuous"]:
            R = self.axis_angle_to_matrix(joint.axis, value)
            return self.make_transform(R, np.zeros(3, dtype=float))

        if joint.joint_type == "prismatic":
            p = joint.axis * value
            return self.make_transform(np.eye(3, dtype=float), p)

        raise NotImplementedError(
            f"Unsupported joint type: {joint.joint_type} for joint {joint.name}"
        )

    def _joint_transform(self, joint: UrdfJoint, q_map: Dict[str, float]) -> np.ndarray:
        """
        URDF parent link 到 child link 的变换：

            T_parent_child = T_origin * T_motion
        """

        value = self._get_joint_value(joint, q_map)

        return self._joint_origin_transform(joint) @ self._joint_motion_transform(joint, value)

    # ============================================================
    # 正运动学 FK
    # ============================================================

    def fk_transform(self, q: np.ndarray) -> np.ndarray:
        """
        使用 URDF 链计算末端完整 4x4 位姿矩阵。

        参数：
            q:
                按 active_joint_names 顺序排列的关节角数组。

        返回：
            T:
                base_link 到 end_link 的 4x4 变换矩阵。
        """

        q_map = self._q_to_map(q)

        T = np.eye(4, dtype=float)

        for joint in self.chain_joints:
            T = T @ self._joint_transform(joint, q_map)

        return T

    def fk(self, q: np.ndarray) -> np.ndarray:
        """
        返回末端位置 [x, y, z]。
        """

        T = self.fk_transform(q)
        return T[:3, 3].copy()

    def fk_rotation(self, q: np.ndarray) -> np.ndarray:
        """
        返回末端旋转矩阵 R。
        """

        T = self.fk_transform(q)
        return T[:3, :3].copy()

    def fk_pose6(self, q: np.ndarray) -> np.ndarray:
        """
        返回末端 pose6:

            [x, y, z, rx, ry, rz]
        """

        T = self.fk_transform(q)
        p = T[:3, 3]
        rpy = self.matrix_to_rpy(T[:3, :3])

        return np.array(
            [p[0], p[1], p[2], rpy[0], rpy[1], rpy[2]],
            dtype=float,
        )

    def pose6_to_transform(self, pose6: np.ndarray) -> np.ndarray:
        pose6 = np.asarray(pose6, dtype=float).reshape(6)

        p = pose6[:3]
        rx, ry, rz = pose6[3:6]
        R = self.rpy_to_matrix(float(rx), float(ry), float(rz))

        return self.make_transform(R, p)

    # ============================================================
    # 位姿误差
    # ============================================================

    def pose_error_rotvec(
        self,
        target_pose6: np.ndarray,
        current_pose6: np.ndarray,
    ) -> np.ndarray:
        """
        使用 rotation vector 计算 6D 位姿误差。
        """

        target_pose6 = np.asarray(target_pose6, dtype=float).reshape(6)
        current_pose6 = np.asarray(current_pose6, dtype=float).reshape(6)

        p_des = target_pose6[:3]
        p_cur = current_pose6[:3]

        R_des = self.rpy_to_matrix(*target_pose6[3:6])
        R_cur = self.rpy_to_matrix(*current_pose6[3:6])

        pos_err = p_des - p_cur
        R_err = R_des @ R_cur.T
        rot_err = self.rotation_matrix_to_rotvec(R_err)

        return np.concatenate([pos_err, rot_err])

    def transform_error_rotvec(
        self,
        target_T: np.ndarray,
        current_T: np.ndarray,
    ) -> np.ndarray:
        target_T = np.asarray(target_T, dtype=float).reshape(4, 4)
        current_T = np.asarray(current_T, dtype=float).reshape(4, 4)

        pos_err = target_T[:3, 3] - current_T[:3, 3]

        R_des = target_T[:3, :3]
        R_cur = current_T[:3, :3]
        R_err = R_des @ R_cur.T
        rot_err = self.rotation_matrix_to_rotvec(R_err)

        return np.concatenate([pos_err, rot_err])

    def pose_error_rpy_subtract(
        self,
        target_pose6: np.ndarray,
        current_pose6: np.ndarray,
    ) -> np.ndarray:
        target_pose6 = np.asarray(target_pose6, dtype=float).reshape(6)
        current_pose6 = np.asarray(current_pose6, dtype=float).reshape(6)

        err = target_pose6 - current_pose6
        err[3:6] = self.normalize_angles(err[3:6])

        return err
    # ============================================================
    # 几何雅可比 Geometric Jacobian
    # ============================================================

    def forward_joint_origin_transforms(
        self,
        q: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        计算 active joint 的 joint origin 坐标系变换，以及末端 T。

        注意：
        URDF 中 parent -> child 的变换是：

            T_parent_child = T_origin * T_motion

        对于 revolute joint，关节轴 axis 定义在 joint origin 坐标系下。
        所以几何雅可比需要使用：

            T_base_joint_origin = T_base_parent * T_origin

        而不是使用 motion 之后的 child link 坐标系。

        返回：
            active_origin_T:
                {
                    joint_name: T_base_joint_origin
                }

            T_ee:
                base_link 到 end_link 的 4x4 变换
        """

        q_map = self._q_to_map(q)

        T = np.eye(4, dtype=float)
        active_origin_T: Dict[str, np.ndarray] = {}

        for joint in self.chain_joints:
            # 先计算 joint origin 在 base 下的位置
            T_origin = self._joint_origin_transform(joint)
            T_joint_origin = T @ T_origin

            if joint.name in self.active_joint_names:
                active_origin_T[joint.name] = T_joint_origin.copy()

            # 再应用 joint motion，得到 child link
            value = self._get_joint_value(joint, q_map)
            T_motion = self._joint_motion_transform(joint, value)

            T = T_joint_origin @ T_motion

        T_ee = T.copy()
        return active_origin_T, T_ee

    def geometric_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        计算 base 坐标系下的 6x7 几何雅可比。

        对 revolute joint：

            Jv_i = z_i × (p_ee - p_i)
            Jw_i = z_i

        对 prismatic joint：

            Jv_i = z_i
            Jw_i = 0

        返回：
            J[0:3, :] = 末端线速度部分 [vx, vy, vz]
            J[3:6, :] = 末端角速度部分 [wx, wy, wz]

        这个 Jacobian 适合用于：

            twist = J(q) @ dq

        其中 twist 是 base frame 下的：

            [vx, vy, vz, wx, wy, wz]
        """

        q = self._check_q(q)

        active_origin_T, T_ee = self.forward_joint_origin_transforms(q)

        p_ee = T_ee[:3, 3]

        n = len(self.active_joint_names)
        J = np.zeros((6, n), dtype=float)

        for i, joint_name in enumerate(self.active_joint_names):
            joint = self.joints_by_name[joint_name]

            if joint_name not in active_origin_T:
                raise ValueError(
                    f"Active joint {joint_name} is not in chain from "
                    f"{self.base_link} to {self.end_link}."
                )

            T_joint = active_origin_T[joint_name]

            p_i = T_joint[:3, 3]

            # URDF axis 是定义在 joint origin 坐标系下的
            # 所以要用 joint origin 的旋转矩阵转到 base 坐标系
            axis_base = T_joint[:3, :3] @ joint.axis

            norm = np.linalg.norm(axis_base)
            if norm > 1e-12:
                axis_base = axis_base / norm

            if joint.joint_type in ["revolute", "continuous"]:
                J[0:3, i] = np.cross(axis_base, p_ee - p_i)
                J[3:6, i] = axis_base

            elif joint.joint_type == "prismatic":
                J[0:3, i] = axis_base
                J[3:6, i] = np.zeros(3, dtype=float)

            else:
                # 理论上 active joints 不应该是 fixed
                J[:, i] = np.zeros(6, dtype=float)

        return J
    # ============================================================
    # 数值雅可比
    # ============================================================

    def numerical_jacobian(self, q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        位置雅可比 3x7。
        """

        q = self._check_q(q)

        J = np.zeros((3, len(self.active_joint_names)), dtype=float)

        for i in range(len(self.active_joint_names)):
            dq = np.zeros_like(q)
            dq[i] = eps

            p_plus = self.fk(q + dq)
            p_minus = self.fk(q - dq)

            J[:, i] = (p_plus - p_minus) / (2.0 * eps)

        return J

    def numerical_pose_jacobian(
        self,
        q: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray:
        """
        6D 位姿雅可比 6x7。

        位置部分：
            dp / dq

        姿态部分：
            d rotvec / dq
        """

        q = self._check_q(q)

        n = len(self.active_joint_names)
        J6 = np.zeros((6, n), dtype=float)

        for i in range(n):
            dq = np.zeros_like(q)
            dq[i] = eps

            T_plus = self.fk_transform(q + dq)
            T_minus = self.fk_transform(q - dq)

            p_plus = T_plus[:3, 3]
            p_minus = T_minus[:3, 3]

            R_plus = T_plus[:3, :3]
            R_minus = T_minus[:3, :3]

            J6[:3, i] = (p_plus - p_minus) / (2.0 * eps)

            R_delta = R_plus @ R_minus.T
            rotvec_delta = self.rotation_matrix_to_rotvec(R_delta)
            J6[3:6, i] = rotvec_delta / (2.0 * eps)

        return J6

    # ============================================================
    # 关节限位
    # ============================================================

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回 active joints 的 lower / upper limit。
        """

        lower = []
        upper = []

        for name in self.active_joint_names:
            joint = self.joints_by_name[name]
            lower.append(-np.inf if joint.lower is None else joint.lower)
            upper.append(np.inf if joint.upper is None else joint.upper)

        return np.array(lower, dtype=float), np.array(upper, dtype=float)

    def clamp_to_joint_limits(self, q: np.ndarray, margin: float = 1e-6) -> np.ndarray:
        """
        将 q 限制到 URDF 关节限位范围内。
        """

        q = self._check_q(q).copy()
        lower, upper = self.get_joint_limits()

        lower = np.where(np.isfinite(lower), lower + margin, lower)
        upper = np.where(np.isfinite(upper), upper - margin, upper)

        return np.clip(q, lower, upper)

    # ============================================================
    # 调试辅助
    # ============================================================

    def print_chain(self) -> None:
        print(f"base_link = {self.base_link}")
        print(f"end_link  = {self.end_link}")
        print("chain:")
        for joint in self.chain_joints:
            active_flag = "ACTIVE" if joint.name in self.active_joint_names else "fixed/passive"
            print(
                f"  {joint.parent} --[{joint.name}, {joint.joint_type}, {active_flag}]--> {joint.child}"
            )

    def print_pose(self, q: np.ndarray) -> None:
        pose = self.fk_pose6(q)
        print(
            "pose6 = "
            f"x={pose[0]:.4f}, y={pose[1]:.4f}, z={pose[2]:.4f}, "
            f"rx={pose[3]:.4f}, ry={pose[4]:.4f}, rz={pose[5]:.4f}"
        )


if __name__ == "__main__":
    """
    简单自测。

    你需要把 urdf_path 改成你的实际路径，例如：

        /home/hhl/xxx/fr3.urdf

    如果你把文件放到工程目录：

        config/fr3.urdf
        config/fr3_robot_description.yaml

    那么默认路径就可以直接使用。
    """

    kin = PandaKinematics(
        urdf_path="./config/fr3.urdf",
        robot_description_yaml_path="./config/fr3_robot_description.yaml",
        base_link="fr3_link0",
        end_link="fr3_hand_tcp",
    )

    kin.print_chain()

    q_test = np.array(
        [0.0, -1.3, 0.0, -2.87, 0.0, 2.0, 0.75],
        dtype=float,
    )

    p = kin.fk(q_test)
    T = kin.fk_transform(q_test)
    pose6 = kin.fk_pose6(q_test)
    J = kin.numerical_jacobian(q_test)
    J6 = kin.numerical_pose_jacobian(q_test)
    J_geo = kin.geometric_jacobian(q_test)

    lower, upper = kin.get_joint_limits()

    print("p =", p)
    print("T =\n", T)
    print("pose6 =", pose6)
    print("J shape =", J.shape)
    print("J6 shape =", J6.shape)
    print("J_geo shape =", J_geo.shape)
    print("J_geo =\n", J_geo)
    print("lower =", lower)
    print("upper =", upper)

    kin.print_pose(q_test)