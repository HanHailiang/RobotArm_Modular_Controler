from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from sensor_msgs.msg import JointState


@dataclass
class RobotStateBuffer:
    """缓存 /joint_states，并按指定关节顺序输出 q、dq。"""
    joint_names: List[str]
    latest_joint_state: Optional[JointState] = None
    q_map: Dict[str, float] = field(default_factory=dict)
    dq_map: Dict[str, float] = field(default_factory=dict)
    last_error: str = ""

    def update_from_msg(self, msg: JointState) -> None:
        self.latest_joint_state = msg
        q_map: Dict[str, float] = {}
        dq_map: Dict[str, float] = {}
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                q_map[name] = float(msg.position[i])
            if i < len(msg.velocity):
                dq_map[name] = float(msg.velocity[i])
            else:
                dq_map[name] = 0.0
        self.q_map = q_map
        self.dq_map = dq_map

    def get_q_dq_arrays(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.latest_joint_state is None:
            self.last_error = "Waiting for /joint_states ..."
            return None
        q = np.zeros(7, dtype=float)
        dq = np.zeros(7, dtype=float)
        for i, name in enumerate(self.joint_names):
            if name not in self.q_map:
                self.last_error = f"Joint name mismatch: {name} not found."
                return None
            q[i] = self.q_map[name]
            dq[i] = self.dq_map.get(name, 0.0)
        self.last_error = ""
        return q, dq
