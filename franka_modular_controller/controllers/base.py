from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ControlDebugData:
    tau_total: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=float))
    tau_cart: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=float))
    tau_null: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=float))
    tau_dist: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=float))
    p: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    p_des: Optional[np.ndarray] = None
    p_err: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    mode: str = "disabled"
    message: str = ""


class BaseController:
    """控制器基类。后续位置控制、力控、导纳控制都可以继承它。"""

    def compute(self, q: np.ndarray, dq: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError

    def capture_target(self, q: np.ndarray) -> None:
        raise NotImplementedError

    def get_debug_data(self) -> ControlDebugData:
        raise NotImplementedError
