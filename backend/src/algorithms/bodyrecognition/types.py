"""疲劳驾驶识别单帧输出类型。"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FatigueResult:
    """``FatigueDetector.process_frame()`` 的单帧输出。

    同时返回原始几何量（mar、pitch/yaw/roll）与业务判定
    （is_yawning / is_drowsy），便于上层展示与调试。
    """

    face_detected: bool
    mar: Optional[float]
    pitch: Optional[float]
    yaw: Optional[float]
    roll: Optional[float]
    is_yawning: bool
    yawn_duration: float
    is_drowsy: bool