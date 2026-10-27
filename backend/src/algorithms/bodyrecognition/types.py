"""疲劳驾驶识别单帧输出类型。"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BodyPoseResult:
    """``BodyPoseDetector.process_frame()`` 的单帧输出（OpenPose 上半身）。

    关键点为 ``[x, y]`` 像素坐标，未检测到为 None；
    动作为几何指标，未检测到为 None。
    """

    person_detected: bool = False
    nose: Optional[List[float]] = None
    neck: Optional[List[float]] = None
    left_shoulder: Optional[List[float]] = None
    right_shoulder: Optional[List[float]] = None
    left_elbow: Optional[List[float]] = None
    right_elbow: Optional[List[float]] = None
    left_wrist: Optional[List[float]] = None
    right_wrist: Optional[List[float]] = None
    left_hip: Optional[List[float]] = None
    right_hip: Optional[List[float]] = None
    # 动作指标
    left_elbow_angle: Optional[float] = None    # 手肘弯曲角（度）
    right_elbow_angle: Optional[float] = None
    left_wrist_raise: Optional[float] = None    # 手腕相对肩部高度（归一化）
    right_wrist_raise: Optional[float] = None
    lean_angle: Optional[float] = None          # 身体侧倾角（度）
    shoulder_tilt: Optional[float] = None       # 肩部倾斜角（度）


@dataclass
class FatigueResult:
    """``FatigueDetector.process_frame()`` 的单帧输出。

    同时返回原始几何量（mar、pitch/yaw/roll）与业务判定
    （is_yawning / is_drowsy），以及可选的上半身肢体动作
    （pose，OpenPose），便于上层展示与调试。

    - is_drowsy：最终疲劳判定（打哈欠疲劳或肢体动作疲劳之一即为真）；
    - pose_drowsy：仅由肢体动作（身体侧倾）判定的疲劳；
    - pose_lean_duration：身体侧倾超过阈值的持续时长（秒）。
    """

    face_detected: bool
    mar: Optional[float]
    pitch: Optional[float]
    yaw: Optional[float]
    roll: Optional[float]
    is_yawning: bool
    yawn_duration: float
    is_drowsy: bool
    pose_drowsy: bool = False
    pose_lean_duration: float = 0.0
    pose: Optional[BodyPoseResult] = None