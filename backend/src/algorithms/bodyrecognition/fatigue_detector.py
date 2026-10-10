"""基于 dlib 68 点关键点的疲劳驾驶检测器。

移植自 OpenFace 项目的疲劳检测示例（打哈欠 + 头部姿态），对外只暴露
:class:`FatigueDetector`，上层每次调用 ``process_frame(frame)`` 得到一个
:class:`FatigueResult`。
"""

from __future__ import annotations

import os
import time
from typing import Optional

import cv2
import dlib

from backend.src.algorithms.bodyrecognition.config import BodyRecognitionConfig
from backend.src.algorithms.bodyrecognition.landmarks import head_pose, mouth_aspect_ratio
from backend.src.algorithms.bodyrecognition.types import FatigueResult


class FatigueDetector:
    """疲劳驾驶检测器：打哈欠（MAR）+ 头部姿态角（pitch/yaw/roll）。

    打哈欠判定使用连续张嘴时长：MAR 持续超过 ``mar_threshold`` 达到
    ``yawn_seconds`` 后置位 ``is_drowsy``。
    """

    def __init__(self, config: Optional[BodyRecognitionConfig] = None):
        self.config = config if config is not None else BodyRecognitionConfig()
        if not os.path.isfile(self.config.predictor_path):
            raise FileNotFoundError(
                f"dlib 关键点模型不存在：{self.config.predictor_path}")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.config.predictor_path)
        self._yawn_start: Optional[float] = None

    def process_frame(self, frame) -> FatigueResult:
        """输入一帧 BGR 图像，返回 :class:`FatigueResult`。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)

        if not faces:
            self._yawn_start = None
            return FatigueResult(
                face_detected=False,
                mar=None,
                pitch=None,
                yaw=None,
                roll=None,
                is_yawning=False,
                yawn_duration=0.0,
                is_drowsy=False,
            )

        # 取面积最大的人脸，与 monitoring 中情绪识别保持一致。
        rect = max(faces, key=lambda r: r.width() * r.height())
        shape = self.predictor(gray, rect)
        landmarks = [(p.x, p.y) for p in shape.parts()]

        mar = mouth_aspect_ratio(landmarks)
        pitch = yaw = roll = None
        pose = head_pose(landmarks, frame.shape)
        if pose is not None and len(pose) >= 3:
            pitch, yaw, roll = float(pose[0]), float(pose[1]), float(pose[2])

        now = time.monotonic()
        if mar > self.config.mar_threshold:
            if self._yawn_start is None:
                self._yawn_start = now
            is_yawning = True
            yawn_duration = now - self._yawn_start
        else:
            self._yawn_start = None
            is_yawning = False
            yawn_duration = 0.0

        is_drowsy = is_yawning and yawn_duration >= self.config.yawn_seconds

        return FatigueResult(
            face_detected=True,
            mar=mar,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            is_yawning=is_yawning,
            yawn_duration=yawn_duration,
            is_drowsy=is_drowsy,
        )

    def reset(self) -> None:
        """重置打哈欠计时。"""
        self._yawn_start = None

    def close(self) -> None:
        """dlib 检测器无需显式释放资源，这里仅清理计时状态。"""
        self._yawn_start = None