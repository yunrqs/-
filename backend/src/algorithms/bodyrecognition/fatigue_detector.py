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

from backend.src.algorithms.bodyrecognition.body_pose import BodyPoseDetector
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
        self._lean_start: Optional[float] = None
        self._pose: Optional[BodyPoseDetector] = None
        if self.config.pose_enabled:
            self._pose = BodyPoseDetector(
                model_path=self.config.pose_model_path,
                interval=self.config.pose_interval,
                device=self.config.pose_device,
            )

    def process_frame(self, frame) -> FatigueResult:
        """输入一帧 BGR 图像，返回 :class:`FatigueResult`。"""
        # OpenPose 上半身肢体动作（启用时按隔帧策略运行，且不依赖人脸检测）。
        body_pose = self._pose.process_frame(frame) if self._pose is not None else None

        # 肢体动作疲劳：身体侧倾角持续超过阈值即视为打瞌睡侧倾。
        now = time.monotonic()
        lean_angle = (
            body_pose.lean_angle
            if body_pose is not None and body_pose.person_detected else None
        )
        if lean_angle is not None and abs(lean_angle) > self.config.pose_lean_threshold:
            if self._lean_start is None:
                self._lean_start = now
            pose_lean_duration = now - self._lean_start
        else:
            self._lean_start = None
            pose_lean_duration = 0.0
        pose_drowsy = pose_lean_duration >= self.config.pose_fatigue_seconds

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
                is_drowsy=pose_drowsy,
                pose_drowsy=pose_drowsy,
                pose_lean_duration=pose_lean_duration,
                pose=body_pose,
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

        if mar > self.config.mar_threshold:
            if self._yawn_start is None:
                self._yawn_start = now
            is_yawning = True
            yawn_duration = now - self._yawn_start
        else:
            self._yawn_start = None
            is_yawning = False
            yawn_duration = 0.0

        # 最终疲劳判定：打哈欠疲劳 或 肢体动作疲劳，任一即视为疲劳。
        is_drowsy = (
            (is_yawning and yawn_duration >= self.config.yawn_seconds)
            or pose_drowsy
        )

        return FatigueResult(
            face_detected=True,
            mar=mar,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            is_yawning=is_yawning,
            yawn_duration=yawn_duration,
            is_drowsy=is_drowsy,
            pose_drowsy=pose_drowsy,
            pose_lean_duration=pose_lean_duration,
            pose=body_pose,
        )

    def reset(self) -> None:
        """重置打哈欠计时、侧倾计时与肢体动作缓存。"""
        self._yawn_start = None
        self._lean_start = None
        if self._pose is not None:
            self._pose.reset()

    def close(self) -> None:
        """释放模型资源。"""
        self._yawn_start = None
        self._lean_start = None
        if self._pose is not None:
            self._pose.close()
        self._pose = None