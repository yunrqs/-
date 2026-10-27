"""识别会话：调用共享情绪与眨眼算法。"""
from __future__ import annotations

import contextlib
import io
import threading
import time
from dataclasses import asdict

import cv2
import numpy as np

from backend.src.config.paths import DEFAULT_EMOTION_MODEL, DEFAULT_FACE_PREDICTOR


class AnalysisSession:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.blink_detector = None
        self.emotion_recognizer = None
        self.fatigue_detector = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.started_at = time.monotonic()
        self.frames = 0
        self.face_frames = 0

    @property
    def ready(self) -> bool:
        return (
            self.blink_detector is not None
            and self.emotion_recognizer is not None
            and self.fatigue_detector is not None
        )

    def start(self) -> None:
        with self.lock:
            if self.ready:
                return
            from backend.src.algorithms.blink_detection.blink_detector import BlinkDetector
            from backend.src.algorithms.blink_detection.config import BlinkConfig
            from backend.src.algorithms.emotion.use_model import EmotionRecognizer
            from backend.src.algorithms.bodyrecognition.fatigue_detector import FatigueDetector

            model = DEFAULT_EMOTION_MODEL
            if not model.is_file():
                raise FileNotFoundError(f"情绪模型不存在：{model}")
            if not DEFAULT_FACE_PREDICTOR.is_file():
                raise FileNotFoundError(f"dlib 关键点模型不存在：{DEFAULT_FACE_PREDICTOR}")
            # 两个模型都成功后再更新会话，加载失败时释放已经创建的资源。
            detector = BlinkDetector(BlinkConfig(mode="robust"))
            try:
                # 旧模块的 Unicode 输出在部分 Windows 控制台会报编码错误。
                with contextlib.redirect_stdout(io.StringIO()):
                    recognizer = EmotionRecognizer(str(model))
                fatigue = FatigueDetector()
            except Exception:
                detector.close()
                raise
            self.blink_detector = detector
            self.emotion_recognizer = recognizer
            self.fatigue_detector = fatigue
            self.started_at = time.monotonic()

    def close(self) -> None:
        """服务退出时释放 MediaPipe 模型资源。"""
        with self.lock:
            if self.blink_detector is not None:
                self.blink_detector.close()
            self.blink_detector = None
            self.emotion_recognizer = None
            if self.fatigue_detector is not None:
                self.fatigue_detector.close()
            self.fatigue_detector = None

    def reset(self) -> None:
        with self.lock:
            if self.blink_detector is not None:
                self.blink_detector.reset()
            if self.emotion_recognizer is not None and hasattr(self.emotion_recognizer, "history"):
                self.emotion_recognizer.history.clear()
            if self.fatigue_detector is not None:
                self.fatigue_detector.reset()
            self.started_at = time.monotonic()
            self.frames = 0
            self.face_frames = 0

    def analyze(self, frame: np.ndarray) -> dict:
        with self.lock:
            if not self.ready:
                raise RuntimeError("识别模型尚未加载")
            self.frames += 1
            blink = self.blink_detector.process_frame(frame)
            fatigue = self.fatigue_detector.process_frame(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            face_box = None
            emotion = None
            confidence = None
            probabilities = {}
            if len(faces):
                x, y, w, h = max(faces, key=lambda item: int(item[2]) * int(item[3]))
                face_box = [int(x), int(y), int(w), int(h)]
                roi = frame[y:y + h, x:x + w]
                emotion, confidence, probabilities = self.emotion_recognizer.predict(roi)
            if blink.face_detected:
                self.face_frames += 1
            elapsed = max(time.monotonic() - self.started_at, 1.0)
            fatigue_flags = self._combine_fatigue(blink, fatigue, emotion, confidence)
            return {
                "frame_width": int(frame.shape[1]),
                "frame_height": int(frame.shape[0]),
                "face_detected": bool(blink.face_detected),
                "face_box": face_box,
                "ear": blink.ear,
                "threshold": blink.threshold,
                "eye_state": blink.state,
                "blink_detected": bool(blink.blink_detected),
                "blink_count": int(self.blink_detector.blink_count),
                "blink_rate_per_min": float(self.blink_detector.blink_count * 60.0 / elapsed),
                "valid_face_ratio": float(self.face_frames / self.frames),
                "emotion": emotion,
                "emotion_confidence": confidence,
                "emotion_probabilities": probabilities,
                "mar": fatigue.mar,
                "pitch": fatigue.pitch,
                "yaw": fatigue.yaw,
                "roll": fatigue.roll,
                "is_yawning": fatigue.is_yawning,
                "yawn_duration": fatigue.yawn_duration,
                "is_drowsy": fatigue.is_drowsy,
                "pose_drowsy": fatigue.pose_drowsy,
                "pose_lean_duration": fatigue.pose_lean_duration,
                "is_fatigued": fatigue_flags["is_fatigued"],
                "eye_drowsy": fatigue_flags["eye_drowsy"],
                "emotion_drowsy": fatigue_flags["emotion_drowsy"],
                "gaze_drowsy": fatigue_flags["gaze_drowsy"],
                "pose": asdict(fatigue.pose) if fatigue.pose is not None else None,
            }

    def _combine_fatigue(self, blink, fatigue, emotion, emotion_confidence) -> dict:
        """汇总各信号源，得到最终疲劳判定。

        这是所有疲劳信号的唯一融合点：眨眼、打哈欠/肢体动作已接入；
        情绪与视线为预留在位组合点，待确定各自的"疲劳信号 -> 布尔判定"规则后
        在本方法内补充（无需改动 ``analyze`` 或上层接口）。
        """
        # 眨眼：眼睛持续闭合（状态机 CLOSED 需连续多帧低 EAR，非单帧抖动）。
        eye_drowsy = bool(
            blink is not None and blink.face_detected and blink.state == "CLOSED"
        )
        # 打哈欠 + 肢体动作：FatigueDetector 内部已完成时长判定后置位。
        yawn_pose_drowsy = bool(fatigue is not None and fatigue.is_drowsy)
        # 情绪：预留在位，尚未接入情绪 -> 疲劳的映射（如低落/困倦持续时长）。
        emotion_drowsy = False  # TODO(情绪): 依据 emotion/emotion_confidence 判定。
        # 视线：预留在位，实时视线检测暂未接入（离线仅 mind_wandering 有眼动特征）。
        gaze_drowsy = False  # TODO(视线): 接入注视偏移/眼动特征后再判定。
        is_fatigued = eye_drowsy or yawn_pose_drowsy or emotion_drowsy or gaze_drowsy
        return {
            "is_fatigued": is_fatigued,
            "eye_drowsy": eye_drowsy,
            "emotion_drowsy": emotion_drowsy,
            "gaze_drowsy": gaze_drowsy,
        }

