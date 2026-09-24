"""识别会话：调用共享情绪与眨眼算法。"""
from __future__ import annotations

import contextlib
import io
import threading
import time

import cv2
import numpy as np

from backend.src.config.paths import DEFAULT_EMOTION_MODEL


class AnalysisSession:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.blink_detector = None
        self.emotion_recognizer = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.started_at = time.monotonic()
        self.frames = 0
        self.face_frames = 0

    @property
    def ready(self) -> bool:
        return self.blink_detector is not None and self.emotion_recognizer is not None

    def start(self) -> None:
        with self.lock:
            if self.ready:
                return
            from backend.src.algorithms.blink_detection.blink_detector import BlinkDetector
            from backend.src.algorithms.blink_detection.config import BlinkConfig
            from backend.src.algorithms.emotion.use_model import EmotionRecognizer

            model = DEFAULT_EMOTION_MODEL
            if not model.is_file():
                raise FileNotFoundError(f"情绪模型不存在：{model}")
            # 两个模型都成功后再更新会话，加载失败时释放已经创建的资源。
            detector = BlinkDetector(BlinkConfig(mode="robust"))
            try:
                # 旧模块的 Unicode 输出在部分 Windows 控制台会报编码错误。
                with contextlib.redirect_stdout(io.StringIO()):
                    recognizer = EmotionRecognizer(str(model))
            except Exception:
                detector.close()
                raise
            self.blink_detector = detector
            self.emotion_recognizer = recognizer
            self.started_at = time.monotonic()

    def close(self) -> None:
        """服务退出时释放 MediaPipe 模型资源。"""
        with self.lock:
            if self.blink_detector is not None:
                self.blink_detector.close()
            self.blink_detector = None
            self.emotion_recognizer = None

    def reset(self) -> None:
        with self.lock:
            if self.blink_detector is not None:
                self.blink_detector.reset()
            if self.emotion_recognizer is not None and hasattr(self.emotion_recognizer, "history"):
                self.emotion_recognizer.history.clear()
            self.started_at = time.monotonic()
            self.frames = 0
            self.face_frames = 0

    def analyze(self, frame: np.ndarray) -> dict:
        with self.lock:
            if not self.ready:
                raise RuntimeError("识别模型尚未加载")
            self.frames += 1
            blink = self.blink_detector.process_frame(frame)
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
            }

