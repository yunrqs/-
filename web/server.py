"""Local bridge between the browser UI and the existing Python detectors."""

from __future__ import annotations

import sys
import threading
import time
import contextlib
import io
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "zhayan" / "blink_detection"))
sys.path.insert(0, str(ROOT / "emo" / "emotion"))

app = Flask(__name__, static_folder=str(WEB_ROOT / "static"), static_url_path="/static")


def _load_face_cascade() -> cv2.CascadeClassifier:
    """加载 Haar 人脸检测级联模型。

    opencv-python 5.x 的 cv2.data.haarcascades 目录可能为空（不再随包分发
    XML 文件），因此优先使用项目自带的 web/haarcascades 目录，避免静默失败。
    """
    candidates = [
        WEB_ROOT / "haarcascades" / "haarcascade_frontalface_default.xml",
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
    ]
    for path in candidates:
        if path.is_file():
            cascade = cv2.CascadeClassifier(str(path))
            if not cascade.empty():
                return cascade
    raise RuntimeError(
        "未找到人脸检测模型 haarcascade_frontalface_default.xml，"
        f"请将模型文件放到 {WEB_ROOT / 'haarcascades'}"
    )


class AnalysisSession:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.blink_detector = None
        self.emotion_recognizer = None
        self.face_cascade = _load_face_cascade()
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
            from blink_detector import BlinkDetector
            from config import BlinkConfig
            from use_model import EmotionRecognizer

            model = ROOT / "emo" / "models" / "fer2013_best_model.pth"
            if not model.is_file():
                raise FileNotFoundError(f"情绪模型不存在：{model}")
            if self.blink_detector is not None:
                self.blink_detector.close()
            self.blink_detector = BlinkDetector(BlinkConfig(mode="robust"))
            # The legacy module prints Unicode status glyphs which can fail on
            # Windows consoles using GBK. The web UI owns user-facing status.
            with contextlib.redirect_stdout(io.StringIO()):
                self.emotion_recognizer = EmotionRecognizer(str(model))
            self.started_at = time.monotonic()

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


session = AnalysisSession()


@app.get("/")
def index():
    return send_from_directory(WEB_ROOT, "index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "ready": session.ready})


@app.post("/api/session/start")
def start_session():
    try:
        session.start()
        return jsonify({"ok": True, "ready": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/api/session/reset")
def reset_session():
    session.reset()
    return jsonify({"ok": True})


@app.post("/api/analyze")
def analyze():
    uploaded = request.files.get("frame")
    if uploaded is None:
        return jsonify({"error": "请求中缺少 frame 图像"}), 400
    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "无法解码摄像头帧"}), 400
    try:
        return jsonify(session.analyze(frame))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
