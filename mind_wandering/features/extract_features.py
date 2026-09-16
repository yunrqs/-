"""Extract blink and emotion features before each attention probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

from mind_wandering.config import DEFAULT_EMOTION_MODEL, EMOTIONS, PROJECT_ROOT
from mind_wandering.features.aggregation import aggregate_window


def _load_detectors(emotion_model: Path):
    blink_dir = PROJECT_ROOT / "zhayan" / "blink_detection"
    emotion_dir = PROJECT_ROOT / "emo" / "emotion"
    sys.path.insert(0, str(blink_dir))
    sys.path.insert(0, str(emotion_dir))
    from blink_detector import BlinkDetector
    from config import BlinkConfig
    from use_model import EmotionRecognizer

    return BlinkDetector(BlinkConfig(mode="robust")), EmotionRecognizer(str(emotion_model))


def extract_probe(video_path: Path, probe_sec: float, window_sec: float, sample_fps: float, emotion_model: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_sec = max(0.0, probe_sec - window_sec)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    stride = max(1, round(native_fps / sample_fps))
    detector, emotion = _load_detectors(emotion_model)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    rows, decoded = [], 0
    try:
        while cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 < probe_sec:
            ok, frame = cap.read()
            if not ok:
                break
            decoded += 1
            if (decoded - 1) % stride:
                continue
            blink = detector.process_frame(frame)
            row = {"face_detected": blink.face_detected, "ear": blink.ear, "blink": blink.blink_detected}
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces):
                x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
                label, _, probabilities = emotion.predict(frame[y:y+h, x:x+w])
                row["emotion"] = label
                for name in EMOTIONS:
                    row[f"prob_{name}"] = probabilities.get(name) if probabilities else None
            rows.append(row)
    finally:
        cap.release()
        detector.close()
    return aggregate_window(rows, window_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract paper-compatible features from probe windows")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument("--emotion-model", type=Path, default=DEFAULT_EMOTION_MODEL)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    records = []
    for index, probe in manifest.iterrows():
        features = extract_probe(Path(probe.video_path), float(probe.probe_time_sec), args.window_sec, args.sample_fps, args.emotion_model)
        records.append({**probe.to_dict(), "window_sec": args.window_sec, **features})
        print(f"[{index + 1}/{len(manifest)}] {probe.subject_id} @ {probe.probe_time_sec}s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False)
    print(f"Wrote {len(records)} feature rows to {args.output}")


if __name__ == "__main__":
    main()
