"""Frame-to-window feature aggregation."""

from __future__ import annotations

from collections import Counter

import numpy as np

from mind_wandering.config import EMOTIONS, NEGATIVE_EMOTIONS, POSITIVE_EMOTIONS


def _stats(prefix: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {f"{prefix}_{name}": np.nan for name in ("mean", "std", "q25", "median", "q75", "mad")}
    a = np.asarray(values, dtype=float)
    median = float(np.median(a))
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_std": float(np.std(a)),
        f"{prefix}_q25": float(np.quantile(a, 0.25)),
        f"{prefix}_median": median,
        f"{prefix}_q75": float(np.quantile(a, 0.75)),
        f"{prefix}_mad": float(np.median(np.abs(a - median))),
    }


def aggregate_window(frame_rows: list[dict], duration_sec: float) -> dict[str, float]:
    total = len(frame_rows)
    valid_face = [r for r in frame_rows if r.get("face_detected")]
    out = _stats("ear", [float(r["ear"]) for r in valid_face if r.get("ear") is not None])
    blink_count = sum(bool(r.get("blink")) for r in frame_rows)
    out.update({
        "blink_count": float(blink_count),
        "blink_rate_per_min": float(blink_count * 60.0 / duration_sec),
        "valid_face_ratio": float(len(valid_face) / total) if total else 0.0,
        "frame_count": float(total),
    })
    labels = Counter(r.get("emotion") for r in valid_face if r.get("emotion"))
    for emotion in EMOTIONS:
        probs = [float(r[f"prob_{emotion}"]) for r in valid_face if r.get(f"prob_{emotion}") is not None]
        out[f"emotion_{emotion}_mean"] = float(np.mean(probs)) if probs else np.nan
        out[f"emotion_{emotion}_std"] = float(np.std(probs)) if probs else np.nan
        out[f"emotion_{emotion}_ratio"] = float(labels[emotion] / len(valid_face)) if valid_face else 0.0
    out["positive_valence"] = sum(out[f"emotion_{e}_mean"] for e in POSITIVE_EMOTIONS)
    out["negative_valence"] = sum(out[f"emotion_{e}_mean"] for e in NEGATIVE_EMOTIONS)
    out["neutral_valence"] = out["emotion_neutral_mean"]
    return out
