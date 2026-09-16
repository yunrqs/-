"""Shared labels and feature definitions."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMOTION_MODEL = PROJECT_ROOT / "emo" / "models" / "fer2013_best_model.pth"

BINARY_LABELS = {"focused": 0, "not_focused": 1}
THREE_CLASS_LABELS = {
    "focused": 0,
    "deliberate_mw": 1,
    "spontaneous_mw": 2,
}

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
POSITIVE_EMOTIONS = {"happy", "surprise"}
NEGATIVE_EMOTIONS = {"angry", "disgust", "fear", "sad"}


def label_mapping(task: str) -> dict[str, int]:
    if task == "binary":
        return BINARY_LABELS
    if task == "three_class":
        return THREE_CLASS_LABELS
    raise ValueError(f"Unknown task: {task}")
