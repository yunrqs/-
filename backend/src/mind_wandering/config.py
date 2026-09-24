"""Shared labels and feature definitions."""

from backend.src.config.paths import PROJECT_ROOT, DEFAULT_EMOTION_MODEL

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
