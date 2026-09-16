"""Validate and normalize a PAFE-style probe manifest.

The raw PAFE release is not bundled.  Convert its annotations into a CSV with:
subject_id, video_path, probe_time_sec, label.  This module validates that
portable interface and writes an absolute-path normalized manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mind_wandering.config import label_mapping

REQUIRED_COLUMNS = ("subject_id", "video_path", "probe_time_sec", "label")
ALIASES = {
    "focus": "focused",
    "focused": "focused",
    "on_task": "focused",
    "not-focus": "not_focused",
    "not_focus": "not_focused",
    "not-focused": "not_focused",
    "not_focused": "not_focused",
    "mind_wandering": "not_focused",
    "deliberate": "deliberate_mw",
    "intentional": "deliberate_mw",
    "deliberate_mw": "deliberate_mw",
    "spontaneous": "spontaneous_mw",
    "unintentional": "spontaneous_mw",
    "spontaneous_mw": "spontaneous_mw",
}


def normalize_label(value: object) -> str:
    key = str(value).strip().lower().replace(" ", "_")
    if key in {"skip", "other", "nan", ""}:
        return "skip"
    if key not in ALIASES:
        raise ValueError(f"Unsupported label: {value!r}")
    return ALIASES[key]


def prepare_manifest(input_csv: Path, output_csv: Path, task: str, check_files: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(input_csv)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["label"] = frame["label"].map(normalize_label)
    frame = frame.loc[frame["label"] != "skip"].reset_index(drop=True)
    allowed = label_mapping(task)
    unexpected = sorted(set(frame["label"]) - set(allowed))
    if unexpected:
        raise ValueError(f"Labels incompatible with task={task}: {unexpected}")

    base = input_csv.resolve().parent
    frame["video_path"] = frame["video_path"].map(
        lambda p: str((base / Path(str(p))).resolve()) if not Path(str(p)).is_absolute() else str(Path(str(p)).resolve())
    )
    frame["probe_time_sec"] = pd.to_numeric(frame["probe_time_sec"], errors="raise")
    if (frame["probe_time_sec"] <= 0).any():
        raise ValueError("probe_time_sec must be positive")
    if frame["subject_id"].isna().any():
        raise ValueError("subject_id cannot be empty")
    if check_files:
        absent = sorted({p for p in frame["video_path"] if not Path(p).is_file()})
        if absent:
            preview = "\n".join(absent[:5])
            raise FileNotFoundError(f"Video files not found ({len(absent)}):\n{preview}")

    frame["target"] = frame["label"].map(allowed)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a PAFE-style annotation manifest")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--task", choices=("binary", "three_class"), default="binary")
    parser.add_argument("--no-check-files", action="store_true", help="Validate schema before videos are available")
    args = parser.parse_args()
    frame = prepare_manifest(args.input, args.output, args.task, not args.no_check_files)
    print(f"Wrote {len(frame)} probes from {frame['subject_id'].nunique()} subjects to {args.output}")


if __name__ == "__main__":
    main()
