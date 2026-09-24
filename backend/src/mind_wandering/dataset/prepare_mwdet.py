"""Convert the public MWDET WebGazer release into model-ready features.

Each auditory bell defines the end of an observation interval.  A button press
(`rating=1`) shortly after that bell labels the preceding interval as
mind-wandering; no press labels it focused.  This follows Zhao et al. (2017).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _summary(prefix: str, values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    names = ("mean", "std", "min", "q25", "median", "q75", "max", "range")
    if clean.empty:
        return {f"{prefix}_{name}": np.nan for name in names}
    return {
        f"{prefix}_mean": float(clean.mean()),
        f"{prefix}_std": float(clean.std(ddof=0)),
        f"{prefix}_min": float(clean.min()),
        f"{prefix}_q25": float(clean.quantile(0.25)),
        f"{prefix}_median": float(clean.median()),
        f"{prefix}_q75": float(clean.quantile(0.75)),
        f"{prefix}_max": float(clean.max()),
        f"{prefix}_range": float(clean.max() - clean.min()),
    }


def _parse_time(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True)


def _is_positive(bell: dict, ratings: list[dict], response_sec: float) -> bool:
    bell_time = _parse_time(bell["time"])
    for rating in ratings:
        if str(rating.get("stage")) != str(bell.get("stage")):
            continue
        delay = (_parse_time(rating["time"]) - bell_time).total_seconds()
        if 0 <= delay <= response_sec and str(rating.get("rating")) == "1":
            return True
    return False


def _interval_features(rows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    part = rows.loc[(rows["timestamp"] >= start) & (rows["timestamp"] < end)].copy()
    duration = max((end - start).total_seconds(), 1e-9)
    out: dict[str, float] = {
        "interval_sec": duration,
        "sample_count": float(len(part)),
        "sample_rate_hz": float(len(part) / duration),
    }
    for source, prefix in (("GazeX_s_px", "gaze_x"), ("GazeY_s_px", "gaze_y"),
                           ("GazeEventDuration", "event_duration_ms"),
                           ("AbsoluteSaccadicDirection", "saccade_angle")):
        out.update(_summary(prefix, part[source] if source in part else pd.Series(dtype=float)))

    x = pd.to_numeric(part.get("GazeX_s_px"), errors="coerce")
    y = pd.to_numeric(part.get("GazeY_s_px"), errors="coerce")
    valid = x.notna() & y.notna()
    out["valid_gaze_ratio"] = float(valid.mean()) if len(part) else 0.0
    if valid.sum() >= 2:
        xy = np.column_stack([x[valid].to_numpy(), y[valid].to_numpy()])
        distance = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        out.update(_summary("gaze_step_px", pd.Series(distance)))
        out["gaze_path_px"] = float(distance.sum())
    else:
        out.update(_summary("gaze_step_px", pd.Series(dtype=float)))
        out["gaze_path_px"] = np.nan

    if "saccade" in part:
        saccade = part["saccade"].astype(str).str.upper().eq("TRUE")
        out["saccade_sample_ratio"] = float(saccade.mean()) if len(part) else 0.0
        out["saccade_count"] = float((saccade & ~saccade.shift(fill_value=False)).sum())
        out["saccade_rate_per_min"] = out["saccade_count"] * 60.0 / duration
    else:
        out.update({"saccade_sample_ratio": np.nan, "saccade_count": np.nan, "saccade_rate_per_min": np.nan})

    if "FixationIndex" in part:
        fixation = part.dropna(subset=["FixationIndex"]).drop_duplicates("FixationIndex")
        out["fixation_count"] = float(len(fixation))
        out["fixation_rate_per_min"] = float(len(fixation) * 60.0 / duration)
        out.update(_summary("fixation_duration_ms", fixation.get("GazeEventDuration", pd.Series(dtype=float))))
    return out


def convert_subject(event_file: Path, gaze_file: Path, response_sec: float = 5.0) -> list[dict]:
    with event_file.open(encoding="utf-8") as handle:
        user = json.load(handle)["user"]
    gaze = pd.read_csv(gaze_file)
    timestamp_column = "Timestamp_utc"
    gaze["timestamp"] = pd.to_datetime(gaze[timestamp_column], utc=True)
    bells = sorted(user["ratingbells"], key=lambda item: _parse_time(item["time"]))
    starts = {
        str(item["stage"]): _parse_time(item["time"])
        for item in user.get("videostatus", []) if item.get("status") == "PLAYING"
    }
    previous = dict(starts)
    records = []
    for index, bell in enumerate(bells):
        stage = str(bell["stage"])
        end = _parse_time(bell["time"])
        start = previous.get(stage)
        previous[stage] = end
        if start is None or end <= start:
            continue
        positive = _is_positive(bell, user.get("ratings", []), response_sec)
        records.append({
            "subject_id": event_file.stem,
            "stage": stage,
            "probe_index": index,
            "probe_time": end.isoformat(),
            "video_time_sec": float(bell["videoTime"]),
            "label": "not_focused" if positive else "focused",
            "target": int(positive),
            **_interval_features(gaze, start, end),
        })
    return records


def convert_dataset(root: Path, output: Path, response_sec: float = 5.0) -> pd.DataFrame:
    event_dir = root / "Data_Event"
    gaze_dir = root / "Data_WebGazer" / "Data_ProcessedByScript02"
    records = []
    for event_file in sorted(event_dir.glob("Anon*.json")):
        gaze_file = gaze_dir / f"{event_file.stem}.csv"
        if not gaze_file.is_file():
            raise FileNotFoundError(f"Missing WebGazer file: {gaze_file}")
        records.extend(convert_subject(event_file, gaze_file, response_sec))
    frame = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real MWDET WebGazer features")
    parser.add_argument("--root", type=Path, required=True, help="Extracted Data_Publish/Data_Publish directory")
    parser.add_argument("--output", type=Path, default=Path("data/processed/mwdet_webgazer_features.csv"))
    parser.add_argument("--response-sec", type=float, default=5.0)
    args = parser.parse_args()
    frame = convert_dataset(args.root, args.output, args.response_sec)
    counts = frame["label"].value_counts().to_dict()
    print(f"Wrote {len(frame)} probes from {frame.subject_id.nunique()} subjects to {args.output}")
    print(f"Class counts: {counts}")


if __name__ == "__main__":
    main()
