"""Predict labels from an already extracted feature CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from backend.src.config.paths import OUTPUTS_ROOT

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict mind wandering from feature rows")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUTS_ROOT / "predictions.csv")
    args = parser.parse_args()
    artifact = joblib.load(args.model)
    frame = pd.read_csv(args.features)
    columns = artifact["feature_columns"]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Features missing model columns: {sorted(missing)}")
    pred = artifact["pipeline"].predict(frame[columns])
    inverse = {value: key for key, value in artifact["labels"].items()}
    frame["predicted_target"] = pred
    frame["predicted_label"] = [inverse[int(value)] for value in pred]
    if hasattr(artifact["pipeline"], "predict_proba"):
        probabilities = artifact["pipeline"].predict_proba(frame[columns])
        for index, cls in enumerate(artifact["pipeline"].classes_):
            frame[f"prob_{inverse[int(cls)]}"] = probabilities[:, index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} predictions to {args.output}")


if __name__ == "__main__":
    main()
