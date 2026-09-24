"""Train and compare classical ML models with subject-independent splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from backend.src.config.paths import OUTPUTS_ROOT

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import make_column_selector
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score, average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from backend.src.mind_wandering.config import label_mapping

META_COLUMNS = {
    "subject_id", "video_path", "probe_time_sec", "label", "target", "window_sec",
    "stage", "probe_index", "probe_time", "video_time_sec",
}


def models(seed: int) -> dict[str, object]:
    scaled = lambda model: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", model)])
    plain = lambda model: Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    return {
        "random_forest": plain(RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)),
        "decision_tree": plain(DecisionTreeClassifier(class_weight="balanced", random_state=seed)),
        "gradient_boosting": plain(GradientBoostingClassifier(random_state=seed)),
        "svm": scaled(SVC(class_weight="balanced", probability=True, random_state=seed)),
        "knn": scaled(KNeighborsClassifier()),
        "logistic_regression": scaled(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)),
        "naive_bayes": scaled(GaussianNB()),
    }


def train(features_csv: Path, output_dir: Path, task: str, folds: int, seed: int) -> dict:
    data = pd.read_csv(features_csv)
    required = {"subject_id", "label"}
    if missing := required - set(data.columns):
        raise ValueError(f"Feature CSV is missing: {sorted(missing)}")
    mapping = label_mapping(task)
    unknown = set(data.label) - set(mapping)
    if unknown:
        raise ValueError(f"Labels incompatible with task={task}: {sorted(unknown)}")
    y = data.label.map(mapping).to_numpy()
    groups = data.subject_id.astype(str).to_numpy()
    numeric = data.drop(columns=[c for c in META_COLUMNS if c in data], errors="ignore").select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("No numeric feature columns found")
    splits = min(folds, len(np.unique(groups)))
    if splits < 2:
        raise ValueError("At least two subjects are required for grouped validation")
    cv = GroupKFold(n_splits=splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores, predictions = {}, {}
    for name, estimator in models(seed).items():
        pred = cross_val_predict(estimator, numeric, y, groups=groups, cv=cv, method="predict", n_jobs=1)
        entry = {
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"),
            "weighted_f1": f1_score(y, pred, average="weighted"),
            "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred).tolist(),
        }
        scores[name] = entry
        predictions[name] = pred
        print(f"{name:20} balanced_accuracy={entry['balanced_accuracy']:.3f} macro_f1={entry['macro_f1']:.3f}")
    best_name = max(scores, key=lambda n: scores[n]["macro_f1"])
    best = clone(models(seed)[best_name]).fit(numeric, y)
    artifact = {"pipeline": best, "feature_columns": list(numeric.columns), "task": task, "labels": mapping, "best_model": best_name}
    joblib.dump(artifact, output_dir / "mind_wandering_model.joblib")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"task": task, "folds": splits, "best_model": best_name, "models": scores}, handle, ensure_ascii=False, indent=2)
    pd.DataFrame({"subject_id": groups, "true": y, **predictions}).to_csv(output_dir / "cross_validated_predictions.csv", index=False)
    return {"best_model": best_name, "scores": scores}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mind-wandering classifiers")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_ROOT / "experiment")
    parser.add_argument("--task", choices=("binary", "three_class"), default="binary")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train(args.features, args.output_dir, args.task, args.folds, args.seed)
    print(f"Best model: {result['best_model']}")


if __name__ == "__main__":
    main()
