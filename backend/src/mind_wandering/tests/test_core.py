from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from backend.src.mind_wandering.dataset.prepare_manifest import normalize_label, prepare_manifest
from backend.src.mind_wandering.dataset.prepare_mwdet import _is_positive
from backend.src.mind_wandering.features.aggregation import aggregate_window
from backend.src.mind_wandering.training.train import train


class CoreTests(unittest.TestCase):
    def test_label_aliases(self):
        self.assertEqual(normalize_label("Focused"), "focused")
        self.assertEqual(normalize_label("Not-Focused"), "not_focused")
        self.assertEqual(normalize_label("Skip"), "skip")

    def test_mwdet_rating_is_paired_to_bell(self):
        bell = {"time": "2017-01-01T00:00:00Z", "stage": "1"}
        ratings = [{"time": "2017-01-01T00:00:01Z", "stage": "1", "rating": "1"}]
        self.assertTrue(_is_positive(bell, ratings, 5.0))
        self.assertFalse(_is_positive(bell, ratings, 0.5))

    def test_manifest_can_be_checked_without_videos(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "raw.csv", root / "manifest.csv"
            pd.DataFrame([{"subject_id": "P1", "video_path": "missing.mp4", "probe_time_sec": 40, "label": "Focused"}]).to_csv(source, index=False)
            result = prepare_manifest(source, output, "binary", check_files=False)
            self.assertEqual(result.iloc[0].target, 0)
            self.assertTrue(output.exists())

    def test_aggregation(self):
        rows = [
            {"face_detected": True, "ear": 0.3, "blink": False, "emotion": "happy", **{f"prob_{e}": (1.0 if e == "happy" else 0.0) for e in ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")}},
            {"face_detected": True, "ear": 0.1, "blink": True, "emotion": "sad", **{f"prob_{e}": (1.0 if e == "sad" else 0.0) for e in ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")}},
        ]
        result = aggregate_window(rows, 5.0)
        self.assertEqual(result["blink_count"], 1.0)
        self.assertEqual(result["blink_rate_per_min"], 12.0)
        self.assertAlmostEqual(result["ear_mean"], 0.2)

    def test_training_smoke(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for subject in range(6):
                rows.append({"subject_id": f"P{subject}", "label": "focused", "ear_mean": 0.3 + subject * 0.001, "blink_rate_per_min": 8.0})
                rows.append({"subject_id": f"P{subject}", "label": "not_focused", "ear_mean": 0.2 + subject * 0.001, "blink_rate_per_min": 18.0})
            features = root / "features.csv"
            pd.DataFrame(rows).to_csv(features, index=False)
            result = train(features, root / "outputs", "binary", folds=3, seed=42)
            self.assertIn(result["best_model"], result["scores"])
            self.assertTrue((root / "outputs" / "mind_wandering_model.joblib").exists())


if __name__ == "__main__":
    unittest.main()
