"""单元测试（覆盖规格说明第 32 节的 6 个测试用例）。

运行方式（在 ``blink_detection/`` 目录下）::

    python -m unittest test_blink_detector -v
    # 或
    python -m unittest discover -p "test_*.py" -v
"""

import unittest

from adaptive_threshold import Calibrator, calculate_threshold, update_baseline
from blink_detector import BlinkDetector, BlinkStateMachine
from blink_types import EyeState
from config import BlinkConfig
from ear import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    calculate_average_ear,
    calculate_ear,
)


class _FakeFaceMesh:
    """不依赖 mediapipe 的 fake face mesh，``process`` 返回固定 landmarks。"""

    def __init__(self, landmarks=None):
        self._landmarks = landmarks

    def process(self, frame):
        return self._landmarks

    def close(self):
        pass


class _SequenceFaceMesh:
    """按顺序逐帧返回不同的 landmarks（用于完整 pipeline 测试）。"""

    def __init__(self, sequence):
        self._sequence = sequence
        self._index = 0

    def process(self, frame):
        landmarks = self._sequence[min(self._index, len(self._sequence) - 1)]
        self._index += 1
        return landmarks

    def close(self):
        pass


def _make_landmarks(ear_value: float, horizontal: float = 0.1):
    """构造 468 个 landmark，使左右眼 EAR 近似等于 ``ear_value``。

    眼睛几何：p1/p4 为左右眼角（水平距离 = horizontal），p2/p3 为上眼睑，
    p5/p6 为下眼睑。令上下垂直距离 v = ear_value * horizontal，则
    EAR = (v + v) / (2 * horizontal) = ear_value。
    """
    landmarks = [(0.0, 0.0, 0.0)] * 468

    def set_eye(indices, cx, cy):
        h = horizontal
        v = ear_value * h
        pts = {
            indices[0]: (cx - h / 2, cy),          # p1
            indices[1]: (cx - h / 4, cy - v / 2),  # p2
            indices[2]: (cx + h / 4, cy - v / 2),  # p3
            indices[3]: (cx + h / 2, cy),          # p4
            indices[4]: (cx + h / 4, cy + v / 2),  # p5
            indices[5]: (cx - h / 4, cy + v / 2),  # p6
        }
        for idx, (x, y) in pts.items():
            landmarks[idx] = (x, y, 0.0)

    set_eye(LEFT_EYE_INDICES, 0.30, 0.50)
    set_eye(RIGHT_EYE_INDICES, 0.70, 0.50)
    return landmarks


class TestEar(unittest.TestCase):
    """Test 6 与 EAR 基础计算。"""

    def test_calculate_ear_normal(self):
        # p1=(0,0) p4=(1,0) 水平=1；两条垂直各 0.1 -> EAR = 0.1
        pts = [
            (0.0, 0.0),
            (0.25, -0.05),
            (0.75, -0.05),
            (1.0, 0.0),
            (0.75, 0.05),
            (0.25, 0.05),
        ]
        self.assertAlmostEqual(calculate_ear(pts), 0.1, places=6)

    def test_calculate_ear_returns_none_when_p1_equals_p4(self):
        # Test 6：p1 == p4 -> 水平距离为 0 -> EAR = None，且不抛 ZeroDivisionError
        pts = [
            (0.5, 0.5),
            (0.4, 0.4),
            (0.6, 0.4),
            (0.5, 0.5),
            (0.6, 0.6),
            (0.4, 0.6),
        ]
        self.assertIsNone(calculate_ear(pts))

    def test_average_ear_single_eye_fallback(self):
        self.assertEqual(calculate_average_ear(None, 0.3), 0.3)
        self.assertEqual(calculate_average_ear(0.3, None), 0.3)
        self.assertIsNone(calculate_average_ear(None, None))
        self.assertAlmostEqual(calculate_average_ear(0.2, 0.4), 0.3)


class TestStateMachine(unittest.TestCase):
    """Test 1~4：状态机时序行为。"""

    THRESHOLD = 0.21

    def _sm(self):
        return BlinkStateMachine(consec_frames=3)

    def test_open_no_blink(self):
        # Test 1：EAR > threshold -> OPEN，无 blink
        sm = self._sm()
        for ear in (0.31, 0.32, 0.30):
            self.assertFalse(sm.update(ear, self.THRESHOLD))
        self.assertIs(sm.state, EyeState.OPEN)

    def test_single_low_frame_no_blink(self):
        # Test 2：单帧异常低 EAR -> 0 blink
        sm = self._sm()
        seq = [0.31, 0.31, 0.15, 0.31, 0.31]
        blinks = [sm.update(e, self.THRESHOLD) for e in seq]
        self.assertEqual(sum(blinks), 0)
        self.assertIs(sm.state, EyeState.OPEN)

    def test_full_blink(self):
        # Test 3：OPEN -> LOW LOW LOW -> OPEN -> 1 blink
        sm = self._sm()
        seq = [0.31, 0.15, 0.15, 0.15, 0.31]
        blinks = [sm.update(e, self.THRESHOLD) for e in seq]
        self.assertEqual(sum(blinks), 1)

    def test_long_close_single_blink(self):
        # Test 4：长时间闭眼只产生 1 次 blink
        sm = self._sm()
        seq = [0.31, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.31]
        blinks = [sm.update(e, self.THRESHOLD) for e in seq]
        self.assertEqual(sum(blinks), 1)


class TestAdaptiveThreshold(unittest.TestCase):
    def test_calibrator_uses_median(self):
        cal = Calibrator(required_samples=3)
        cal.add_sample(0.30)
        cal.add_sample(0.32)
        cal.add_sample(0.31)
        self.assertTrue(cal.is_ready())
        self.assertAlmostEqual(cal.get_baseline(), 0.31)

    def test_update_baseline_freezes_when_not_open(self):
        baseline = 0.3
        self.assertEqual(update_baseline(0.1, EyeState.CLOSED, baseline, 0.01), baseline)
        self.assertEqual(update_baseline(0.1, EyeState.CLOSING, baseline, 0.01), baseline)
        # OPEN 时 EMA 更新
        new = update_baseline(0.31, EyeState.OPEN, baseline, 0.01)
        self.assertAlmostEqual(new, 0.3 + 0.01 * 0.01)

    def test_calculate_threshold(self):
        self.assertAlmostEqual(calculate_threshold(0.3, 0.7), 0.21)


class TestDetector(unittest.TestCase):
    def test_no_face_no_crash(self):
        # Test 5：face = None -> blink=False 且不 crash
        detector = BlinkDetector(BlinkConfig(), face_mesh=_FakeFaceMesh(None))
        for _ in range(5):
            result = detector.process_frame(None)
            self.assertFalse(result.blink_detected)
            self.assertFalse(result.face_detected)
        detector.close()

    def test_full_pipeline_calibration_and_blink(self):
        # 完整链路：先校准（睁眼），再闭眼 3 帧，再睁眼 -> 1 次 blink
        config = BlinkConfig(calibration_frames=3, consec_frames=3, threshold_ratio=0.7)
        open_lm = _make_landmarks(0.30)
        closed_lm = _make_landmarks(0.10)
        sequence = [open_lm] * 3 + [closed_lm] * 3 + [open_lm]

        detector = BlinkDetector(config, face_mesh=_SequenceFaceMesh(sequence))
        blinks = 0
        for _ in sequence:
            result = detector.process_frame(None)
            if result.blink_detected:
                blinks += 1
        self.assertEqual(blinks, 1)
        self.assertEqual(detector.blink_count, 1)
        detector.close()

    def test_constructor_accepts_individual_params(self):
        # 规格说明第 18 节：支持用单独参数构造
        detector = BlinkDetector(
            consec_frames=3,
            calibration_frames=5,
            adaptation_alpha=0.02,
            threshold_ratio=0.6,
            face_mesh=_FakeFaceMesh(None),
        )
        self.assertEqual(detector.config.consec_frames, 3)
        self.assertEqual(detector.config.calibration_frames, 5)
        self.assertAlmostEqual(detector.config.adaptation_alpha, 0.02)
        self.assertAlmostEqual(detector.config.threshold_ratio, 0.6)
        detector.close()


if __name__ == "__main__":
    unittest.main()
