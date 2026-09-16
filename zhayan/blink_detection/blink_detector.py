"""眨眼检测核心：EAR -> threshold -> 状态机 -> blink event（规格说明第 14~24、28 节）。

对外只暴露 :class:`BlinkDetector`，上层每次调用 ``process_frame(frame)`` 得到
一个 :class:`BlinkResult`。
"""

from typing import List, Optional, Tuple

from adaptive_threshold import (
    Calibrator,
    calculate_threshold,
    update_baseline,
)
from blink_types import BlinkResult, EyeState
from config import BlinkConfig
from ear import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    calculate_average_ear,
    calculate_ear,
    extract_landmarks,
)
from face_mesh import FaceMeshProcessor


class BlinkStateMachine:
    """基于连续帧的眨眼时序状态机（规格说明第 15~17 节）。

    状态转移：:

        UNKNOWN -> OPEN -> (EAR < threshold) CLOSING
                -> (连续低 EAR >= consec_frames) CLOSED
                -> (EAR >= threshold) OPEN + blink event

    通过 ``consec_frames`` 消除单帧 landmark 抖动误报；一次持续闭眼只在恢复
    睁眼时产生**一次** blink event。
    """

    def __init__(self, consec_frames: int):
        self._consec_frames = consec_frames
        self.state: EyeState = EyeState.UNKNOWN
        self._low_counter: int = 0

    def reset(self) -> None:
        """重置为 ``UNKNOWN``（用于人脸长时间丢失时，规格说明第 22 节）。"""
        self.state = EyeState.UNKNOWN
        self._low_counter = 0

    def enter_open(self) -> None:
        """显式进入 ``OPEN``（校准完成后，用户已连续睁眼若干帧）。"""
        self.state = EyeState.OPEN
        self._low_counter = 0

    def is_stable_open(self) -> bool:
        """是否处于稳定睁眼状态（用于决定是否允许 baseline 更新）。"""
        return self.state is EyeState.OPEN

    def update(self, ear: float, threshold: float) -> bool:
        """输入当前 EAR 与阈值，返回本帧是否产生一次 blink event。"""
        blink = False
        low = ear < threshold

        if self.state is EyeState.UNKNOWN:
            # 首次观测：按规格说明第 15 节的 UNKNOWN -> OPEN 转移。
            self.state = EyeState.OPEN
            self._low_counter = 0
            return False

        if self.state is EyeState.OPEN:
            if low:
                self.state = EyeState.CLOSING
                self._low_counter = 1

        elif self.state is EyeState.CLOSING:
            if low:
                self._low_counter += 1
                if self._low_counter >= self._consec_frames:
                    self.state = EyeState.CLOSED
            else:
                # 单帧抖动：未达到连续帧数就恢复，回到 OPEN，不产生 blink。
                self.state = EyeState.OPEN
                self._low_counter = 0

        elif self.state is EyeState.CLOSED:
            if not low:
                # 恢复睁眼：产生一次且仅一次 blink event。
                self.state = EyeState.OPEN
                self._low_counter = 0
                blink = True

        return blink


class BlinkDetector:
    """眨眼检测器。

    支持两种构造方式：

    - ``BlinkDetector(BlinkConfig(...))``（规格说明第 36 节）；
    - ``BlinkDetector(consec_frames=3, calibration_frames=60, ...)``
      （规格说明第 18 节）。
    """

    def __init__(
        self,
        config: Optional[BlinkConfig] = None,
        face_mesh=None,
        **kwargs,
    ):
        if config is not None and kwargs:
            raise ValueError("不能同时提供 config 与单独参数")
        self.config = config if config is not None else BlinkConfig(**kwargs)
        self._robust = self.config.mode == "robust"

        # 允许注入 fake face mesh，便于单元测试（不依赖 mediapipe）。
        self.face_mesh = (
            face_mesh if face_mesh is not None else FaceMeshProcessor(self.config)
        )

        self.calibrator = Calibrator(
            self.config.calibration_frames, robust=self._robust
        )
        self.state_machine = BlinkStateMachine(self.config.consec_frames)

        self.baseline: Optional[float] = None
        self._blink_count: int = 0
        self._frame_id: int = 0
        self._no_face_counter: int = 0
        self._smoothed_ear: Optional[float] = None

        # 最近一帧的左右眼 landmark（归一化坐标），供可视化使用。
        self.last_left_eye_points: List[Tuple[float, float]] = []
        self.last_right_eye_points: List[Tuple[float, float]] = []

    # ------------------------------------------------------------------ #
    # 只读属性
    # ------------------------------------------------------------------ #
    @property
    def blink_count(self) -> int:
        """累计眨眼次数（规格说明第 26 节的 Blinks 计数）。"""
        return self._blink_count

    @property
    def frame_id(self) -> int:
        """已处理帧序号（从 1 开始）。"""
        return self._frame_id

    # ------------------------------------------------------------------ #
    # 核心接口
    # ------------------------------------------------------------------ #
    def process_frame(self, frame) -> BlinkResult:
        """输入一帧 BGR 图像，返回 :class:`BlinkResult`。

        严格按规格说明第 21 节规定的顺序执行（人脸 -> 提取眼睛 -> EAR ->
        校准 -> 阈值 -> 状态机 -> blink event -> 返回结果）。
        """
        self._frame_id += 1

        # Step 1~5：人脸 landmark
        landmarks = self.face_mesh.process(frame)
        if landmarks is None:
            return self._handle_no_face()

        self._no_face_counter = 0

        # Step 6：提取左右眼 landmark
        left_points = extract_landmarks(landmarks, LEFT_EYE_INDICES)
        right_points = extract_landmarks(landmarks, RIGHT_EYE_INDICES)
        self.last_left_eye_points = left_points
        self.last_right_eye_points = right_points

        # Step 7~8：左右 EAR
        left_ear = calculate_ear(left_points)
        right_ear = calculate_ear(right_points)

        # Step 9：平均 EAR
        ear = calculate_average_ear(left_ear, right_ear)

        if ear is None:
            return self._make_result(
                blink_detected=False,
                face_detected=True,
                left_ear=left_ear,
                right_ear=right_ear,
                ear=None,
                baseline=self.baseline,
                threshold=None,
                state=self.state_machine.state.name,
            )

        # robust 模式：对 EAR 做 EMA 平滑，减少单帧抖动。
        ear = self._smooth_ear(ear)

        # Step 10：校准 / baseline 初始化
        if not self.calibrator.is_ready():
            self.calibrator.add_sample(ear)
            if self.calibrator.is_ready():
                self.baseline = self.calibrator.get_baseline()
                # 校准意味着用户已连续睁眼若干帧，可安全进入 OPEN。
                self.state_machine.enter_open()
            return self._make_result(
                blink_detected=False,
                face_detected=True,
                left_ear=left_ear,
                right_ear=right_ear,
                ear=ear,
                baseline=self.baseline,
                threshold=None,
                state="CALIBRATING",
            )

        # Step 11：自适应 threshold
        threshold = calculate_threshold(self.baseline, self.config.threshold_ratio)

        # Step 12~13：状态机 + blink event
        blink_detected = self.state_machine.update(ear, threshold)
        if blink_detected:
            self._blink_count += 1

        # Step 10（续）：缓慢更新 baseline（第 11、13 节）。
        # 只在稳定睁眼时更新；闭眼期间冻结 baseline。
        if self.state_machine.is_stable_open():
            self.baseline = update_baseline(
                current_ear=ear,
                current_state=self.state_machine.state,
                baseline=self.baseline,
                alpha=self.config.adaptation_alpha,
            )

        return self._make_result(
            blink_detected=blink_detected,
            face_detected=True,
            left_ear=left_ear,
            right_ear=right_ear,
            ear=ear,
            baseline=self.baseline,
            threshold=threshold,
            state=self.state_machine.state.name,
        )

    def reset(self) -> None:
        """重置整个检测器（baseline、状态机与计数）。"""
        self.baseline = None
        self.calibrator = Calibrator(
            self.config.calibration_frames, robust=self._robust
        )
        self.state_machine.reset()
        self._blink_count = 0
        self._no_face_counter = 0
        self._smoothed_ear = None

    def close(self) -> None:
        """释放底层 Face Mesh 资源。"""
        self.face_mesh.close()

    # ------------------------------------------------------------------ #
    # 内部辅助
    # ------------------------------------------------------------------ #
    def _smooth_ear(self, ear: float) -> float:
        """robust 模式下的 EAR EMA 平滑（规格说明第 30 节）。"""
        if not self._robust or self.config.ear_smoothing_alpha <= 0.0:
            return ear
        if self._smoothed_ear is None:
            self._smoothed_ear = ear
        else:
            alpha = self.config.ear_smoothing_alpha
            self._smoothed_ear = alpha * ear + (1.0 - alpha) * self._smoothed_ear
        return self._smoothed_ear

    def _handle_no_face(self) -> BlinkResult:
        """无人脸处理（规格说明第 22 节）。

        不产生 blink；连续丢失超过 ``max_no_face_frames`` 后重置状态机，
        但保留 baseline，避免用户短暂离开后需要完全重新校准。
        """
        self._no_face_counter += 1
        if self._no_face_counter >= self.config.max_no_face_frames:
            self.state_machine.reset()
        return self._make_result(
            blink_detected=False,
            face_detected=False,
            left_ear=None,
            right_ear=None,
            ear=None,
            baseline=self.baseline,
            threshold=None,
            state=self.state_machine.state.name,
        )

    @staticmethod
    def _make_result(
        blink_detected: bool,
        face_detected: bool,
        left_ear: Optional[float],
        right_ear: Optional[float],
        ear: Optional[float],
        baseline: Optional[float],
        threshold: Optional[float],
        state: str,
    ) -> BlinkResult:
        return BlinkResult(
            blink_detected=blink_detected,
            face_detected=face_detected,
            left_ear=left_ear,
            right_ear=right_ear,
            ear=ear,
            baseline=baseline,
            threshold=threshold,
            state=state,
        )
