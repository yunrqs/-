"""集中配置。

所有可调参数都集中在本文件的 :class:`BlinkConfig` 中，业务代码只读取
``config.xxx``，不允许在函数内部散落 magic numbers（规格说明第 25 节）。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BlinkConfig:
    """眨眼检测模块的集中配置。

    参数与规格说明一一对应：

    - MediaPipe Face Mesh 参数（第 4 节）；
    - 校准 / baseline / threshold / 状态机参数（第 10~16、25 节）；
    - 模式选择（第 30 节）；
    - 调试与 CSV 记录（第 26、27 节）。
    """

    # ---- MediaPipe Face Mesh（第 4 节）----
    static_image_mode: bool = False
    max_num_faces: int = 1
    refine_landmarks: bool = False
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # 仅 tasks 后端使用：本地 face_landmarker.task 模型路径；为 None 时按需下载。
    model_asset_path: Optional[str] = None

    # ---- 校准：baseline 初始化（第 10 节）----
    calibration_frames: int = 60

    # ---- 自适应 baseline：slowly adaptive EMA（第 11、13 节）----
    adaptation_alpha: float = 0.01

    # ---- 自适应 threshold（第 12 节）----
    threshold_ratio: float = 0.7

    # ---- 时序状态机：连续帧判定（第 16 节）----
    consec_frames: int = 3

    # ---- 无人脸容错（第 22 节）----
    max_no_face_frames: int = 30

    # ---- 模式：paper（论文复现）/ robust（工程增强）（第 30 节）----
    mode: str = "paper"

    # ---- 工程增强（仅 robust 模式生效）----
    # EAR 的 EMA 平滑系数，0 表示不平滑。
    ear_smoothing_alpha: float = 0.3

    # ---- 调试 / 记录（第 26、27 节）----
    debug: bool = False
    record: bool = False
    csv_path: str = "blink_log.csv"

    def __post_init__(self) -> None:
        """对关键参数做轻量校验，尽早暴露非法配置。"""
        if self.mode not in ("paper", "robust"):
            raise ValueError(f"mode 必须是 'paper' 或 'robust'，得到 {self.mode!r}")
        if not (0.0 < self.adaptation_alpha < 1.0):
            raise ValueError("adaptation_alpha 必须在 (0, 1) 之间")
        if not (0.0 < self.threshold_ratio < 1.0):
            raise ValueError("threshold_ratio 必须在 (0, 1) 之间")
        if self.consec_frames < 1:
            raise ValueError("consec_frames 必须 >= 1")
        if self.calibration_frames < 1:
            raise ValueError("calibration_frames 必须 >= 1")
        if self.max_no_face_frames < 1:
            raise ValueError("max_no_face_frames 必须 >= 1")
        if not (0.0 <= self.ear_smoothing_alpha < 1.0):
            raise ValueError("ear_smoothing_alpha 必须在 [0, 1) 之间")
