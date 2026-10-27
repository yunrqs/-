"""集中配置：疲劳驾驶识别参数。

所有可调参数都集中在本文件的 :class:`BodyRecognitionConfig` 中，业务代码只读取
``config.xxx``，不在函数内部散落 magic numbers（沿用 blink_detection 的约定）。
"""

from dataclasses import dataclass

from backend.src.config.paths import DEFAULT_BODY_POSE_MODEL, DEFAULT_FACE_PREDICTOR


@dataclass
class BodyRecognitionConfig:
    """疲劳驾驶识别模块的集中配置。

    - predictor_path：dlib 68 点人脸关键点模型路径；
    - mar_threshold：判定为张嘴（打哈欠）的 MAR 阈值；
    - yawn_seconds：连续张嘴超过该秒数即判定为疲劳驾驶；
    - pose_model_path：OpenPose 身体姿态模型路径（上半身肢体动作）；
    - pose_enabled：是否启用上半身肢体动作检测；
    - pose_interval：OpenPose 隔帧检测间隔（每 N 帧推理一次）；
    - pose_device：OpenPose 推理设备（auto/cpu/cuda）；
    - pose_lean_threshold：身体侧倾角绝对值阈值（度），超过即视为打瞌睡侧倾；
    - pose_fatigue_seconds：侧倾持续超过该秒数即判定为肢体疲劳。
    """

    predictor_path: str = str(DEFAULT_FACE_PREDICTOR)
    mar_threshold: float = 0.6
    yawn_seconds: float = 3.0
    pose_model_path: str = str(DEFAULT_BODY_POSE_MODEL)
    pose_enabled: bool = True
    pose_interval: int = 5
    pose_device: str = "auto"
    pose_lean_threshold: float = 15.0
    pose_fatigue_seconds: float = 2.0

    def __post_init__(self) -> None:
        """对关键参数做轻量校验，尽早暴露非法配置。"""
        if self.mar_threshold <= 0.0:
            raise ValueError("mar_threshold 必须 > 0")
        if self.yawn_seconds <= 0.0:
            raise ValueError("yawn_seconds 必须 > 0")
        if self.pose_interval <= 0:
            raise ValueError("pose_interval 必须 > 0")
        if self.pose_device not in ("auto", "cpu", "cuda"):
            raise ValueError("pose_device 必须是 auto/cpu/cuda 之一")
        if self.pose_lean_threshold <= 0.0:
            raise ValueError("pose_lean_threshold 必须 > 0")
        if self.pose_fatigue_seconds <= 0.0:
            raise ValueError("pose_fatigue_seconds 必须 > 0")