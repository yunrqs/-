"""MediaPipe Face Mesh 封装：输入图像 -> landmarks（规格说明第 4 节）。

本模块只负责 MediaPipe 初始化与推理，输出第一张人脸的 468 个归一化 landmark。

实现细节：规格说明第 4 节要求使用 ``mp.solutions.face_mesh.FaceMesh``。较新的
mediapipe（>= 0.10.35 起）已移除旧的 ``mp.solutions`` API，仅保留新的
``mediapipe.tasks`` API。因此这里做了双后端兼容：

1. **legacy 后端**：优先使用 ``mp.solutions.face_mesh``，参数与规格说明第 4 节
   完全一致；
2. **tasks 后端**：当 ``mp.solutions`` 不可用时，回退到
   ``mediapipe.tasks.python.vision.FaceLandmarker``（会按需下载官方
   ``face_landmarker.task`` 模型）。

两个后端返回相同格式的 468 点 ``(x, y, z)`` 归一化 landmark，眼部索引不变。
"""

import os
import urllib.request
from typing import List, Optional, Tuple

import cv2

from config import BlinkConfig

# 官方 face_landmarker.task 模型下载地址（仅 tasks 后端需要）。
_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# VIDEO 模式下按约 30 FPS 递增的时间戳（毫秒）。
_FRAME_INTERVAL_MS = 33

Landmarks = List[Tuple[float, float, float]]


class FaceMeshProcessor:
    """封装 MediaPipe Face Mesh 的初始化与单帧推理。

    对外只暴露 ``process(frame_bgr) -> Optional[Landmarks]``。
    """

    def __init__(self, config: BlinkConfig):
        # 延迟导入 mediapipe：只有真正实例化本类时才需要它，
        # 便于单元测试在不安装 mediapipe 的情况下仍可导入其它模块。
        import mediapipe as mp

        self._mp = mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self._backend = "legacy"
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=config.static_image_mode,
                max_num_faces=config.max_num_faces,
                refine_landmarks=config.refine_landmarks,
                min_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )
        else:
            self._backend = "tasks"
            self._init_tasks(config)

    # ------------------------------------------------------------------ #
    # tasks 后端
    # ------------------------------------------------------------------ #
    def _init_tasks(self, config: BlinkConfig) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = self._resolve_model_path(config.model_asset_path)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=config.max_num_faces,
            # 旧 API 的 detection/tracking confidence 映射到 tasks API；
            # 额外的 presence confidence 复用 detection confidence。
            min_face_detection_confidence=config.min_detection_confidence,
            min_face_presence_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    @staticmethod
    def _resolve_model_path(explicit_path: Optional[str]) -> str:
        """返回 face_landmarker.task 模型路径，必要时下载到本地缓存。"""
        if explicit_path and os.path.exists(explicit_path):
            return explicit_path

        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "blink_detection"
        )
        default_path = os.path.join(cache_dir, "face_landmarker.task")
        if os.path.exists(default_path):
            return default_path

        os.makedirs(cache_dir, exist_ok=True)
        tmp_path = default_path + ".part"
        try:
            # 先下载到临时文件，成功后再原子重命名，避免中断留下损坏的模型文件。
            urllib.request.urlretrieve(_DEFAULT_MODEL_URL, tmp_path)
            os.replace(tmp_path, default_path)
        except Exception as exc:  # noqa: BLE001 - 转成清晰的用户提示
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(
                f"无法下载 Face Landmarker 模型，请手动下载到 {default_path}："
                f"{_DEFAULT_MODEL_URL}\n原始错误：{exc}"
            ) from exc
        return default_path

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def process(self, frame_bgr) -> Optional[Landmarks]:
        """输入一帧 BGR 图像，返回第一张人脸的 468 个 landmark，否则 ``None``。"""
        if frame_bgr is None:
            return None

        # MediaPipe 需要 RGB 输入。
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._backend == "legacy":
            return self._process_legacy(rgb)
        return self._process_tasks(rgb)

    def _process_legacy(self, rgb) -> Optional[Landmarks]:
        results = self._face_mesh.process(rgb)
        if results.multi_face_landmarks is None:
            return None
        # 第一版只处理第一张脸（max_num_faces=1），见规格说明第 24 节。
        face = results.multi_face_landmarks[0]
        return [(lm.x, lm.y, lm.z) for lm in face.landmark]

    def _process_tasks(self, rgb) -> Optional[Landmarks]:
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        self._timestamp_ms += _FRAME_INTERVAL_MS

        if not result.face_landmarks:
            return None
        face = result.face_landmarks[0]
        return [(lm.x, lm.y, lm.z) for lm in face]

    def close(self) -> None:
        """释放 MediaPipe 资源。"""
        if self._backend == "legacy":
            self._face_mesh.close()
        else:
            self._landmarker.close()
