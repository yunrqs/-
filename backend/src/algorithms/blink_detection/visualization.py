"""调试可视化（规格说明第 26 节）。

在视频画面上叠加：

- 左眼 6 个 landmark、右眼 6 个 landmark；
- EAR / Baseline / Threshold / State / Blinks / Face 文本。
"""

from typing import List, Optional, Tuple

import cv2

from backend.src.algorithms.blink_detection.blink_types import BlinkResult

_LEFT_COLOR = (0, 255, 0)      # 绿色
_RIGHT_COLOR = (0, 255, 255)   # 黄色
_TEXT_COLOR = (0, 255, 0)


def _fmt(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def draw_debug_overlay(
    frame,
    result: BlinkResult,
    blink_count: int,
    left_eye_points: Optional[List[Tuple[float, float]]] = None,
    right_eye_points: Optional[List[Tuple[float, float]]] = None,
):
    """在 ``frame`` 上叠加调试信息（原地修改并返回）。

    ``left_eye_points`` / ``right_eye_points`` 使用归一化 ``(x, y)`` 坐标
    （取值 [0, 1]），绘制时再乘以画面宽高换算成像素。
    """
    h, w = frame.shape[:2]

    for color, points in ((_LEFT_COLOR, left_eye_points), (_RIGHT_COLOR, right_eye_points)):
        if not points:
            continue
        for (x, y) in points:
            cx = int(round(x * w))
            cy = int(round(y * h))
            cv2.circle(frame, (cx, cy), 2, color, -1)

    lines = [
        f"EAR:       {_fmt(result.ear)}",
        f"Baseline:  {_fmt(result.baseline)}",
        f"Threshold: {_fmt(result.threshold)}",
        f"State:     {result.state}",
        f"Blinks:    {blink_count}",
        f"Face:      {'detected' if result.face_detected else 'missing'}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (10, 28 + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

    return frame
