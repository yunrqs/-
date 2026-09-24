"""EAR（Eye Aspect Ratio，眼睛纵横比）计算（规格说明第 5~7、23 节）。

本模块只负责 ``landmark -> EAR``，不处理任何 blink state / baseline 逻辑。
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

# 防止异常 landmark 导致除零（规格说明第 6、23 节）。
EPSILON = 1e-6

# 左右眼 6 个 landmark 索引（规格说明第 5 节）。
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# 每只眼睛 6 个点统一按几何顺序传入：
#   p1 = eye_points[0], p2 = eye_points[1], ..., p6 = eye_points[5]
# 代码不依赖“左眼/右眼”语义，只保证两组 landmark 顺序一致（规格说明第 5 节）。

Point = Tuple[float, float]


def euclidean_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """两点之间的欧氏距离。"""
    a = np.asarray(p1, dtype=float)
    b = np.asarray(p2, dtype=float)
    return float(np.linalg.norm(a - b))


def extract_landmarks(
    landmarks: Sequence[Sequence[float]],
    indices: Sequence[int],
) -> List[Point]:
    """从 468 点 landmarks 中取出指定索引，返回 ``(x, y)`` 列表。

    ``landmarks`` 中每个元素可以是 ``(x, y)`` 或 ``(x, y, z)``，
    这里只取前两维参与 2D EAR 计算。
    """
    return [(float(landmarks[i][0]), float(landmarks[i][1])) for i in indices]


def calculate_ear(eye_points: Sequence[Sequence[float]]) -> Optional[float]:
    """计算单眼 EAR。

    .. math::
        EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

    若水平距离接近 0（异常 landmark），返回 ``None``，而不是抛
    ``ZeroDivisionError`` 或返回 0（0 会被误解释为“完全闭眼”）。
    """
    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = euclidean_distance(p2, p6)
    vertical_2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    if horizontal < EPSILON:
        return None

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_average_ear(
    left_ear: Optional[float],
    right_ear: Optional[float],
) -> Optional[float]:
    """合并左右眼 EAR（规格说明第 7 节）。

    - 两只眼都无效 -> ``None``；
    - 单眼无效 -> 使用另一只眼，避免单眼异常直接导致检测器失效；
    - 都有效 -> 平均。
    """
    if left_ear is None and right_ear is None:
        return None
    if left_ear is None:
        return right_ear
    if right_ear is None:
        return left_ear
    return (left_ear + right_ear) / 2.0
