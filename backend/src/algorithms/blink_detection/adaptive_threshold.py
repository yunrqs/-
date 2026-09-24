"""自适应 baseline 与动态 threshold（规格说明第 9~13 节）。

职责：

- baseline 初始化（calibration，第 10 节）；
- baseline 缓慢更新（EMA，仅在稳定睁眼时，第 11、13 节）；
- threshold 计算（第 12 节）。

把 baseline 和 threshold 分开，二者是不同概念：

- ``ear_baseline``：当前用户正常睁眼状态下的 EAR；
- ``ear_threshold``：根据 baseline 计算出的闭眼判断阈值。
"""

from typing import List

import numpy as np

from backend.src.algorithms.blink_detection.blink_types import EyeState


class Calibrator:
    """收集有效 EAR 样本，计算初始 baseline。

    使用中位数而非均值，避免眨眼帧和异常 landmark 的影响（规格说明第 10 节）。
    """

    def __init__(self, required_samples: int, robust: bool = False):
        self._required = required_samples
        self._robust = robust
        self._samples: List[float] = []

    def is_ready(self) -> bool:
        """是否已收集到足够样本。"""
        return len(self._samples) >= self._required

    def add_sample(self, ear: float) -> None:
        """加入一个有效 EAR 样本（调用方需保证 ``ear`` 不是 ``None``）。"""
        if not self.is_ready():
            self._samples.append(float(ear))

    def get_baseline(self) -> float:
        """计算 baseline。

        ``paper`` 模式直接取中位数；``robust`` 模式额外做一次基于 MAD 的
        离群点剔除，把明显偏低（疑似闭眼）的样本过滤掉。
        """
        samples = np.asarray(self._samples, dtype=float)
        if self._robust and len(samples) > 1:
            samples = _trim_outliers(samples)
        return float(np.median(samples))


def _trim_outliers(samples: np.ndarray) -> np.ndarray:
    """剔除明显偏低的离群样本（疑似闭眼帧），返回保留的样本。"""
    median = np.median(samples)
    mad = np.median(np.abs(samples - median))
    if mad < 1e-9:
        return samples
    # 1.4826 是把 MAD 缩放到正态分布标准差的无偏因子；只剔除下侧离群点。
    threshold = median - 2.0 * 1.4826 * mad
    keep = samples >= threshold
    return samples[keep] if keep.any() else samples


def update_baseline(
    current_ear: float,
    current_state: EyeState,
    baseline: float,
    alpha: float,
) -> float:
    """缓慢更新 baseline（规格说明第 11 节）。

    只有在确认处于稳定睁眼状态（``EyeState.OPEN``）时才允许更新，否则冻结
    baseline，避免 baseline 跟随一次眨眼快速下降（第 13 节）。

    禁止写成 ``baseline = current_ear``，那会让 baseline 对单帧噪声高度敏感。
    """
    if current_state is EyeState.OPEN:
        return (1.0 - alpha) * baseline + alpha * current_ear
    return baseline


def calculate_threshold(baseline: float, ratio: float) -> float:
    """根据 baseline 计算动态闭眼阈值（规格说明第 12 节）。

    注意：``baseline * ratio`` 只是默认工程参数，**不是论文公式**。若论文正文
    给出了明确的 Modified EAR / Adaptive EAR 公式，应替换这里并保持论文公式原样。
    """
    return baseline * ratio
