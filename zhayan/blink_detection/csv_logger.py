"""CSV 记录（规格说明第 27 节）。

把每帧的 EAR / baseline / threshold / state / blink 写入 CSV，便于画出
``EAR vs Frame`` 曲线并观察 threshold 与 blink event。
"""

import csv
from typing import Optional, TextIO

from blink_types import BlinkResult


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.6f}"


class BlinkRecorder:
    """把每帧的 :class:`BlinkResult` 追加写入 CSV 文件。"""

    HEADER = [
        "frame_id",
        "timestamp",
        "left_ear",
        "right_ear",
        "ear",
        "baseline",
        "threshold",
        "state",
        "blink",
    ]

    def __init__(self, path: str):
        self._file: TextIO = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADER)

    def record(self, frame_id: int, timestamp: float, result: BlinkResult) -> None:
        """写入一帧记录。``timestamp`` 由调用方提供（便于统一时钟）。"""
        self._writer.writerow(
            [
                frame_id,
                f"{timestamp:.6f}",
                _fmt(result.left_ear),
                _fmt(result.right_ear),
                _fmt(result.ear),
                _fmt(result.baseline),
                _fmt(result.threshold),
                result.state,
                int(result.blink_detected),
            ]
        )

    def close(self) -> None:
        """关闭并刷写文件。"""
        self._file.close()
