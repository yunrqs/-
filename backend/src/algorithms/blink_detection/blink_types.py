"""基础类型定义：``EyeState`` 与 ``BlinkResult``（规格说明第 15、19 节）。

说明：规格说明第 20 节建议的文件名为 ``types.py``，但 ``types`` 与 Python
标准库模块同名，直接命名会导致 ``import types`` 被本模块遮蔽，进而破坏
``typing`` / ``dataclasses`` 等标准库的正常导入。为避免该冲突，这里命名为
``blink_types.py``，仅做文件名调整，职责与接口保持不变。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EyeState(Enum):
    """眨眼时序状态机的状态（规格说明第 15 节）。"""

    UNKNOWN = 0
    OPEN = 1
    CLOSING = 2
    CLOSED = 3


@dataclass
class BlinkResult:
    """``BlinkDetector.process_frame()`` 的单帧输出（规格说明第 19 节）。

    上层既可以读取 ``blink_detected`` 触发业务逻辑，也可以读取
    ``ear`` / ``baseline`` / ``threshold`` / ``state`` 等字段做调试与论文实验。
    """

    blink_detected: bool
    face_detected: bool
    left_ear: Optional[float]
    right_ear: Optional[float]
    ear: Optional[float]
    baseline: Optional[float]
    threshold: Optional[float]
    state: str
