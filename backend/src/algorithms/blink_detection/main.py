"""入口（规格说明第 20、33 节）。

负责：读取摄像头 / 视频文件 -> 调用 :class:`BlinkDetector` -> 显示结果。

用法示例::

    # 摄像头（默认）
    python main.py

    # 摄像头 + 调试 overlay + 保存 CSV
    python main.py --debug --record

    # 视频文件
    python main.py --source demo.mp4 --debug

    # robust 模式
    python main.py --mode robust --debug
"""

import argparse
import time

import cv2

from backend.src.algorithms.blink_detection.blink_detector import BlinkDetector
from backend.src.algorithms.blink_detection.config import BlinkConfig
from backend.src.algorithms.blink_detection.csv_logger import BlinkRecorder
from backend.src.algorithms.blink_detection.visualization import draw_debug_overlay


def _resolve_source(source_arg: str):
    """把 ``"0"`` 解析为摄像头索引，其它字符串视为视频文件路径。"""
    try:
        return int(source_arg)
    except ValueError:
        return source_arg


def build_config(args: argparse.Namespace) -> BlinkConfig:
    """根据命令行参数构造 :class:`BlinkConfig`。"""
    return BlinkConfig(
        mode=args.mode,
        debug=args.debug,
        record=args.record,
        csv_path=args.csv,
        calibration_frames=args.calibration_frames,
        consec_frames=args.consec_frames,
        adaptation_alpha=args.adaptation_alpha,
        threshold_ratio=args.threshold_ratio,
    )


def run(source, config: BlinkConfig) -> None:
    """主循环：逐帧读取、检测、可视化、记录。"""
    detector = BlinkDetector(config)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        detector.close()
        raise RuntimeError(f"无法打开视频源: {source}")

    recorder = BlinkRecorder(config.csv_path) if config.record else None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = detector.process_frame(frame)

            if config.debug:
                draw_debug_overlay(
                    frame,
                    result,
                    detector.blink_count,
                    detector.last_left_eye_points,
                    detector.last_right_eye_points,
                )

            if recorder is not None:
                recorder.record(detector.frame_id, time.time(), result)

            if result.blink_detected:
                print(f"[frame {detector.frame_id}] Blink detected")

            cv2.imshow("Blink Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        if recorder is not None:
            recorder.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="眨眼检测模块")
    parser.add_argument(
        "--source",
        default="0",
        help="摄像头索引（默认 0）或视频文件路径",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "robust"],
        default="paper",
        help="paper=论文复现；robust=工程增强（默认 paper）",
    )
    parser.add_argument("--debug", action="store_true", help="显示调试 overlay")
    parser.add_argument("--record", action="store_true", help="保存 CSV")
    parser.add_argument("--csv", default="blink_log.csv", help="CSV 输出路径")
    parser.add_argument("--calibration-frames", type=int, default=60)
    parser.add_argument("--consec-frames", type=int, default=3)
    parser.add_argument("--adaptation-alpha", type=float, default=0.01)
    parser.add_argument("--threshold-ratio", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(args)
    run(_resolve_source(args.source), config)


if __name__ == "__main__":
    main()
