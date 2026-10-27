import argparse
import csv
import json
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont
from gazefollower import GazeFollower


CLIP_RANGE_PATTERN = re.compile(r"^(?P<scene>.+)_(?P<start>\d+)-(?P<end>\d+)$")
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 720


def draw_chinese_status(frame: np.ndarray, text: str) -> np.ndarray:
    height, width = frame.shape[:2]
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    if not font_path.is_file():
        font_path = Path(r"C:\Windows\Fonts\simhei.ttf")
    if not font_path.is_file():
        raise FileNotFoundError("Could not find a Chinese font (Microsoft YaHei or SimHei).")

    font = ImageFont.truetype(str(font_path), max(20, width // 36))
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    bounds = draw.textbbox((12, 8), text, font=font)
    panel_bottom = max(54, bounds[3] + 10)
    draw.rectangle((0, 0, width, panel_bottom), fill=(0, 0, 0))
    draw.text((12, 8), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def clip_sort_key(path: Path) -> tuple[str, int, str]:
    match = CLIP_RANGE_PATTERN.fullmatch(path.stem)
    if match is None:
        return path.stem.casefold(), -1, path.name.casefold()
    return (
        match.group("scene").casefold(),
        int(match.group("start")),
        path.name.casefold(),
    )


def get_video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"Video has invalid duration metadata: {path}")
        return frame_count / fps
    finally:
        capture.release()


def select_videos(
    video: Path | None,
    input_dir: Path | None,
    min_duration: float,
    longest_only: bool = False,
) -> list[Path]:
    if video is not None:
        candidates = [video]
    else:
        if input_dir is None:
            raise ValueError("Specify either --video or --input-dir.")
        candidates = sorted(input_dir.glob("*.mp4"), key=clip_sort_key)

    selected = []
    for path in candidates:
        if not path.is_file():
            raise FileNotFoundError(f"Video does not exist: {path}")
        duration = get_video_duration(path)
        if duration >= min_duration:
            selected.append((duration, path.resolve()))

    if not selected:
        raise FileNotFoundError(
            f"No MP4 videos at least {min_duration:g} seconds long were found."
        )
    if longest_only:
        return [max(selected, key=lambda item: item[0])[1]]
    return [path for _, path in selected]


class EyeCueStream:
    def __init__(
        self,
        python: Path,
        script: Path,
        checkpoint: Path,
        device: str,
    ) -> None:
        self.process = subprocess.Popen(
            [
                str(python),
                str(script),
                "--stream",
                "--checkpoint",
                str(checkpoint),
                "--device",
                device,
            ],
            cwd=script.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.condition = threading.Condition()
        self.pending: dict | None = None
        self.active_start_frame: int | None = None
        self.latest_result: dict | None = None
        self.error: str | None = None
        self.closing = False
        ready_line = self.process.stdout.readline()
        if not ready_line:
            return_code = self.process.poll()
            raise RuntimeError(
                f"EyeCue inference process exited before loading the model "
                f"(exit code {return_code})."
            )
        ready = json.loads(ready_line)
        if not ready.get("ready"):
            raise RuntimeError(f"EyeCue inference process did not become ready: {ready}")
        print(
            f"EyeCue model ready on {ready['device']} "
            f"(checkpoint epoch {ready.get('epoch')})."
        )
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(
        self,
        video_path: Path,
        start_frame: int,
        gaze_points: list[tuple[float, float]],
    ) -> None:
        request = {
            "video_path": str(video_path),
            "start_frame": start_frame,
            "gaze_points": gaze_points,
        }
        with self.condition:
            if self.error is not None:
                raise RuntimeError(f"EyeCue inference failed: {self.error}")
            self.pending = request
            self.condition.notify()

    def get_result(self) -> dict | None:
        with self.condition:
            if self.error is not None:
                raise RuntimeError(f"EyeCue inference failed: {self.error}")
            return self.latest_result

    def _run(self) -> None:
        try:
            while True:
                with self.condition:
                    while self.pending is None and not self.closing:
                        self.condition.wait()
                    if self.pending is None:
                        return
                    request = self.pending
                    self.pending = None
                    self.active_start_frame = request["start_frame"]

                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
                result_line = self.process.stdout.readline()
                if not result_line:
                    raise RuntimeError(
                        "EyeCue inference process closed its output unexpectedly."
                    )
                result = json.loads(result_line)
                if "error" in result:
                    raise RuntimeError(result["error"])
                result["completed_at"] = time.perf_counter()
                with self.condition:
                    self.latest_result = result
                    self.active_start_frame = None
        except Exception as error:
            with self.condition:
                self.error = str(error)
                self.active_start_frame = None
                self.condition.notify_all()

    def get_state(self) -> tuple[dict | None, bool, int | None]:
        with self.condition:
            if self.error is not None:
                raise RuntimeError(f"EyeCue inference failed: {self.error}")
            start_frame = self.active_start_frame
            if start_frame is None and self.pending is not None:
                start_frame = self.pending["start_frame"]
            is_running = start_frame is not None
            return self.latest_result, is_running, start_frame

    def close(self) -> None:
        with self.condition:
            self.closing = True
            self.condition.notify_all()
        if hasattr(self, "worker"):
            self.worker.join()
        if self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps({"command": "quit"}) + "\n")
                self.process.stdin.flush()
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            self.process.wait()
        if self.error is not None:
            raise RuntimeError(f"EyeCue inference failed: {self.error}")


def fullscreen_viewport(screen_size: tuple[int, int]) -> pygame.Rect:
    screen_width, screen_height = screen_size
    scale = min(screen_width / VIDEO_WIDTH, screen_height / VIDEO_HEIGHT)
    width = round(VIDEO_WIDTH * scale)
    height = round(VIDEO_HEIGHT * scale)
    return pygame.Rect(
        (screen_width - width) // 2,
        (screen_height - height) // 2,
        width,
        height,
    )


def play_video(
    video_path: Path,
    screen: pygame.Surface,
    viewport: pygame.Rect,
    gaze_state: dict[str, tuple[float, float, float] | None],
    gaze_lock: threading.Lock,
    predictor: EyeCueStream,
    gaze_writer: csv.writer,
    prediction_writer: csv.writer,
    annotated_path: Path,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Video has invalid metadata: {video_path}")

    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create annotated video: {annotated_path}")

    clock = pygame.time.Clock()
    gaze_window: deque[tuple[float, float] | None] = deque(maxlen=8)
    last_submitted_frame = -1
    last_recorded_prediction: tuple[int, int] | None = None
    frame_index = 0
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt("Playback window was closed.")
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt("Playback was stopped with Escape.")

            success, frame = capture.read()
            if not success:
                break

            frame_time = time.perf_counter()
            with gaze_lock:
                latest_gaze = gaze_state["latest"]

            point = None
            if latest_gaze is not None and frame_time - latest_gaze[0] <= 0.5:
                screen_x, screen_y = latest_gaze[1], latest_gaze[2]
                gaze_x = (screen_x - viewport.left) * width / viewport.width
                gaze_y = (screen_y - viewport.top) * height / viewport.height
                if 0 <= gaze_x < width and 0 <= gaze_y < height:
                    point = (
                        float(np.clip(gaze_x, 0, VIDEO_WIDTH - 1)),
                        float(np.clip(gaze_y, 0, VIDEO_HEIGHT - 1)),
                    )

            gaze_window.append(point)
            if point is not None:
                cv2.circle(
                    frame,
                    (int(point[0]), int(point[1])),
                    max(10, width // 80),
                    (0, 40, 255),
                    thickness=max(2, width // 400),
                )
                cv2.circle(
                    frame,
                    (int(point[0]), int(point[1])),
                    max(2, width // 250),
                    (255, 255, 255),
                    thickness=-1,
                )

            if (
                len(gaze_window) == 8
                and frame_index >= last_submitted_frame + 8
                and all(sample is not None for sample in gaze_window)
            ):
                predictor.submit(
                    video_path,
                    frame_index - 7,
                    [sample for sample in gaze_window if sample is not None],
                )
                last_submitted_frame = frame_index

            prediction, inference_running, active_start = predictor.get_state()
            if (
                prediction is not None
                and Path(prediction["video_path"]) != video_path.resolve()
            ):
                prediction = None
            if (
                prediction is not None
                and (prediction["start_frame"], prediction["end_frame"])
                != last_recorded_prediction
            ):
                probability = prediction["probability"]
                prediction_writer.writerow(
                    [
                        str(video_path),
                        frame_index,
                        prediction["start_frame"],
                        prediction["end_frame"],
                        f"{probability:.6f}",
                        "认知分心" if probability >= 0.5 else "未检测到认知分心",
                    ]
                )
                last_recorded_prediction = (
                    prediction["start_frame"],
                    prediction["end_frame"],
                )
            if prediction is None:
                if inference_running:
                    status_text = (
                        f"认知分心预测中：视频第 {active_start}–"
                        f"{active_start + 7} 帧"
                    )
                else:
                    status_text = "认知分心预测：等待首个有效结果"
            else:
                probability = prediction["probability"]
                label = (
                    "认知分心"
                    if probability >= 0.5
                    else "未检测到认知分心"
                )
                age = max(0, frame_index - prediction["end_frame"]) / fps
                status_text = (
                    f"预测：{label} | 分心概率 {probability:.0%} | "
                    f"对应视频时间 {prediction['end_frame'] / fps:.1f} 秒 | "
                    f"结果延迟 {age:.1f} 秒"
                )
                if inference_running:
                    status_text += (
                        f" | 正在分析第 {active_start}–{active_start + 7} 帧"
                    )
            if point is None:
                status_text += " | 未检测到注视点，请检查摄像头和校准"
            frame = draw_chinese_status(frame, status_text)

            writer.write(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            if surface.get_size() != viewport.size:
                surface = pygame.transform.smoothscale(surface, viewport.size)

            screen.fill((0, 0, 0))
            screen.blit(surface, viewport.topleft)
            pygame.display.flip()
            if point is None:
                gaze_writer.writerow([str(video_path), frame_index, "", ""])
            else:
                gaze_writer.writerow(
                    [str(video_path), frame_index, f"{point[0]:.3f}", f"{point[1]:.3f}"]
                )
            frame_index += 1
            clock.tick(fps)
    finally:
        capture.release()
        writer.release()


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    eyecue_dir = project_dir.parent / "EyeCue"
    parser = argparse.ArgumentParser(
        description=(
            "Play driving clips full-screen, record calibrated webcam gaze, "
            "and run EyeCue inference."
        )
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--video", type=Path, help="Play one MP4.")
    selection.add_argument(
        "--input-dir",
        type=Path,
        help="Play every qualifying MP4 in a directory instead of one video.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=3.0,
        help="Skip videos shorter than this duration in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=eyecue_dir / "outputs" / "live_assessments",
        help="Directory for gaze samples and EyeCue results.",
    )
    parser.add_argument(
        "--eyecue-python",
        type=Path,
        default=eyecue_dir / ".venv312" / "Scripts" / "python.exe",
        help="Python executable for the EyeCue environment.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=eyecue_dir / "checkpoints" / "best_model.pth",
        help="Trained EyeCue checkpoint.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.min_duration <= 0:
        parser.error("--min-duration must be greater than zero.")
    if not args.eyecue_python.is_file():
        raise FileNotFoundError(f"EyeCue Python executable not found: {args.eyecue_python}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"EyeCue checkpoint not found: {args.checkpoint}")

    if args.video is None and args.input_dir is None:
        parser.error("Specify either --video or --input-dir.")
    else:
        videos = select_videos(args.video, args.input_dir, args.min_duration)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}"
    gaze_csv = args.output_dir / f"gaze_samples_{run_id}.csv"
    result_dir = args.output_dir / f"results_{run_id}"
    result_dir.mkdir(parents=True, exist_ok=False)

    print(
        f"Selected {len(videos)} MP4 video(s), each at least "
        f"{args.min_duration:g}s."
    )
    for video_path in videos:
        print(f"  {video_path.name}: {get_video_duration(video_path):.1f}s")
    print(
        "EyeCue will load before playback so it can infer while the video runs. "
        "Follow the full-screen gaze calibration prompts."
    )
    print(
        "Predictions are updated asynchronously; the overlay shows the last "
        "completed window and its age. Press Esc to stop playback."
    )

    follower = GazeFollower(use_multiprocessing=False)
    gaze_state: dict[str, tuple[float, float, float] | None] = {"latest": None}
    samples_lock = threading.Lock()
    sampling_started = False
    pygame_initialized = False
    predictor = None
    try:
        predictor = EyeCueStream(
            args.eyecue_python,
            eyecue_dir / "live_assess.py",
            args.checkpoint,
            args.device,
        )
        follower.calibrate()
        pygame_initialized = True
        pygame.display.set_mode(tuple(map(int, follower.screen_size)), pygame.FULLSCREEN)
        screen = pygame.display.get_surface()
        if screen is None:
            raise RuntimeError("Could not create a full-screen playback window.")
        if screen.get_size() != tuple(map(int, follower.screen_size)):
            raise RuntimeError(
                "The playback display size differs from the calibration display. "
                "Use the same primary monitor for calibration and playback."
            )

        viewport = fullscreen_viewport(screen.get_size())

        def collect_gaze(_face_info, gaze_info) -> None:
            if not gaze_info.status or gaze_info.filtered_gaze_coordinates is None:
                return
            x, y = gaze_info.filtered_gaze_coordinates[:2]
            if not np.isfinite(x) or not np.isfinite(y):
                return
            with samples_lock:
                gaze_state["latest"] = (time.perf_counter(), float(x), float(y))

        follower.add_subscriber(collect_gaze)
        follower.start_sampling()
        sampling_started = True

        with (
            gaze_csv.open("w", newline="", encoding="utf-8") as gaze_output,
            (result_dir / "live_predictions.csv").open(
                "w", newline="", encoding="utf-8-sig"
            ) as prediction_output,
        ):
            gaze_writer = csv.writer(gaze_output)
            gaze_writer.writerow(["video_path", "frame_index", "gaze_x", "gaze_y"])
            prediction_writer = csv.writer(prediction_output)
            prediction_writer.writerow(
                [
                    "video_path",
                    "display_frame",
                    "window_start_frame",
                    "window_end_frame",
                    "probability_distracted",
                    "predicted_label",
                ]
            )
            for video_path in videos:
                print(f"Playing: {video_path.name}")
                annotated_path = result_dir / f"{video_path.stem}_live.mp4"
                play_video(
                    video_path,
                    screen,
                    viewport,
                    gaze_state,
                    samples_lock,
                    predictor,
                    gaze_writer,
                    prediction_writer,
                    annotated_path,
                )
                gaze_output.flush()
                prediction_output.flush()
    finally:
        if sampling_started:
            follower.stop_sampling()
        follower.release()
        if pygame_initialized:
            pygame.quit()
        if predictor is not None:
            predictor.close()

    print(f"\nAnnotated video saved to: {result_dir.resolve()}")
    print(f"Gaze samples saved to: {gaze_csv.resolve()}")
    if predictor is not None:
        final_result = predictor.get_result()
        if final_result is not None:
            label = (
                "认知分心"
                if final_result["probability"] >= 0.5
                else "未检测到认知分心"
            )
            print(
                f"Latest prediction: {label}, "
                f"P(distraction)={final_result['probability']:.3f}, "
                f"window frames={final_result['start_frame']}-"
                f"{final_result['end_frame']}."
            )
            summary = {
                "video": final_result["video_path"],
                "prediction_window_start_frame": final_result["start_frame"],
                "prediction_window_end_frame": final_result["end_frame"],
                "probability_distracted": final_result["probability"],
                "predicted_label": label,
                "threshold": 0.5,
                "note": (
                    "Exploratory estimate while watching a video; not a "
                    "diagnosis or real-world driving-safety assessment."
                ),
            }
            with (result_dir / "session_summary.json").open(
                "w", encoding="utf-8"
            ) as output:
                json.dump(summary, output, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
