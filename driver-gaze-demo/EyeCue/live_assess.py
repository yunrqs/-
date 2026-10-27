import argparse
import csv
import contextlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from infer import build_models
from new_train import select


CLIP_LEN = 8
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 720
DISTRACTION_THRESHOLD = 0.5


def load_live_model(checkpoint: Path, device: torch.device):
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
        torch.set_num_interop_threads(1)
    image_processor = AutoImageProcessor.from_pretrained(
        "facebook/timesformer-base-finetuned-k600",
        local_files_only=True,
    )
    video_encoder, gaze_encoder, semantic, head, metadata = build_models(
        checkpoint,
        device,
        CLIP_LEN,
        output_attentions=False,
        local_files_only=True,
    )
    return image_processor, (video_encoder, gaze_encoder, semantic, head), metadata


def read_gaze_csv(path: Path) -> dict[Path, dict[int, tuple[float, float]]]:
    gaze_by_video: dict[Path, dict[int, tuple[float, float]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"video_path", "frame_index", "gaze_x", "gaze_y"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Gaze CSV must contain columns: {sorted(required)}")
        for row in reader:
            video_path = Path(row["video_path"]).resolve()
            frame_index = int(row["frame_index"])
            point = (float(row["gaze_x"]), float(row["gaze_y"]))
            if not all(math.isfinite(value) for value in point):
                raise ValueError(f"Non-finite gaze coordinate in {path}: {row}")
            gaze_by_video[video_path][frame_index] = point
    if not gaze_by_video:
        raise ValueError(f"No gaze samples found in {path}")
    return gaze_by_video


def read_video_window(
    capture: cv2.VideoCapture, start_index: int, frame_count: int
) -> list[Image.Image]:
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    frames = []
    last_frame = None
    for _ in range(frame_count):
        success, frame = capture.read()
        if success:
            if frame.shape[1] != VIDEO_WIDTH or frame.shape[0] != VIDEO_HEIGHT:
                frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
            last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif last_frame is None:
            raise OSError(f"Could not decode video frame at index {start_index}.")
        frames.append(Image.fromarray(last_frame))
    return frames


def predict_window(
    frames: list[Image.Image],
    gaze_points: list[tuple[float, float]],
    image_processor,
    models,
    device: torch.device,
) -> float:
    video_encoder, gaze_encoder, semantic, head = models
    processed = image_processor(images=[frames], return_tensors="pt")
    pixel_values = processed["pixel_values"].to(device)
    gazes = torch.tensor([gaze_points], dtype=torch.float32, device=device)
    video_features, video_tokens, _ = video_encoder(pixel_values)
    gaze_features, gaze_tokens = gaze_encoder(gazes)
    selected_tokens = select(video_tokens, gazes)
    semantic_features = semantic(selected_tokens, gaze_tokens)
    fused = torch.cat([video_features, gaze_features, semantic_features], dim=1)
    logits = head(fused)
    return float(torch.softmax(logits, dim=1)[0, 1].item())


def run_stream(checkpoint: Path, device: torch.device) -> None:
    with contextlib.redirect_stdout(sys.stderr):
        image_processor, models, metadata = load_live_model(checkpoint, device)
    print(
        json.dumps(
            {
                "ready": True,
                "epoch": metadata["epoch"],
                "val_acc": metadata["val_acc"],
                "device": str(device),
            }
        ),
        flush=True,
    )

    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if request.get("command") == "quit":
                return

            video_path = Path(request["video_path"])
            start_frame = int(request["start_frame"])
            gaze_points = [
                (float(point[0]), float(point[1]))
                for point in request["gaze_points"]
            ]
            if len(gaze_points) != CLIP_LEN:
                raise ValueError(
                    f"Expected {CLIP_LEN} gaze points, got {len(gaze_points)}"
                )

            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise OSError(f"Could not open video: {video_path}")
            try:
                frames = read_video_window(capture, start_frame, CLIP_LEN)
            finally:
                capture.release()

            with contextlib.redirect_stdout(sys.stderr):
                probability = predict_window(
                    frames, gaze_points, image_processor, models, device
                )
            print(
                json.dumps(
                    {
                        "video_path": str(video_path.resolve()),
                        "start_frame": start_frame,
                        "end_frame": start_frame + CLIP_LEN - 1,
                        "probability": probability,
                    }
                ),
                flush=True,
            )
        except Exception as error:
            print(
                json.dumps(
                    {
                        "error": f"{type(error).__name__}: {error}",
                        "start_frame": request.get("start_frame"),
                    }
                ),
                flush=True,
            )


def assess_video(
    video_path: Path,
    gaze: dict[int, tuple[float, float]],
    image_processor,
    models,
    device: torch.device,
) -> list[tuple[int, int, float]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < CLIP_LEN:
            raise ValueError(
                f"{video_path.name} has fewer than {CLIP_LEN} frames."
            )

        window_starts = list(range(0, total_frames - CLIP_LEN + 1, CLIP_LEN))
        final_start = total_frames - CLIP_LEN
        if window_starts[-1] != final_start:
            window_starts.append(final_start)

        results = []
        for start_index in window_starts:
            frame_indices = range(start_index, start_index + CLIP_LEN)
            missing = [index for index in frame_indices if index not in gaze]
            if missing:
                raise ValueError(
                    f"Missing aligned gaze samples for {video_path.name}: "
                    f"frame {missing[0]} (and possibly more)."
                )
            frames = read_video_window(capture, start_index, CLIP_LEN)
            gaze_points = [gaze[index] for index in frame_indices]
            probability = predict_window(
                frames, gaze_points, image_processor, models, device
            )
            results.append((start_index, start_index + CLIP_LEN - 1, probability))
            print(
                f"{video_path.name} frames {start_index}-"
                f"{start_index + CLIP_LEN - 1}: "
                f"P(distraction)={probability:.3f}"
            )
        return results
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EyeCue on videos paired with gaze recorded during playback."
    )
    parser.add_argument("--gaze-csv", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_model.pth"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Read live 8-frame inference requests from stdin as JSON lines.",
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but it is not available.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"EyeCue checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    if args.stream:
        run_stream(args.checkpoint, device)
        return

    if args.gaze_csv is None or args.output_dir is None:
        parser.error("--gaze-csv and --output-dir are required outside stream mode.")
    if not args.gaze_csv.is_file():
        raise FileNotFoundError(f"Gaze CSV not found: {args.gaze_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gaze_by_video = read_gaze_csv(args.gaze_csv)
    image_processor = AutoImageProcessor.from_pretrained(
        "facebook/timesformer-base-finetuned-k600"
    )
    print(f"Loading trained EyeCue checkpoint from {args.checkpoint}")
    video_encoder, gaze_encoder, semantic, head, metadata = build_models(
        args.checkpoint, device, CLIP_LEN, output_attentions=False
    )
    if metadata["epoch"] is not None:
        print(
            f"Checkpoint epoch={metadata['epoch']}, "
            f"validation accuracy={metadata['val_acc']:.2f}%"
        )

    all_window_results = []
    video_results = []
    models = (video_encoder, gaze_encoder, semantic, head)
    for video_path, gaze in sorted(
        gaze_by_video.items(), key=lambda item: item[0].name.casefold()
    ):
        if not video_path.is_file():
            raise FileNotFoundError(f"Video from gaze CSV not found: {video_path}")
        predictions = assess_video(
            video_path, gaze, image_processor, models, device
        )
        mean_probability = float(np.mean([item[2] for item in predictions]))
        distracted = mean_probability >= DISTRACTION_THRESHOLD
        video_results.append(
            {
                "video": str(video_path),
                "windows": len(predictions),
                "mean_probability": mean_probability,
                "predicted_class": int(distracted),
                "predicted_label": (
                    "认知分心" if distracted else "未检测到认知分心"
                ),
            }
        )
        all_window_results.extend(
            (str(video_path), start, end, probability)
            for start, end, probability in predictions
        )

    session_probability = float(
        np.mean([result[3] for result in all_window_results])
    )
    session_distracted = session_probability >= DISTRACTION_THRESHOLD
    session_summary = {
        "videos": len(video_results),
        "windows": len(all_window_results),
        "mean_window_probability": session_probability,
        "predicted_class": int(session_distracted),
        "predicted_label": (
            "认知分心" if session_distracted else "未检测到认知分心"
        ),
        "threshold": DISTRACTION_THRESHOLD,
        "interpretation": (
            "Exploratory model estimate while watching driving videos; "
            "not a validated diagnosis or real-world driving safety measure."
        ),
    }

    with (args.output_dir / "video_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as output:
        writer = csv.DictWriter(output, fieldnames=list(video_results[0].keys()))
        writer.writeheader()
        writer.writerows(video_results)

    with (args.output_dir / "window_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as output:
        writer = csv.writer(output)
        writer.writerow(
            ["video", "start_frame", "end_frame", "probability_distracted"]
        )
        writer.writerows(all_window_results)

    with (args.output_dir / "session_summary.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(session_summary, output, ensure_ascii=False, indent=2)

    print("\nPer-video results:")
    for result in video_results:
        print(
            f"{Path(result['video']).name}: {result['predicted_label']} "
            f"(mean P={result['mean_probability']:.3f}, "
            f"{result['windows']} windows)"
        )
    print(
        f"\nSession estimate: {session_summary['predicted_label']} "
        f"(mean P={session_probability:.3f}; "
        f"{len(video_results)} videos, {len(all_window_results)} windows)"
    )
    print(f"Results saved to {args.output_dir.resolve()}")
    print(
        "This is an exploratory model estimate while watching video, "
        "not a diagnosis or a real-world driving safety assessment."
    )


if __name__ == "__main__":
    main()
