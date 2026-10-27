import argparse
import re
from pathlib import Path

import cv2


OUTPUT_FPS = 30.0
CLIP_RANGE_PATTERN = re.compile(r"^(?P<scene>.+)_(?P<start>\d+)-(?P<end>\d+)$")


def find_continuous_groups(input_dir: Path) -> tuple[list[list[Path]], int]:
    clips_by_scene: dict[str, list[tuple[int, int, Path]]] = {}
    unmatched_count = 0

    for video_path in input_dir.glob("*.mp4"):
        match = CLIP_RANGE_PATTERN.fullmatch(video_path.stem)
        if match is None:
            unmatched_count += 1
            continue

        scene = match.group("scene")
        start = int(match.group("start"))
        end = int(match.group("end"))
        if end < start:
            raise ValueError(f"Invalid frame range in {video_path.name}")
        clips_by_scene.setdefault(scene, []).append((start, end, video_path))

    continuous_groups = []
    for clips in clips_by_scene.values():
        ordered_clips = sorted(clips, key=lambda clip: (clip[0], clip[1], clip[2].name))
        current_group = []
        current_end = None

        for start, end, video_path in ordered_clips:
            if current_group and start != current_end + 1:
                if len(current_group) > 1:
                    continuous_groups.append(current_group)
                current_group = []

            current_group.append(video_path)
            current_end = end

        if len(current_group) > 1:
            continuous_groups.append(current_group)

    continuous_groups.sort(key=lambda group: group[0].name.casefold())
    return continuous_groups, unmatched_count


def concatenate_group(videos: list[Path], output_path: Path) -> int:
    frame_size = None
    frames_written = 0
    writer = None

    try:
        for video_path in videos:
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise OSError(f"Could not open video: {video_path}")

            try:
                source_fps = capture.get(cv2.CAP_PROP_FPS)
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if source_fps <= 0 or width <= 0 or height <= 0:
                    raise ValueError(f"Invalid video metadata: {video_path}")

                current_size = (width, height)
                if frame_size is None:
                    frame_size = current_size
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        OUTPUT_FPS,
                        frame_size,
                    )
                    if not writer.isOpened():
                        raise OSError(f"Could not create output video: {output_path}")
                elif current_size != frame_size:
                    raise ValueError(
                        f"Video dimensions differ in {video_path}: "
                        f"expected {frame_size}, got {current_size}"
                    )

                source_frame_index = 0
                while True:
                    success, frame = capture.read()
                    if not success:
                        break

                    expected_before = round(
                        source_frame_index * OUTPUT_FPS / source_fps
                    )
                    expected_after = round(
                        (source_frame_index + 1) * OUTPUT_FPS / source_fps
                    )
                    for _ in range(expected_after - expected_before):
                        writer.write(frame)
                        frames_written += 1
                    source_frame_index += 1

                if source_frame_index == 0:
                    raise ValueError(f"Video contains no readable frames: {video_path}")
            finally:
                capture.release()
    finally:
        if writer is not None:
            writer.release()

    return frames_written


def concatenate_videos(input_dir: Path, output_dir: Path) -> None:
    continuous_groups, unmatched_count = find_continuous_groups(input_dir)
    if not continuous_groups:
        raise FileNotFoundError(
            f"No continuous groups of clips found in {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, videos in enumerate(continuous_groups, start=1):
        first_match = CLIP_RANGE_PATTERN.fullmatch(videos[0].stem)
        last_match = CLIP_RANGE_PATTERN.fullmatch(videos[-1].stem)
        if first_match is None or last_match is None:
            raise ValueError("A grouped video has an unexpected filename.")

        output_path = output_dir / (
            f"{first_match.group('scene')}_"
            f"{first_match.group('start')}-{last_match.group('end')}.mp4"
        )
        if output_path.exists():
            raise FileExistsError(
                f"Output already exists; choose an empty output directory: {output_path}"
            )

        frames_written = concatenate_group(videos, output_path)
        if index % 25 == 0 or index == len(continuous_groups):
            print(f"Created {index}/{len(continuous_groups)} continuous videos")
        print(f"{output_path.name}: {len(videos)} clips, {frames_written} frames")

    print(f"Saved {len(continuous_groups)} videos to {output_dir.resolve()}")
    if unmatched_count:
        print(
            f"Ignored {unmatched_count} MP4 files without a scene ID and frame range."
        )


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Join adjacent EyeCue driving-video clips from the same scene."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_dir.parent / "all_video_raw_resize",
        help="Directory containing the source MP4 clips.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir.parent / "driving_continuous_groups",
        help="Directory for the concatenated scene videos.",
    )
    args = parser.parse_args()
    concatenate_videos(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
