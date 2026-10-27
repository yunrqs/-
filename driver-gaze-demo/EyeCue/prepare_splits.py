import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser(
        description="Create reproducible, group-aware EyeCue data lists"
    )
    parser.add_argument("--labels", default="../labeling.csv")
    parser.add_argument("--video_dir", default="../all_video_raw_resize")
    parser.add_argument("--gaze_dir", default="../all_gaze_coordinate")
    parser.add_argument("--output_dir", default="splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    gaze_dir = Path(args.gaze_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    group_labels = defaultdict(Counter)
    excluded_wrong = 0
    excluded_missing = 0
    seen = set()

    with open(args.labels, newline="", encoding="utf-8-sig") as csv_file:
        for row in csv.DictReader(csv_file):
            video_name = PurePosixPath(row["video"].replace("\\", "/")).name
            attention = row["attention"].strip().casefold()

            if attention == "wrong video":
                excluded_wrong += 1
                continue
            if attention not in {"attentive", "distracted"}:
                raise ValueError(f"Unknown attention label: {row['attention']!r}")

            video_path = video_dir / video_name
            gaze_path = gaze_dir / f"{Path(video_name).stem}.txt"
            if not video_path.is_file() or not gaze_path.is_file():
                excluded_missing += 1
                continue
            if video_name in seen:
                raise ValueError(f"Duplicate label row for {video_name}")
            seen.add(video_name)

            label = 0 if attention == "attentive" else 1
            group = video_name.split("_pure_hm_", 1)[0]
            groups[group].append((video_path, gaze_path, label))
            group_labels[group][label] += 1

    if not groups:
        raise ValueError("No labeled video/gaze pairs were found")

    fractions = {"train": 0.70, "val": 0.15, "test": 0.15}
    total_by_label = Counter(
        label for samples in groups.values() for _, _, label in samples
    )
    targets = {
        split: {label: total * fraction for label, total in total_by_label.items()}
        for split, fraction in fractions.items()
    }
    assigned_counts = {split: Counter() for split in fractions}

    rng = random.Random(args.seed)
    group_names = list(groups)
    rng.shuffle(group_names)
    group_names.sort(key=lambda name: -len(groups[name]))

    split_names = list(fractions)
    assignment = {}
    for group in group_names:
        labels = group_labels[group]
        best_split = None
        best_score = None
        for split in split_names:
            score = 0.0
            for candidate in split_names:
                for label, total in total_by_label.items():
                    count = assigned_counts[candidate][label]
                    if candidate == split:
                        count += labels[label]
                    target = targets[candidate][label]
                    score += ((count - target) / max(target, 1.0)) ** 2
            if best_score is None or score < best_score:
                best_split = split
                best_score = score

        assignment[group] = best_split
        assigned_counts[best_split].update(labels)

    for split in fractions:
        split_groups = [group for group, target in assignment.items() if target == split]
        output_path = output_dir / f"{split}.txt"
        with output_path.open("w", encoding="utf-8", newline="\n") as output:
            for group in split_groups:
                for video_path, gaze_path, label in groups[group]:
                    output.write(f"{video_path.as_posix()} {gaze_path.as_posix()} {label}\n")

        print(
            f"{split}: {assigned_counts[split].total()} samples, "
            f"attentive={assigned_counts[split][0]}, "
            f"distracted={assigned_counts[split][1]}, "
            f"groups={len(split_groups)}"
        )

    print(f"Excluded wrong-video annotations: {excluded_wrong}")
    print(f"Excluded rows without local video/gaze pair: {excluded_missing}")
    print(f"Created lists in: {output_dir}")
    print("Split is a custom group-aware split, not an official paper split.")


if __name__ == "__main__":
    main()