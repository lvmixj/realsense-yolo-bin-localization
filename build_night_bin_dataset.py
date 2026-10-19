#!/usr/bin/env python3
"""将已标注夜间帧按录包分组，构建单类别 YOLO 检测数据集。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


SPLITS = ("train", "valid", "test")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Annotated directory from extract_night_annotation_frames.py")
    parser.add_argument("output", type=Path, help="New empty YOLO dataset directory")
    for split in SPLITS:
        parser.add_argument(
            f"--{split}-recording",
            action="append",
            default=[],
            help=f"DB3 filename assigned to {split}; pass once per recording",
        )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing {manifest_path}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}")

    split_by_recording: dict[str, str] = {}
    for split in SPLITS:
        for recording in getattr(args, f"{split}_recording"):
            if recording in split_by_recording:
                raise SystemExit(
                    f"Recording assigned more than once: {recording}"
                )
            split_by_recording[recording] = split
    if not split_by_recording:
        raise SystemExit("Assign every recording with --train/--valid/--test-recording")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise SystemExit("Manifest has no items")

    output.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    seen_recordings: set[str] = set()
    for item in items:
        recording = Path(str(item["recording"])).name
        seen_recordings.add(recording)
        split = split_by_recording.get(recording)
        if split is None:
            raise SystemExit(f"No split assigned for recording: {recording}")
        image = source / str(item["image"])
        label = source / str(item["label"])
        if not image.is_file() or not label.is_file():
            raise SystemExit(f"Missing labeled pair: {image} / {label}")
        link_or_copy(image, output / split / "images" / image.name)
        link_or_copy(label, output / split / "labels" / label.name)
        counts[split] += 1

    unreferenced = set(split_by_recording) - seen_recordings
    if unreferenced:
        raise SystemExit(f"Split recordings absent from manifest: {sorted(unreferenced)}")
    if any(counts[split] == 0 for split in SPLITS):
        raise SystemExit(f"Every split needs samples, got {dict(counts)}")

    (output / "data.yaml").write_text(
        "\n".join(
            (
                f"path: {output.as_posix()}",
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                "",
                "names:",
                "  0: garbage bin",
                "",
            )
        ),
        encoding="utf-8",
    )
    (output / "split_manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "split_by_recording": split_by_recording,
                "counts": dict(counts),
                "class_names": ["garbage bin"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"counts": dict(counts), "data_yaml": str(output / "data.yaml")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
