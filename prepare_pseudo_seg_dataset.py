#!/usr/bin/env python3
"""Convert YOLO detection boxes to rectangular pseudo-segmentation labels.

This is a reproducible baseline for cases where only bounding-box labels are
available. It must not be confused with real instance segmentation: every
generated polygon is exactly the source bounding box.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


CLASS_NAMES = {
    0: "garbage",
    1: "garbage bin",
    2: "overflow",
    3: "person",
}
SPLITS = ("train", "valid", "test")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def box_to_polygon(fields: list[str], source: Path) -> str:
    if len(fields) != 5:
        raise ValueError(
            f"{source}: expected detection label with 5 fields, got {len(fields)}"
        )
    class_id = int(fields[0])
    if class_id not in CLASS_NAMES:
        raise ValueError(f"{source}: unexpected class id {class_id}")
    cx, cy, width, height = (float(value) for value in fields[1:])
    if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
        raise ValueError(f"{source}: coordinates must be in [0, 1]")
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"{source}: box width and height must be positive")

    x1 = max(0.0, cx - width / 2.0)
    y1 = max(0.0, cy - height / 2.0)
    x2 = min(1.0, cx + width / 2.0)
    y2 = min(1.0, cy + height / 2.0)
    polygon = (x1, y1, x2, y1, x2, y2, x1, y2)
    values = " ".join(f"{value:.8f}" for value in polygon)
    return f"{class_id} {values}"


def convert_label(source: Path) -> str:
    lines = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            lines.append(box_to_polygon(line.split(), source))
        except ValueError as error:
            raise ValueError(f"{error} (line {line_number})") from error
    return "\n".join(lines) + ("\n" if lines else "")


def convert_split(source: Path, output: Path, split: str) -> dict[str, int]:
    image_dir = source / split / "images"
    label_dir = source / split / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing {split}/images or {split}/labels under {source}")

    output_image_dir = output / split / "images"
    output_label_dir = output / split / "labels"
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")

    empty_labels = 0
    instances = 0
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {image_path}")
        label_text = convert_label(label_path)
        if not label_text:
            empty_labels += 1
        instances += len(label_text.splitlines())
        link_or_copy(image_path, output_image_dir / image_path.name)
        (output_label_dir / label_path.name).parent.mkdir(parents=True, exist_ok=True)
        (output_label_dir / label_path.name).write_text(label_text, encoding="utf-8")

    return {
        "images": len(image_paths),
        "empty_labels": empty_labels,
        "instances": instances,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Prepared dataset with train/valid/test")
    parser.add_argument("output", type=Path, help="New pseudo-segmentation dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise SystemExit(f"Source directory does not exist: {source}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    summary = {split: convert_split(source, output, split) for split in SPLITS}
    yaml_text = f"""path: {output}
train: train/images
val: valid/images
test: test/images

names:
  0: garbage
  1: garbage bin
  2: overflow
  3: person
"""
    (output / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (output / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "method": "bounding_box_to_rectangle_polygon",
                "warning": "These are pseudo masks, not manually annotated object contours.",
                "splits": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "README.pseudo_seg.txt").write_text(
        "This dataset was generated by converting each YOLO bounding box to a "
        "four-corner rectangle polygon. It is a training/pipeline baseline, "
        "not a substitute for true instance-mask annotation.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"data_yaml={output / 'data.yaml'}")


if __name__ == "__main__":
    main()
