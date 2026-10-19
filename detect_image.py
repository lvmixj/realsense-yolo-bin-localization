#!/usr/bin/env python3
"""Run the bundled garbage-bin model on one RGB image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "bin_v3_yolo11s_960_best.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description="Garbage-bin image test")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not image_path.is_file():
        parser.error(f"image does not exist: {image_path}")
    if not model_path.is_file():
        parser.error(f"model does not exist: {model_path}")

    model = YOLO(str(model_path))
    garbage_bin_ids = [
        class_id
        for class_id, name in model.names.items()
        if str(name).strip().lower() == "garbage bin"
    ]
    if not garbage_bin_ids:
        raise RuntimeError("The model has no 'garbage bin' class")

    device = "0" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    results = model.predict(
        source=str(image_path),
        classes=garbage_bin_ids,
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        verbose=False,
    )
    result = results[0]

    output_path = args.output
    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}_detected.jpg")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), result.plot()):
        raise RuntimeError(f"failed to save output image: {output_path}")

    print(f"Model: {model_path}")
    print(f"Input: {image_path}")
    print(f"Detections: {len(result.boxes)}")
    for index, (confidence, box) in enumerate(
        zip(result.boxes.conf.cpu().tolist(), result.boxes.xyxy.cpu().tolist()),
        start=1,
    ):
        rounded_box = tuple(round(float(value), 1) for value in box)
        print(f"  {index}: garbage bin conf={confidence:.3f} bbox={rounded_box}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
