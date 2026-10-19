#!/usr/bin/env python3
"""用 OpenCV 为夜间训练集标注垃圾桶框，写出 YOLO detection 标签。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
WINDOW_NAME = "Night bin labeling"


class BinBoxLabeler:
    def __init__(self, dataset: Path, max_width: int, max_height: int) -> None:
        self.dataset = dataset.resolve()
        self.image_dir = self.dataset / "images"
        self.label_dir = self.dataset / "labels"
        if not self.image_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError("Dataset must contain images/ and labels/ directories")
        self.images = sorted(
            path for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.images:
            raise ValueError(f"No images found in {self.image_dir}")

        self.max_width = max_width
        self.max_height = max_height
        self.index = 0
        self.image = None
        self.boxes: list[tuple[int, int, int, int]] = []
        self.drag_start: tuple[int, int] | None = None
        self.drag_current: tuple[int, int] | None = None
        self.scale = 1.0

    @property
    def current_path(self) -> Path:
        return self.images[self.index]

    @property
    def label_path(self) -> Path:
        return self.label_dir / f"{self.current_path.stem}.txt"

    def load_current(self) -> None:
        image = cv2.imread(str(self.current_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {self.current_path}")
        self.image = image
        height, width = image.shape[:2]
        self.scale = min(1.0, self.max_width / width, self.max_height / height)
        self.boxes = self.read_boxes(width, height)
        self.drag_start = None
        self.drag_current = None

    def read_boxes(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        if not self.label_path.is_file():
            return []
        boxes = []
        for line_number, line in enumerate(
            self.label_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5 or fields[0] != "0":
                raise ValueError(f"Invalid label at {self.label_path}:{line_number}")
            cx, cy, box_w, box_h = (float(value) for value in fields[1:])
            x1 = round((cx - box_w / 2.0) * width)
            y1 = round((cy - box_h / 2.0) * height)
            x2 = round((cx + box_w / 2.0) * width)
            y2 = round((cy + box_h / 2.0) * height)
            boxes.append((max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)))
        return boxes

    def save_current(self) -> None:
        assert self.image is not None
        height, width = self.image.shape[:2]
        lines = []
        for x1, y1, x2, y2 in self.boxes:
            x1, x2 = sorted((max(0, x1), min(width - 1, x2)))
            y1, y2 = sorted((max(0, y1), min(height - 1, y2)))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            cx = ((x1 + x2) / 2.0) / width
            cy = ((y1 + y2) / 2.0) / height
            box_w = (x2 - x1) / width
            box_h = (y2 - y1) / height
            lines.append(f"0 {cx:.8f} {cy:.8f} {box_w:.8f} {box_h:.8f}")
        self.label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def display_to_image(self, x: int, y: int) -> tuple[int, int]:
        assert self.image is not None
        height, width = self.image.shape[:2]
        return (
            min(width - 1, max(0, round(x / self.scale))),
            min(height - 1, max(0, round(y / self.scale))),
        )

    def draw(self) -> None:
        assert self.image is not None
        canvas = self.image.copy()
        for x1, y1, x2, y2 in self.boxes:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(canvas, "garbage bin", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        if self.drag_start is not None and self.drag_current is not None:
            cv2.rectangle(canvas, self.drag_start, self.drag_current, (0, 255, 255), 2)
        status = (
            f"{self.index + 1}/{len(self.images)} | boxes={len(self.boxes)} | "
            "drag: add, Backspace: undo, C: clear, N/P: save+next/previous, Q: save+quit"
        )
        cv2.putText(canvas, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2, cv2.LINE_AA)
        if self.scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(WINDOW_NAME, canvas)

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if self.image is None:
            return
        point = self.display_to_image(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = point
            self.drag_current = point
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_current = point
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            x1, y1 = self.drag_start
            x2, y2 = point
            if abs(x2 - x1) >= 4 and abs(y2 - y1) >= 4:
                self.boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
            self.drag_start = None
            self.drag_current = None

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        self.load_current()
        while True:
            self.draw()
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key in (ord("q"), ord("Q"), 27):
                self.save_current()
                break
            if key in (8, 127):
                if self.boxes:
                    self.boxes.pop()
            elif key in (ord("c"), ord("C")):
                self.boxes.clear()
            elif key in (ord("n"), ord("N"), ord("d"), ord("D")):
                self.save_current()
                self.index = min(self.index + 1, len(self.images) - 1)
                self.load_current()
            elif key in (ord("p"), ord("P"), ord("a"), ord("A")):
                self.save_current()
                self.index = max(self.index - 1, 0)
                self.load_current()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Directory generated by extract_night_annotation_frames.py")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--max-height", type=int, default=820)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_width <= 0 or args.max_height <= 0:
        raise SystemExit("--max-width and --max-height must be positive")
    BinBoxLabeler(args.dataset, args.max_width, args.max_height).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
