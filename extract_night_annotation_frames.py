#!/usr/bin/env python3
"""从 RealSense Viewer DB3 录包均匀抽取 RGB 帧，供夜间垃圾桶标注使用。

输出目录使用标准 YOLO 结构：``images/`` 保存 JPEG，``labels/`` 由本目录的
``label_bin_boxes.py`` 写入。抽帧清单会保留每张图对应的录包、原始帧号和时间戳，
便于后续按录包而非随机帧划分 train/valid/test，避免相邻帧数据泄漏。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import struct
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def align4(value: int) -> int:
    return (value + 3) & ~3


def decode_rgb8_image(message: bytes) -> np.ndarray:
    """Decode the compact CDR sensor_msgs/msg/Image payload written by Viewer."""

    if len(message) < 32:
        raise ValueError("CDR image message is truncated")

    offset = 4  # CDR encapsulation
    offset += 8  # builtin_interfaces/msg/Time
    frame_id_length = struct.unpack_from("<I", message, offset)[0]
    offset += 4 + frame_id_length
    offset = align4(offset)

    height, width = struct.unpack_from("<II", message, offset)
    offset += 8
    encoding_length = struct.unpack_from("<I", message, offset)[0]
    offset += 4
    encoding_end = offset + encoding_length
    if encoding_length == 0 or encoding_end > len(message):
        raise ValueError("CDR image encoding is invalid")
    encoding = message[offset : encoding_end - 1].decode("ascii").lower()
    offset = encoding_end

    if offset + 1 > len(message):
        raise ValueError("CDR image endianness is missing")
    offset += 1
    offset = align4(offset)
    step = struct.unpack_from("<I", message, offset)[0]
    offset += 4
    data_length = struct.unpack_from("<I", message, offset)[0]
    offset += 4
    data_end = offset + data_length
    if data_end > len(message) or data_length != step * height:
        raise ValueError("CDR image payload size is invalid")
    if encoding != "rgb8" or step < width * 3:
        raise ValueError(f"Expected rgb8 image, got encoding={encoding}, step={step}")

    image = np.frombuffer(message[offset:data_end], dtype=np.uint8)
    return image.reshape(height, step)[:, : width * 3].reshape(height, width, 3)


def sample_offsets(frame_count: int, requested_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    count = min(frame_count, requested_count)
    if count == 1:
        return [0]
    return sorted(
        {
            round(index * (frame_count - 1) / (count - 1))
            for index in range(count)
        }
    )


def safe_stem(path: Path) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in path.stem
    )


def extract_recording(
    recording: Path,
    image_dir: Path,
    frames_per_recording: int,
    jpeg_quality: int,
) -> Iterable[dict[str, object]]:
    with sqlite3.connect(f"file:{recording}?mode=ro", uri=True) as connection:
        topics = dict(connection.execute("SELECT name, id FROM topics"))
        if COLOR_TOPIC not in topics:
            raise ValueError(f"{recording}: missing color topic {COLOR_TOPIC}")
        topic_id = int(topics[COLOR_TOPIC])
        frame_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM messages WHERE topic_id = ?", (topic_id,)
            ).fetchone()[0]
        )
        offsets = sample_offsets(frame_count, frames_per_recording)
        if not offsets:
            raise ValueError(f"{recording}: color topic has no frames")

        # 先只扫描轻量的 rowid/timestamp 索引，再读取选中的 BLOB。不能对每个
        # 样本使用 ``OFFSET``：SQLite 会为每一个 offset 反复扫描前面的帧，数 GB
        # 的录包会因此慢几个数量级。
        selected_offsets = set(offsets)
        selected_rows: dict[int, tuple[int, int]] = {}
        for offset, (rowid, timestamp_ns) in enumerate(
            connection.execute(
                "SELECT rowid, timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp",
                (topic_id,),
            )
        ):
            if offset in selected_offsets:
                selected_rows[offset] = (int(rowid), int(timestamp_ns))

        if len(selected_rows) != len(offsets):
            raise RuntimeError(f"{recording}: could not resolve all selected frame rows")

        for offset in offsets:
            rowid, timestamp_ns = selected_rows[offset]
            row = connection.execute(
                "SELECT data FROM messages WHERE rowid = ?", (rowid,)
            ).fetchone()
            if row is None:
                raise RuntimeError(f"{recording}: missing data for frame at offset {offset}")
            message = bytes(row[0])
            rgb = decode_rgb8_image(message)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            filename = f"{safe_stem(recording)}_frame_{offset + 1:06d}.jpg"
            destination = image_dir / filename
            if not cv2.imwrite(
                str(destination), bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            ):
                raise OSError(f"Could not write {destination}")

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            yield {
                "image": f"images/{filename}",
                "label": f"labels/{Path(filename).with_suffix('.txt').name}",
                "recording": str(recording),
                "source_frame": offset + 1,
                "timestamp_ns": timestamp_ns,
                "width": int(bgr.shape[1]),
                "height": int(bgr.shape[0]),
                "brightness_mean": round(float(np.mean(gray)), 3),
                "sharpness_laplacian_variance": round(
                    float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3
                ),
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db3",
        type=Path,
        action="append",
        required=True,
        help="RealSense Viewer DB3 recording; pass once per recording",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-recording", type=int, default=40)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames_per_recording <= 0:
        raise SystemExit("--frames-per-recording must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be in [1, 100]")

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {output}. Choose a new directory."
        )
    image_dir = output / "images"
    label_dir = output / "labels"
    image_dir.mkdir(parents=True, exist_ok=False)
    label_dir.mkdir()

    manifest: list[dict[str, object]] = []
    for raw_recording in args.db3:
        recording = raw_recording.resolve()
        if not recording.is_file():
            raise SystemExit(f"Recording does not exist: {recording}")
        rows = list(
            extract_recording(
                recording,
                image_dir,
                args.frames_per_recording,
                args.jpeg_quality,
            )
        )
        manifest.extend(rows)
        print(f"[Extract] {recording.name}: {len(rows)} frames")

    (output / "manifest.json").write_text(
        json.dumps(
            {
                "class_names": ["garbage bin"],
                "label_format": "YOLO detection: class_id cx cy width height",
                "items": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "README_labeling.txt").write_text(
        "Open this directory with label_bin_boxes.py and draw one or more "
        "garbage-bin rectangles per image. Empty label files are valid when "
        "no garbage bin is visible. Keep this manifest: it preserves the "
        "recording grouping required for a leakage-free train/valid/test split.\n",
        encoding="utf-8",
    )
    print(f"[Done] {len(manifest)} images written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
