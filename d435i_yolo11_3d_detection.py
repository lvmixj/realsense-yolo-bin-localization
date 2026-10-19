#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Intel RealSense D435i + YOLO11 RGB-D 三维目标检测。

深度帧在每一帧对齐到彩色帧；程序使用对齐后彩色帧内参和 D435i depth scale
计算目标在相机坐标系中的 ``(X, Y, Z)``，单位为 mm。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import struct
import sys
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ==============================================================================
# 0. 路径
# ==============================================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 1. 默认配置
# ==============================================================================

# ------------------------------------------------------------------------------
# YOLO
# ------------------------------------------------------------------------------

# 推荐把模型放在当前脚本目录的 models/ 下。
# 如果你的模型文件名不同，只改这里即可。
DEFAULT_MODEL = os.path.join(
    _SCRIPT_DIR,
    "models",
    "bin_v3_yolo11s_960_best.pt",
)

YOLO_IMAGE_SIZE = 960
CONF_THRESHOLD = 0.20
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 20

# ------------------------------------------------------------------------------
# 只识别垃圾桶
#
# 只保留这些类别（按模型 names 匹配，忽略大小写）；其余类别直接丢弃。
# 如果你的模型类别名不同（例如 "trash_bin" / "waste_bin"），在这里补充即可。
# ------------------------------------------------------------------------------

TRASH_CLASS_NAMES = {
    "garbage bin",
    "trash",
    "trash bin",
    "trashcan",
    "trash can",
    "waste bin",
    "litter bin",
    "bin",
    # StreetView-Waste 官方检测数据集的 7 类。
    "default container",
    "green container",
    "blue container",
    "yellow container",
    "biodegradable container",
    "oil container",
    "battery container",
    # 为后续使用本车 D435i 数据按桶身颜色微调预留的类别名。
    "red container",
    "black container",
    "gray container",
    "grey container",
    "brown container",
    "orange container",
    "white container",
}

# device：
#   auto -> 有 CUDA 时用 0，否则 CPU
#   0    -> 第一张 NVIDIA GPU
#   cpu  -> CPU
DEFAULT_DEVICE = "auto"


# ------------------------------------------------------------------------------
# Depth 有效范围，单位 mm
# ------------------------------------------------------------------------------

MIN_DEPTH_MM = 20.0
MAX_DEPTH_MM = 10000.0


# ------------------------------------------------------------------------------
# 环境质量、稳定性和抓取配置
#
# 这些值是可调的起始值，不应被当作特定实验场地的最终标定结果。运行时
# 可以通过命令行覆盖，程序会把实际采用的阈值打印出来。
# ------------------------------------------------------------------------------

LOW_LIGHT_BRIGHTNESS = 45.0
DARK_PIXEL_THRESHOLD = 45.0
DARK_PIXEL_RATIO = 0.50
OVEREXPOSED_PIXEL_THRESHOLD = 245.0
OVEREXPOSED_RATIO = 0.20
LOW_CONTRAST_STD = 25.0
MIN_GLOBAL_DEPTH_VALID_RATIO = 0.35
FOG_SUSPECTED_DEPTH_RATIO = 0.55

MIN_TARGET_DEPTH_VALID_RATIO = 0.35
MAX_TARGET_DEPTH_MAD_MM = 40.0
MAX_FILLED_DEPTH_RATIO = 0.30
STABILITY_WINDOW_FRAMES = 5
MIN_STABLE_FRAMES = 5
MAX_POSITION_STD_MM = 30.0

# GUI 光流跟踪必须先通过的确认条件。仅影响画面显示，不会写入 JSONL，
# 更不会参与抓取决策。30 FPS 下 10 帧约为 0.33 秒。
DISPLAY_TRACK_MAX_MISSED_FRAMES = 10
VISUAL_TRACK_MIN_CONFIDENCE = 0.45
VISUAL_TRACK_MIN_CONFIRMATIONS = 3
VISUAL_TRACK_CONFIRMATION_SECONDS = 1.0
# 检测框与光流预测框融合时，当前检测所占权重。值越小越平滑、但跟随越慢；
# 1.0 表示直接显示原始 YOLO 框。
DISPLAY_TRACK_SMOOTHING_ALPHA = 0.55
VISUAL_TRACK_MIN_BOX_WIDTH_PX = 30
VISUAL_TRACK_MIN_BOX_HEIGHT_PX = 50
VISUAL_TRACK_MIN_ASPECT_RATIO = 0.20
VISUAL_TRACK_MAX_ASPECT_RATIO = 1.50

DEFAULT_MAX_JAW_OPENING_MM = 100.0
DEFAULT_PREGRASP_DISTANCE_MM = 100.0
DEFAULT_GRASP_MARGIN_MM = 10.0
MIN_POINT_CLOUD_POINTS = 12
MIN_RIM_POINT_CLOUD_POINTS = 8
APPROACH_OBSTACLE_MARGIN_MM = 50.0

# D435i 的 848x480@30 是一个适合先行标定的常用起点；Viewer 回放仍使用
# 录包内实际分辨率。
DEFAULT_COLOR_WIDTH = 848
DEFAULT_COLOR_HEIGHT = 480
DEFAULT_DEPTH_WIDTH = 848
DEFAULT_DEPTH_HEIGHT = 480
DEFAULT_MAX_GAIN = 64.0

HARD_ENVIRONMENT_FLAGS = {
    "STRONG_LIGHT",
    "DEPTH_UNRELIABLE",
    "FOG_OR_SCATTERING_SUSPECTED",
    "EXPOSURE_SETTLING",
}


# ------------------------------------------------------------------------------
# Detect 模型中心 ROI
#
# 0.50 表示只取 bbox 中心 50% x 50% 区域做深度估计，减少背景干扰。
# ------------------------------------------------------------------------------

CENTER_ROI_RATIO = 0.50


# ------------------------------------------------------------------------------
# Seg 模型 Mask 边缘腐蚀
#
# 分割 Mask 边缘常混入背景深度。
# 这里轻微向内腐蚀，提高深度稳定性。
#
# 0 = 不腐蚀
# 3 = 使用 3x3 kernel 腐蚀一次
# ------------------------------------------------------------------------------

SEG_MASK_ERODE_KERNEL = 3


# ------------------------------------------------------------------------------
# 深度主峰聚类
# ------------------------------------------------------------------------------

# 直方图 bin 宽度：50 mm
DEPTH_HIST_BIN_MM = 50.0

# 只保留主峰中心 +/- 120 mm 的深度点
DEPTH_CLUSTER_WINDOW_MM = 120.0

# 一个区域至少需要多少个有效深度像素
MIN_VALID_DEPTH_POINTS = 20


# ------------------------------------------------------------------------------
# GUI
# ------------------------------------------------------------------------------

WINDOW_NAME = "D435i + YOLO11 3D Detection"
WINDOW_INITIAL_WIDTH = 960
WINDOW_INITIAL_HEIGHT = 540

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.50
FONT_THICKNESS = 1

ESC_KEY = 27

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)

PALETTE = [
    (255, 255, 255),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
]


# ==============================================================================
# 2. 数据结构
# ==============================================================================

@dataclass
class CameraIntrinsics:
    """RGB 针孔相机内参。"""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class FramePacket:
    """一帧 RGB-D 数据及其来源时间戳。"""

    color_bgr: np.ndarray
    depth_raw: np.ndarray
    timestamp_ns: int


@dataclass(frozen=True)
class QualityConfig:
    """环境、深度和多帧稳定性阈值。数值是起始值，需用现场数据标定。"""

    low_light_brightness: float = LOW_LIGHT_BRIGHTNESS
    dark_pixel_threshold: float = DARK_PIXEL_THRESHOLD
    dark_pixel_ratio: float = DARK_PIXEL_RATIO
    overexposed_pixel_threshold: float = OVEREXPOSED_PIXEL_THRESHOLD
    overexposed_ratio: float = OVEREXPOSED_RATIO
    low_contrast_std: float = LOW_CONTRAST_STD
    min_global_depth_valid_ratio: float = MIN_GLOBAL_DEPTH_VALID_RATIO
    fog_suspected_depth_ratio: float = FOG_SUSPECTED_DEPTH_RATIO
    min_target_depth_valid_ratio: float = MIN_TARGET_DEPTH_VALID_RATIO
    max_target_depth_mad_mm: float = MAX_TARGET_DEPTH_MAD_MM
    stability_window_frames: int = STABILITY_WINDOW_FRAMES
    min_stable_frames: int = MIN_STABLE_FRAMES
    max_position_std_mm: float = MAX_POSITION_STD_MM


@dataclass
class EnvironmentQuality:
    """单帧环境质量评估，所有比例均为 0 到 1。"""

    brightness_mean: float
    dark_pixel_ratio: float
    overexposed_ratio: float
    contrast_std: float
    depth_valid_ratio: float
    flags: Tuple[str, ...]
    status: str
    grasp_safe: bool


@dataclass
class ObjectGeometry:
    """由目标点云估计的局部几何量，坐标在彩色相机光学坐标系。"""

    center_camera_xyz_mm: Tuple[float, float, float]
    vertical_axis_camera: Tuple[float, float, float]
    jaw_axis_camera: Tuple[float, float, float]
    approach_axis_camera: Tuple[float, float, float]
    width_mm: float
    height_mm: float
    depth_mm: float
    top_points_camera_mm: np.ndarray = field(repr=False)


@dataclass
class GraspCandidate:
    """一个抓取候选；姿态矩阵列为夹爪闭合轴、侧向轴、接近轴。"""

    grasp_type: str
    grasp_point_camera_xyz_mm: Tuple[float, float, float]
    pregrasp_point_camera_xyz_mm: Tuple[float, float, float]
    approach_vector_camera: Tuple[float, float, float]
    jaw_axis_camera: Tuple[float, float, float]
    orientation_camera_quaternion_xyzw: Tuple[float, float, float, float]
    required_jaw_opening_mm: float
    score: float
    path_clear: bool


@dataclass
class Extrinsics:
    """相机光学坐标到机器人底座坐标的刚体变换。"""

    rotation_base_from_camera: np.ndarray = field(repr=False)
    translation_base_mm: np.ndarray = field(repr=False)
    source_frame: str = "camera_color_optical_frame"
    target_frame: str = "robot_base"


class RealSenseViewerDb3Reader:
    """读取 RealSense Viewer 导出的 SQLite ``.db3`` 录制文件。

    这类文件虽然使用 SQLite 容器，但不是 ROS 2 ``rosbag2`` 目录：图像和
    相机标定以 CDR 消息保存在固定 topic 中。Viewer 没有写入已对齐深度图，
    因此这里根据文件中的内外参把深度图投影到彩色图坐标系。
    """

    _COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data"
    _DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data"
    _COLOR_INFO_TOPIC = "/device_0/sensor_1/Color_0/camera_info"
    _DEPTH_INFO_TOPIC = "/device_0/sensor_0/Depth_0/camera_info"
    _DEPTH_TO_COLOR_TOPIC = "/device_0/sensor_1/Color_0/tf/ref_0"
    _DEPTH_SCALE_TOPIC = "/device_0/sensor_0/option/Depth_Units/value"

    def __init__(self, path: str) -> None:
        requested_path = os.path.abspath(path)
        if not os.path.isfile(requested_path):
            # Viewer 默认名称是 ``日期_时间.db3``。兼容常见的误写形式
            # ``日期\\_时间.db3``，同时仍只在同一父目录内查找一个确定路径。
            date_directory = os.path.basename(os.path.dirname(requested_path))
            filename = os.path.basename(requested_path).lstrip("_")
            fallback_path = os.path.join(
                os.path.dirname(os.path.dirname(requested_path)),
                f"{date_directory}_{filename}",
            )
            if os.path.isfile(fallback_path):
                requested_path = fallback_path
            else:
                raise FileNotFoundError(f"DB3 recording does not exist: {path}")

        self.path = requested_path
        self.connection = sqlite3.connect(
            f"file:{self.path}?mode=ro",
            uri=True,
        )

        try:
            self._topic_ids = self._load_topic_ids()
            self.color_intrinsics = self._read_intrinsics(self._COLOR_INFO_TOPIC)
            self.depth_intrinsics = self._read_intrinsics(self._DEPTH_INFO_TOPIC)
            self.depth_scale_m = self._read_depth_scale_m()
            self.depth_scale_mm = self.depth_scale_m * 1000.0
            self.rotation, self.translation = self._read_depth_to_color_transform()
            self._color_rows = self._message_rows(self._COLOR_TOPIC)
            self._depth_rows = self._message_rows(self._DEPTH_TOPIC)
            self._depth_x, self._depth_y = np.meshgrid(
                np.arange(self.depth_intrinsics.width, dtype=np.float32),
                np.arange(self.depth_intrinsics.height, dtype=np.float32),
            )
            self.frame_count = 0
        except Exception:
            self.connection.close()
            raise

    @staticmethod
    def _read_cdr_string(message: bytes) -> str:
        """反序列化 std_msgs/msg/String 的 CDR 负载。"""

        if len(message) < 8:
            raise ValueError("CDR string message is truncated")

        length = struct.unpack_from("<I", message, 4)[0]
        end = 8 + length
        if length == 0 or end > len(message):
            raise ValueError("CDR string length is invalid")
        return message[8 : end - 1].decode("utf-8")

    @staticmethod
    def _parse_key_values(text: str) -> dict[str, str]:
        result = {}
        for item in text.split(";"):
            key, separator, value = item.partition("=")
            if separator:
                result[key.strip()] = value.strip()
        return result

    def _load_topic_ids(self) -> dict[str, int]:
        rows = self.connection.execute("SELECT id, name FROM topics").fetchall()
        topic_ids = {name: int(topic_id) for topic_id, name in rows}
        required_topics = (
            self._COLOR_TOPIC,
            self._DEPTH_TOPIC,
            self._COLOR_INFO_TOPIC,
            self._DEPTH_INFO_TOPIC,
            self._DEPTH_TO_COLOR_TOPIC,
            self._DEPTH_SCALE_TOPIC,
        )
        missing = [topic for topic in required_topics if topic not in topic_ids]
        if missing:
            raise ValueError(
                "Not a supported RealSense Viewer DB3 recording; "
                f"missing topics: {', '.join(missing)}"
            )
        return topic_ids

    def _first_message(self, topic: str) -> bytes:
        row = self.connection.execute(
            "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1",
            (self._topic_ids[topic],),
        ).fetchone()
        if row is None:
            raise ValueError(f"DB3 topic has no messages: {topic}")
        return bytes(row[0])

    def _read_intrinsics(self, topic: str) -> CameraIntrinsics:
        values = self._parse_key_values(self._read_cdr_string(self._first_message(topic)))
        try:
            return CameraIntrinsics(
                fx=float(values["fx"]),
                fy=float(values["fy"]),
                cx=float(values["ppx"]),
                cy=float(values["ppy"]),
                width=int(values["width"]),
                height=int(values["height"]),
            )
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid camera intrinsics in {topic}") from error

    def _read_depth_scale_m(self) -> float:
        try:
            scale = float(
                self._read_cdr_string(self._first_message(self._DEPTH_SCALE_TOPIC))
            )
        except ValueError as error:
            raise ValueError("Invalid RealSense depth scale in DB3") from error
        if scale <= 0.0:
            raise ValueError(f"Invalid non-positive RealSense depth scale: {scale}")
        return scale

    def _read_depth_to_color_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        values = self._parse_key_values(
            self._read_cdr_string(self._first_message(self._DEPTH_TO_COLOR_TOPIC))
        )
        try:
            rotation = np.fromstring(values["rotation"], sep=",", dtype=np.float32)
            translation = np.fromstring(
                values["translation"],
                sep=",",
                dtype=np.float32,
            )
        except KeyError as error:
            raise ValueError("DB3 depth-to-color transform is missing") from error
        if rotation.size != 9 or translation.size != 3:
            raise ValueError("DB3 depth-to-color transform has an invalid shape")
        return rotation.reshape(3, 3), translation.reshape(3, 1)

    def _message_rows(self, topic: str) -> Iterator[Tuple[int, bytes]]:
        rows = self.connection.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (self._topic_ids[topic],),
        )
        for timestamp, data in rows:
            yield int(timestamp), bytes(data)

    @staticmethod
    def _decode_image(message: bytes) -> Tuple[np.ndarray, str]:
        """反序列化 sensor_msgs/msg/Image（Viewer 写入的紧凑 CDR 格式）。"""

        if len(message) < 32:
            raise ValueError("CDR image message is truncated")

        def align4(value: int) -> int:
            return (value + 3) & ~3

        offset = 4  # CDR encapsulation header
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
            raise ValueError("CDR image metadata is truncated")
        is_bigendian = bool(message[offset])
        offset = align4(offset + 1)
        if offset + 8 > len(message):
            raise ValueError("CDR image metadata is truncated")
        step, data_length = struct.unpack_from("<II", message, offset)
        offset += 8
        data_end = offset + data_length
        if data_end > len(message) or data_length != step * height:
            raise ValueError("CDR image payload length is invalid")

        if encoding == "rgb8":
            if step < width * 3:
                raise ValueError("RGB image row stride is invalid")
            image = np.frombuffer(message[offset:data_end], dtype=np.uint8)
            image = image.reshape(height, step)[:, : width * 3].reshape(height, width, 3)
        elif encoding == "mono16":
            if step < width * 2:
                raise ValueError("Depth image row stride is invalid")
            dtype = ">u2" if is_bigendian else "<u2"
            image = np.frombuffer(message[offset:data_end], dtype=dtype)
            image = image.reshape(height, step // 2)[:, :width]
            if is_bigendian:
                image = image.astype(np.uint16)
        else:
            raise ValueError(f"Unsupported DB3 image encoding: {encoding}")

        return image, encoding

    def _align_depth_to_color(self, depth_raw: np.ndarray) -> np.ndarray:
        """使用录制文件提供的标定，将深度值 z-buffer 投影到彩色图。"""

        depth_m = depth_raw.astype(np.float32) * self.depth_scale_m
        valid = depth_m > 0.0
        if not np.any(valid):
            return np.zeros(
                (self.color_intrinsics.height, self.color_intrinsics.width),
                dtype=np.uint16,
            )

        z = depth_m[valid]
        xyz_depth = np.vstack(
            (
                (self._depth_x[valid] - self.depth_intrinsics.cx)
                * z
                / self.depth_intrinsics.fx,
                (self._depth_y[valid] - self.depth_intrinsics.cy)
                * z
                / self.depth_intrinsics.fy,
                z,
            )
        )
        xyz_color = self.rotation @ xyz_depth + self.translation
        z_color = xyz_color[2]
        visible = z_color > 0.0
        if not np.any(visible):
            return np.zeros(
                (self.color_intrinsics.height, self.color_intrinsics.width),
                dtype=np.uint16,
            )

        u = np.rint(
            self.color_intrinsics.fx * xyz_color[0, visible] / z_color[visible]
            + self.color_intrinsics.cx
        ).astype(np.int32)
        v = np.rint(
            self.color_intrinsics.fy * xyz_color[1, visible] / z_color[visible]
            + self.color_intrinsics.cy
        ).astype(np.int32)
        in_bounds = (
            (u >= 0)
            & (u < self.color_intrinsics.width)
            & (v >= 0)
            & (v < self.color_intrinsics.height)
        )

        aligned = np.zeros(
            (self.color_intrinsics.height, self.color_intrinsics.width),
            dtype=np.uint16,
        )
        if not np.any(in_bounds):
            return aligned

        raw_depth = depth_raw[valid][visible][in_bounds]
        pixels = v[in_bounds] * self.color_intrinsics.width + u[in_bounds]
        z_buffer = np.full(aligned.size, np.iinfo(np.uint16).max, dtype=np.uint16)
        np.minimum.at(z_buffer, pixels, raw_depth)
        z_buffer[z_buffer == np.iinfo(np.uint16).max] = 0
        return z_buffer.reshape(aligned.shape)

    def next_frame(self) -> Optional[FramePacket]:
        """返回下一对 RGB-D 帧，播完时返回 ``None``。"""

        try:
            color_timestamp, color_message = next(self._color_rows)
            depth_timestamp, depth_message = next(self._depth_rows)
        except StopIteration:
            return None

        color_rgb, color_encoding = self._decode_image(color_message)
        depth_raw, depth_encoding = self._decode_image(depth_message)
        if color_encoding != "rgb8" or depth_encoding != "mono16":
            raise ValueError("DB3 color/depth image encodings do not match expected formats")
        if color_rgb.shape[:2] != (
            self.color_intrinsics.height,
            self.color_intrinsics.width,
        ):
            raise ValueError("DB3 color image size differs from its camera intrinsics")
        if depth_raw.shape != (
            self.depth_intrinsics.height,
            self.depth_intrinsics.width,
        ):
            raise ValueError("DB3 depth image size differs from its camera intrinsics")

        self.frame_count += 1
        return FramePacket(
            color_bgr=cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR),
            depth_raw=self._align_depth_to_color(depth_raw),
            timestamp_ns=int(round((color_timestamp + depth_timestamp) / 2.0)),
        )

    def close(self) -> None:
        self.connection.close()


@dataclass
class DepthEstimate:
    """一个目标区域的深度估计结果。"""

    # 最终选中的代表像素
    u: int
    v: int

    # 目标深度中位数
    depth_mm: float

    # 区域中的有效深度点数量
    valid_points: int

    # 主深度簇点数量
    cluster_points: int

    # region 中有效深度占比；优先基于未孔洞填充的原始深度计算
    region_points: int
    valid_ratio: float
    depth_mad_mm: float
    depth_std_mm: float
    raw_valid_points: int
    filled_points: int

    # 主深度簇反投影点，单位 mm
    points_camera_mm: np.ndarray = field(repr=False)


@dataclass
class Detection3D:
    """
    一个完整的三维检测结果。

    后续接 ROS 2 时，可以将它转换成自定义 msg 或 Pose/Point 消息。
    """

    class_id: int
    class_name: str
    confidence: float

    # x1, y1, x2, y2
    bbox: Tuple[int, int, int, int]

    # 用于反投影的像素坐标
    pixel_u: Optional[int]
    pixel_v: Optional[int]

    # 深度
    depth_mm: Optional[float]

    # 相机坐标系三维坐标
    x_mm: Optional[float]
    y_mm: Optional[float]
    z_mm: Optional[float]

    # seg_mask / center_roi
    depth_source: str

    # 深度统计信息
    valid_depth_points: int = 0
    cluster_depth_points: int = 0
    region_depth_points: int = 0
    depth_valid_ratio: float = 0.0
    depth_mad_mm: Optional[float] = None
    depth_std_mm: Optional[float] = None
    raw_valid_depth_points: int = 0
    filled_depth_points: int = 0

    # 帧级跟踪和抓取质量
    target_id: int = 0
    frame_id: int = 0
    timestamp_ns: int = 0
    position_std_mm: Optional[float] = None
    stable_frame_count: int = 0
    stable: bool = False
    # 仅用于展示和记录的时间戳卡尔曼平滑结果。抓取安全判断始终使用本帧实测 XYZ。
    filtered_camera_xyz_mm: Optional[Tuple[float, float, float]] = None
    filter_measurement_sigma_mm: Optional[float] = None
    # 仅供 GUI 使用：True 表示本帧由视觉跟踪补帧，XYZ 为最后一次检测到的值。
    display_predicted: bool = False

    # 目标几何和抓取候选
    object_width_mm: Optional[float] = None
    object_height_mm: Optional[float] = None
    object_depth_mm: Optional[float] = None
    required_jaw_opening_mm: Optional[float] = None
    grasp_type: Optional[str] = None
    grasp_score: float = 0.0
    graspable: bool = False
    status: str = "NOT_READY"
    reject_reason: str = "DEPTH_UNAVAILABLE"
    approach_path_clear: bool = False

    # 相机坐标下的抓取信息，始终可以用于调试和标定检查
    grasp_point_camera_xyz_mm: Optional[Tuple[float, float, float]] = None
    pregrasp_point_camera_xyz_mm: Optional[Tuple[float, float, float]] = None
    approach_vector_camera: Optional[Tuple[float, float, float]] = None
    orientation_quaternion_camera_xyzw: Optional[Tuple[float, float, float, float]] = None

    # 机器人底座坐标下的抓取信息；没有外参时保持 None，禁止误用相机 XYZ
    position_base_xyz_mm: Optional[Tuple[float, float, float]] = None
    grasp_point_base_xyz_mm: Optional[Tuple[float, float, float]] = None
    pregrasp_position_base_xyz_mm: Optional[Tuple[float, float, float]] = None
    approach_vector_base: Optional[Tuple[float, float, float]] = None
    orientation_quaternion_xyzw: Optional[Tuple[float, float, float, float]] = None
    coordinate_frame: str = "camera_color_optical_frame"
    within_robot_workspace: Optional[bool] = None

    # 仅用于当前进程内部规划，不写入 JSON
    point_cloud_camera_mm: Optional[np.ndarray] = field(default=None, repr=False)


# ==============================================================================
# 3. 运行环境 / GPU / GUI 调试
# ==============================================================================

def resolve_device(requested_device: str) -> str:
    """
    把用户指定的 device 解析成 Ultralytics 可使用的字符串。

    auto：
        CUDA 可用 -> "0"
        CUDA 不可用 -> "cpu"
    """

    requested = str(requested_device).strip().lower()

    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"

    return requested


def print_runtime_info(device: str) -> None:
    """打印 PyTorch / CUDA / GPU / DISPLAY 信息。"""

    print("=" * 80)
    print("[Runtime] Environment")
    print("=" * 80)

    print(f"[Runtime] Python executable : {sys.executable}")
    print(f"[Runtime] OpenCV            : {cv2.__version__}")
    print(f"[Runtime] PyTorch           : {torch.__version__}")
    print(f"[Runtime] Torch CUDA build  : {torch.version.cuda}")
    print(f"[Runtime] CUDA available    : {torch.cuda.is_available()}")
    print(f"[Runtime] DISPLAY           : {os.environ.get('DISPLAY', '<empty>')}")
    print(f"[Runtime] Selected device   : {device}")

    if torch.cuda.is_available():
        try:
            print(f"[Runtime] GPU 0             : {torch.cuda.get_device_name(0)}")
        except Exception as error:
            print(f"[Runtime] GPU name failed   : {error}")

    print()


def create_status_image(
    text_lines: List[str],
    width: int = WINDOW_INITIAL_WIDTH,
    height: int = WINDOW_INITIAL_HEIGHT,
) -> np.ndarray:
    """创建简单状态画面，在模型/相机初始化时也能看到 GUI。"""

    image = np.zeros((height, width, 3), dtype=np.uint8)

    title = "D435i + YOLO11 3D Detection"
    cv2.putText(
        image,
        title,
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        GREEN,
        2,
        cv2.LINE_AA,
    )

    y = 130
    for line in text_lines:
        cv2.putText(
            image,
            line,
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            WHITE,
            1,
            cv2.LINE_AA,
        )
        y += 38

    return image


def init_display(enabled: bool) -> bool:
    """
    尽早创建 OpenCV 窗口。

    这样可以明确区分：
        - GUI 问题
        - YOLO 第一次推理问题

    注意：DISPLAY 有值并不等于 X11/Qt 一定能正常显示。
    """

    if not enabled:
        print("[GUI] Display disabled by --no_display")
        return False

    print("[GUI] Creating OpenCV window...")

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            WINDOW_NAME,
            WINDOW_INITIAL_WIDTH,
            WINDOW_INITIAL_HEIGHT,
        )

        status = create_status_image(
            [
                "OpenCV window created.",
                "Loading YOLO model...",
            ]
        )

        cv2.imshow(WINDOW_NAME, status)
        cv2.waitKey(1)

        print("[GUI] OpenCV window created successfully.")
        return True

    except cv2.error as error:
        print("[GUI] OpenCV GUI initialization failed:")
        print(error)
        print("[GUI] Re-run with --no_display for headless mode.")
        return False


def show_status(enabled: bool, lines: List[str]) -> None:
    """更新初始化阶段的状态窗口。"""

    if not enabled:
        return

    image = create_status_image(lines)
    cv2.imshow(WINDOW_NAME, image)
    cv2.waitKey(1)


def yolo_warmup(
    model: YOLO,
    device: str,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    display_enabled: bool,
) -> None:
    """
    在打开相机主循环之前做一次 YOLO warm-up。

    作用：
    1. 第一次 CUDA context / cuDNN 初始化提前完成。
    2. 如果程序卡在第一次 YOLO 推理，会明确停在这里。
    3. 主循环第一次真实帧不会承担全部初始化开销。
    """

    print("=" * 80)
    print("[YOLO] Warm-up")
    print("=" * 80)
    print(f"[YOLO] Warm-up device : {device}")
    print(f"[YOLO] Warm-up imgsz  : {imgsz}")
    print("[YOLO] Warm-up starting...")

    show_status(
        display_enabled,
        [
            f"YOLO model loaded.",
            f"Device: {device}",
            "Running first inference warm-up...",
        ],
    )

    dummy_image = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    start_time = time.perf_counter()

    # Ultralytics 自己会进入 inference mode。
    # 这里使用 predict 的真实路径进行 warm-up，避免主循环第一次才初始化。
    _ = model.predict(
        source=dummy_image,
        imgsz=imgsz,
        rect=False,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        verbose=True,
        retina_masks=True,
    )

    # 对 CUDA 来说 synchronize 能保证计时包含实际 kernel 执行完成。
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    print(f"[YOLO] Warm-up finished: {elapsed_ms:.1f} ms")
    print()

    show_status(
        display_enabled,
        [
            "YOLO warm-up finished.",
            f"Warm-up: {elapsed_ms:.1f} ms",
            "Starting D435i camera...",
        ],
    )


# ==============================================================================
# 4. GUI 绘制辅助
# ==============================================================================

def draw_multiline_label(
    image: np.ndarray,
    lines: List[str],
    x: int,
    y: int,
    color: Tuple[int, int, int],
) -> None:
    """绘制多行文字，并为每行绘制黑色背景。"""

    if not lines:
        return

    image_h, image_w = image.shape[:2]
    line_height = 18

    x = max(2, min(x, image_w - 5))
    y = max(line_height, min(y, image_h - 5))

    for index, text in enumerate(lines):
        text_y = y + index * line_height

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            FONT_FACE,
            FONT_SCALE,
            FONT_THICKNESS,
        )

        if text_y + baseline >= image_h:
            break

        cv2.rectangle(
            image,
            (x, text_y - text_height - 3),
            (
                min(x + text_width + 4, image_w - 1),
                text_y + baseline + 2,
            ),
            BLACK,
            cv2.FILLED,
        )

        cv2.putText(
            image,
            text,
            (x + 2, text_y),
            FONT_FACE,
            FONT_SCALE,
            color,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )


def draw_prominent_xyz_label(
    image: np.ndarray,
    detection: Detection3D,
    x: int,
    y: int,
) -> None:
    """绘制大号 XYZ 标签，其中 Z 深度数字为红色，其余为绿色。"""

    image_h, image_w = image.shape[:2]
    font_scale = 0.85
    thickness = 2
    parts = format_detection_label_parts(detection)
    text_sizes = [
        cv2.getTextSize(text, FONT_FACE, font_scale, thickness)
        for text, _ in parts
    ]
    text_width = sum(size[0][0] for size in text_sizes)
    text_height = max(size[0][1] for size in text_sizes)
    baseline = max(size[1] for size in text_sizes)
    x = max(2, min(x, max(2, image_w - text_width - 8)))
    y = max(text_height + 6, min(y, image_h - baseline - 4))
    cv2.rectangle(
        image,
        (x - 3, y - text_height - 5),
        (min(image_w - 1, x + text_width + 5), y + baseline + 4),
        BLACK,
        cv2.FILLED,
    )
    text_x = x
    for (text, color), ((part_width, _), _) in zip(parts, text_sizes):
        cv2.putText(
            image,
            text,
            (text_x, y),
            FONT_FACE,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        text_x += part_width


def draw_segmentation_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.30,
) -> None:
    """在原图上半透明绘制实例分割 Mask。"""

    if mask is None:
        return

    mask_bool = mask.astype(bool)

    if not np.any(mask_bool):
        return

    color_array = np.array(color, dtype=np.float32)
    original_pixels = image[mask_bool].astype(np.float32)

    blended_pixels = (
        original_pixels * (1.0 - alpha)
        + color_array * alpha
    )

    image[mask_bool] = np.clip(
        blended_pixels,
        0,
        255,
    ).astype(np.uint8)


# ==============================================================================
# 5. Detect 模型：中心 ROI
# ==============================================================================

def create_center_roi_mask(
    image_shape: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
    ratio: float = CENTER_ROI_RATIO,
) -> np.ndarray:
    """
    在 Bounding Box 中创建中心 ROI Mask。

    ratio=0.5：
        只使用 bbox 中间 50% x 50% 区域测深。
    """

    image_h, image_w = image_shape
    x1, y1, x2, y2 = bbox

    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    roi_width = box_width * ratio
    roi_height = box_height * ratio

    roi_x1 = int(round(center_x - roi_width / 2.0))
    roi_y1 = int(round(center_y - roi_height / 2.0))
    roi_x2 = int(round(center_x + roi_width / 2.0))
    roi_y2 = int(round(center_y + roi_height / 2.0))

    roi_x1 = max(0, min(roi_x1, image_w - 1))
    roi_y1 = max(0, min(roi_y1, image_h - 1))
    roi_x2 = max(roi_x1 + 1, min(roi_x2, image_w))
    roi_y2 = max(roi_y1 + 1, min(roi_y2, image_h))

    mask = np.zeros((image_h, image_w), dtype=bool)
    mask[roi_y1:roi_y2, roi_x1:roi_x2] = True

    return mask


# ==============================================================================
# 6. Seg 模型：Mask 处理
# ==============================================================================

def prepare_segmentation_mask(
    raw_mask: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    将 Ultralytics Mask 转成与 RGB/Depth 相同大小的 bool mask。

    retina_masks=True 时通常已经是原图尺寸；
    这里仍保留 resize 作为鲁棒性保护。

    另外会轻微腐蚀边缘，减少背景深度混入。
    """

    image_h, image_w = image_shape

    if raw_mask.shape != (image_h, image_w):
        raw_mask = cv2.resize(
            raw_mask.astype(np.float32),
            (image_w, image_h),
            interpolation=cv2.INTER_NEAREST,
        )

    mask = (raw_mask > 0.5).astype(np.uint8)

    if SEG_MASK_ERODE_KERNEL >= 3:
        kernel_size = int(SEG_MASK_ERODE_KERNEL)

        # 强制使用奇数 kernel
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )

        eroded = cv2.erode(mask, kernel, iterations=1)

        # 如果目标本身很小，腐蚀可能把 mask 全部吃掉。
        # 这时回退到原始 mask。
        if np.count_nonzero(eroded) >= MIN_VALID_DEPTH_POINTS:
            mask = eroded

    return mask.astype(bool)


# ==============================================================================
# 7. 深度主峰聚类
# ==============================================================================

def estimate_depth_from_region(
    depth_map: np.ndarray,
    region_mask: np.ndarray,
    intrinsics: Optional[CameraIntrinsics] = None,
    raw_depth_map: Optional[np.ndarray] = None,
) -> Optional[DepthEstimate]:
    """
    从某个目标区域中估计目标深度。

    流程：
        region mask
            -> 有效深度点
            -> 深度直方图
            -> 最大主峰
            -> 主峰附近 +/- window
            -> 深度中位数
            -> 选取一个真实有效的代表像素 (u, v)

    相比“整个 bbox 直接 median”，这种方法能明显减少背景干扰。
    """

    region_mask = region_mask.astype(bool)
    region_points = int(np.count_nonzero(region_mask))
    if region_points == 0:
        return None

    valid_mask = (
        region_mask
        & (depth_map > MIN_DEPTH_MM)
        & (depth_map < MAX_DEPTH_MM)
    )

    ys, xs = np.nonzero(valid_mask)

    if xs.size < MIN_VALID_DEPTH_POINTS:
        return None

    depths = depth_map[ys, xs].astype(np.float32)

    if raw_depth_map is not None and raw_depth_map.shape == depth_map.shape:
        raw_valid_mask = (
            region_mask
            & (raw_depth_map > MIN_DEPTH_MM)
            & (raw_depth_map < MAX_DEPTH_MM)
        )
        raw_valid_points = int(np.count_nonzero(raw_valid_mask))
    else:
        raw_valid_points = int(depths.size)

    # --------------------------------------------------------------------------
    # 深度直方图
    # --------------------------------------------------------------------------

    bin_ids = np.floor(
        depths / DEPTH_HIST_BIN_MM
    ).astype(np.int32)

    unique_bins, counts = np.unique(
        bin_ids,
        return_counts=True,
    )

    if unique_bins.size == 0:
        return None

    peak_index = int(np.argmax(counts))
    peak_bin = int(unique_bins[peak_index])

    peak_depth_center = (
        peak_bin + 0.5
    ) * DEPTH_HIST_BIN_MM

    cluster_mask = (
        np.abs(depths - peak_depth_center)
        <= DEPTH_CLUSTER_WINDOW_MM
    )

    cluster_count = int(np.count_nonzero(cluster_mask))

    if cluster_count < MIN_VALID_DEPTH_POINTS:
        # 主峰点太少时退化为全部有效区域。
        cluster_depths = depths
        cluster_xs = xs
        cluster_ys = ys
    else:
        cluster_depths = depths[cluster_mask]
        cluster_xs = xs[cluster_mask]
        cluster_ys = ys[cluster_mask]

    if cluster_depths.size == 0:
        return None

    depth_mm = float(np.median(cluster_depths))
    depth_mad_mm = float(
        np.median(np.abs(cluster_depths - depth_mm))
    )
    depth_std_mm = float(np.std(cluster_depths))

    if raw_depth_map is not None and raw_depth_map.shape == depth_map.shape:
        cluster_raw_valid = (
            raw_depth_map[cluster_ys, cluster_xs] > MIN_DEPTH_MM
        ) & (raw_depth_map[cluster_ys, cluster_xs] < MAX_DEPTH_MM)
        cluster_raw_valid_points = int(np.count_nonzero(cluster_raw_valid))
    else:
        cluster_raw_valid_points = int(cluster_depths.size)

    if intrinsics is None:
        points_camera_mm = np.empty((0, 3), dtype=np.float32)
    else:
        z_values = cluster_depths.astype(np.float32)
        points_camera_mm = np.column_stack(
            (
                (cluster_xs.astype(np.float32) - intrinsics.cx)
                * z_values
                / intrinsics.fx,
                (cluster_ys.astype(np.float32) - intrinsics.cy)
                * z_values
                / intrinsics.fy,
                z_values,
            )
        ).astype(np.float32)

    # --------------------------------------------------------------------------
    # 代表像素：
    #
    # 先计算簇内像素坐标的中位数，再在真实簇像素中寻找离该位置最近的点。
    # 这样最终的 (u, v) 一定真的是目标深度簇中的有效像素，
    # 而不是一个可能落在空洞/背景上的“虚构中心点”。
    # --------------------------------------------------------------------------

    median_u = float(np.median(cluster_xs))
    median_v = float(np.median(cluster_ys))

    distance_sq = (
        (cluster_xs.astype(np.float32) - median_u) ** 2
        + (cluster_ys.astype(np.float32) - median_v) ** 2
    )

    representative_index = int(np.argmin(distance_sq))

    u = int(cluster_xs[representative_index])
    v = int(cluster_ys[representative_index])

    return DepthEstimate(
        u=u,
        v=v,
        depth_mm=depth_mm,
        valid_points=int(depths.size),
        cluster_points=int(cluster_depths.size),
        region_points=region_points,
        valid_ratio=float(raw_valid_points / max(region_points, 1)),
        depth_mad_mm=depth_mad_mm,
        depth_std_mm=depth_std_mm,
        raw_valid_points=raw_valid_points,
        filled_points=max(0, int(cluster_depths.size) - cluster_raw_valid_points),
        points_camera_mm=points_camera_mm,
    )


# ==============================================================================
# 8. RGB 相机内参
# ==============================================================================

def get_realsense_color_intrinsics(color_frame) -> CameraIntrinsics:
    """读取对齐后彩色帧的 RealSense 内参，单位和坐标系与本程序一致。"""

    profile = color_frame.get_profile().as_video_stream_profile()
    intrinsics = profile.get_intrinsics()

    return CameraIntrinsics(
        fx=float(intrinsics.fx),
        fy=float(intrinsics.fy),
        cx=float(intrinsics.ppx),
        cy=float(intrinsics.ppy),
        width=int(intrinsics.width),
        height=int(intrinsics.height),
    )


# ==============================================================================
# 9. 像素 + 深度 -> 相机三维坐标
# ==============================================================================

def deproject_pixel_to_3d(
    u: float,
    v: float,
    depth_mm: float,
    intrinsics: CameraIntrinsics,
) -> Tuple[float, float, float]:
    """
    标准针孔模型反投影。

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth

    返回单位：mm
    """

    z = float(depth_mm)

    x = (
        (float(u) - intrinsics.cx)
        * z
        / intrinsics.fx
    )

    y = (
        (float(v) - intrinsics.cy)
        * z
        / intrinsics.fy
    )

    return x, y, z


# ==============================================================================
# 10. 质量评估、抓取规划和坐标变换
# ==============================================================================

def clamp01(value: float) -> float:
    """将数值限制到 [0, 1]。"""

    return float(max(0.0, min(1.0, value)))


def normalize_vector(
    vector: np.ndarray,
    fallback: Optional[np.ndarray] = None,
) -> np.ndarray:
    """返回单位向量；输入退化时使用 fallback。"""

    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm > 1e-9:
        return value / norm

    if fallback is None:
        raise ValueError("Cannot normalize a zero vector")

    fallback_value = np.asarray(fallback, dtype=np.float64)
    fallback_norm = float(np.linalg.norm(fallback_value))
    if fallback_norm <= 1e-9:
        raise ValueError("Fallback vector is also zero")
    return fallback_value / fallback_norm


def compute_environment_quality(
    image_bgr: np.ndarray,
    depth_map: np.ndarray,
    config: QualityConfig,
) -> EnvironmentQuality:
    """根据当前 RGB-D 帧计算可解释的环境质量指标。"""

    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr

    gray_float = gray.astype(np.float32)
    brightness_mean = float(np.mean(gray_float))
    dark_pixel_ratio = float(
        np.mean(gray_float < config.dark_pixel_threshold)
    )
    overexposed_ratio = float(
        np.mean(gray_float > config.overexposed_pixel_threshold)
    )
    contrast_std = float(np.std(gray_float))

    valid_depth = (
        (depth_map > MIN_DEPTH_MM)
        & (depth_map < MAX_DEPTH_MM)
    )
    depth_valid_ratio = float(np.mean(valid_depth))

    flags: List[str] = []
    if (
        brightness_mean < config.low_light_brightness
        or dark_pixel_ratio > config.dark_pixel_ratio
    ):
        flags.append("LOW_LIGHT")

    if overexposed_ratio > config.overexposed_ratio:
        flags.append("STRONG_LIGHT")

    if contrast_std < config.low_contrast_std:
        flags.append("LOW_CONTRAST")

    if depth_valid_ratio < config.min_global_depth_valid_ratio:
        flags.append("DEPTH_UNRELIABLE")

    # 仅作为“疑似”标记：没有雾度传感器时不能把图像启发式当作雾天真值。
    if (
        "LOW_CONTRAST" in flags
        and depth_valid_ratio < config.fog_suspected_depth_ratio
    ):
        flags.append("FOG_OR_SCATTERING_SUSPECTED")

    unique_flags = tuple(dict.fromkeys(flags))
    status = "NORMAL" if not unique_flags else "|".join(unique_flags)
    grasp_safe = not bool(set(unique_flags) & HARD_ENVIRONMENT_FLAGS)

    return EnvironmentQuality(
        brightness_mean=brightness_mean,
        dark_pixel_ratio=dark_pixel_ratio,
        overexposed_ratio=overexposed_ratio,
        contrast_std=contrast_std,
        depth_valid_ratio=depth_valid_ratio,
        flags=unique_flags,
        status=status,
        grasp_safe=grasp_safe,
    )


def enhance_rgb_for_environment(
    image_bgr: np.ndarray,
    quality: EnvironmentQuality,
    enabled: bool = True,
) -> np.ndarray:
    """仅在弱光/低对比度时增强 YOLO 输入，不处理强光过曝帧。"""

    if not enabled:
        return image_bgr

    enhanced = image_bgr.copy()

    if "LOW_LIGHT" in quality.flags:
        # 用 LUT 做温和 Gamma，避免把噪声无限放大；随后只对 L 通道做 CLAHE。
        gamma = 1.35 if quality.brightness_mean < 25.0 else 1.18
        inverse_gamma = 1.0 / gamma
        lut = np.array(
            [
                ((index / 255.0) ** inverse_gamma) * 255.0
                for index in range(256)
            ],
            dtype=np.uint8,
        )
        enhanced = cv2.LUT(enhanced, lut)

    if "LOW_CONTRAST" in quality.flags and "STRONG_LIGHT" not in quality.flags:
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        lab = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced


def rotation_matrix_to_quaternion_xyzw(
    rotation: np.ndarray,
) -> Tuple[float, float, float, float]:
    """将 3x3 旋转矩阵转换为 ROS 常用的 xyzw 四元数。"""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Rotation must have shape (3, 3)")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(
            max(1e-12, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        ) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(
            max(1e-12, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        ) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(
            max(1e-12, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        ) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale

    quaternion = np.asarray((x, y, z, w), dtype=np.float64)
    quaternion = normalize_vector(quaternion)
    return tuple(float(value) for value in quaternion)


# ==============================================================================
# 11. 点云几何和抓取候选
# ==============================================================================

def estimate_object_geometry(
    points_camera_mm: np.ndarray,
) -> Optional[ObjectGeometry]:
    """对目标点云做稳健 PCA，估计尺寸、竖直轴、夹爪轴和接近轴。"""

    points = np.asarray(points_camera_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        return None

    points = points[np.all(np.isfinite(points), axis=1)]
    points = points[
        (points[:, 2] > MIN_DEPTH_MM)
        & (points[:, 2] < MAX_DEPTH_MM)
    ]
    if points.shape[0] < MIN_POINT_CLOUD_POINTS:
        return None

    # 先去掉距离几何中位数最远的少量离群点，避免背景或错误深度主导 PCA。
    median_point = np.median(points, axis=0)
    distances = np.linalg.norm(points - median_point, axis=1)
    keep_limit = float(np.percentile(distances, 95.0))
    inliers = points[distances <= max(keep_limit, 1e-6)]
    if inliers.shape[0] >= MIN_POINT_CLOUD_POINTS:
        points = inliers

    center = np.median(points, axis=0)
    centered = points - center
    if points.shape[0] < 3:
        return None

    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]

    camera_up = np.asarray((0.0, -1.0, 0.0), dtype=np.float64)
    vertical_index = int(np.argmax(np.abs(axes.T @ camera_up)))
    vertical_axis = normalize_vector(axes[:, vertical_index], camera_up)
    if float(np.dot(vertical_axis, camera_up)) < 0.0:
        vertical_axis = -vertical_axis

    remaining_indices = [index for index in range(3) if index != vertical_index]
    jaw_index = max(
        remaining_indices,
        key=lambda index: float(np.ptp(centered @ axes[:, index])),
    )
    jaw_axis = normalize_vector(axes[:, jaw_index])
    jaw_axis = normalize_vector(
        jaw_axis - vertical_axis * float(np.dot(jaw_axis, vertical_axis)),
        np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
    )
    if float(np.dot(jaw_axis, np.asarray((1.0, 0.0, 0.0)))) < 0.0:
        jaw_axis = -jaw_axis

    # camera_up x jaw_axis 指向目标前方时，接近方向与夹爪闭合轴正交。
    approach_axis = normalize_vector(
        np.cross(vertical_axis, jaw_axis),
        np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
    )
    if float(np.dot(approach_axis, center)) < 0.0:
        approach_axis = -approach_axis

    vertical_projection = centered @ vertical_axis
    top_threshold = float(np.percentile(vertical_projection, 75.0))
    top_points = points[vertical_projection >= top_threshold]

    width_mm = float(np.ptp(centered @ jaw_axis))
    height_mm = float(np.ptp(vertical_projection))
    depth_mm = float(np.ptp(centered @ approach_axis))

    return ObjectGeometry(
        center_camera_xyz_mm=tuple(float(value) for value in center),
        vertical_axis_camera=tuple(float(value) for value in vertical_axis),
        jaw_axis_camera=tuple(float(value) for value in jaw_axis),
        approach_axis_camera=tuple(float(value) for value in approach_axis),
        width_mm=max(0.0, width_mm),
        height_mm=max(0.0, height_mm),
        depth_mm=max(0.0, depth_mm),
        top_points_camera_mm=top_points.astype(np.float32),
    )


def make_gripper_rotation(
    jaw_axis_camera: np.ndarray,
    approach_vector_camera: np.ndarray,
) -> np.ndarray:
    """构造夹爪姿态矩阵，列向量依次为闭合轴、侧向轴、接近轴。"""

    jaw_axis = normalize_vector(jaw_axis_camera)
    approach = normalize_vector(approach_vector_camera)
    approach = normalize_vector(
        approach - jaw_axis * float(np.dot(approach, jaw_axis)),
        np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
    )
    side_axis = normalize_vector(np.cross(approach, jaw_axis))
    return np.column_stack((jaw_axis, side_axis, approach))


def check_approach_path_clear(
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    grasp_point_camera_xyz_mm: Sequence[float],
    pregrasp_point_camera_xyz_mm: Sequence[float],
    obstacle_margin_mm: float = APPROACH_OBSTACLE_MARGIN_MM,
) -> bool:
    """在预抓取点到抓取点的投影路径上检查是否存在明显更近的深度。"""

    grasp_point = np.asarray(grasp_point_camera_xyz_mm, dtype=np.float64)
    pregrasp_point = np.asarray(pregrasp_point_camera_xyz_mm, dtype=np.float64)
    image_h, image_w = depth_map.shape[:2]

    # 不检查最后 10%，避免把目标本身的表面误报成障碍物。
    for ratio in np.linspace(0.0, 0.9, 10):
        sample = pregrasp_point * (1.0 - ratio) + grasp_point * ratio
        if sample[2] <= MIN_DEPTH_MM:
            continue

        u = int(round(intrinsics.fx * sample[0] / sample[2] + intrinsics.cx))
        v = int(round(intrinsics.fy * sample[1] / sample[2] + intrinsics.cy))
        if not (0 <= u < image_w and 0 <= v < image_h):
            continue

        radius = 2
        x1 = max(0, u - radius)
        x2 = min(image_w, u + radius + 1)
        y1 = max(0, v - radius)
        y2 = min(image_h, v + radius + 1)
        observed = depth_map[y1:y2, x1:x2]
        observed = observed[
            (observed > MIN_DEPTH_MM)
            & (observed < MAX_DEPTH_MM)
        ]
        if observed.size == 0:
            continue

        nearest_observed = float(np.percentile(observed, 20.0))
        if nearest_observed < float(sample[2]) - obstacle_margin_mm:
            return False

    return True


def create_grasp_candidates(
    points_camera_mm: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: CameraIntrinsics,
    pregrasp_distance_mm: float,
    grasp_margin_mm: float,
) -> Tuple[Optional[ObjectGeometry], List[GraspCandidate]]:
    """生成 SIDE_GRASP 和可用时的 RIM_GRASP 候选。"""

    geometry = estimate_object_geometry(points_camera_mm)
    if geometry is None:
        return None, []

    approach = np.asarray(geometry.approach_axis_camera, dtype=np.float64)
    jaw_axis = np.asarray(geometry.jaw_axis_camera, dtype=np.float64)
    candidates: List[GraspCandidate] = []

    def build_candidate(
        grasp_type: str,
        grasp_point: np.ndarray,
        width_mm: float,
        base_score: float,
    ) -> GraspCandidate:
        pregrasp_point = grasp_point - pregrasp_distance_mm * approach
        orientation = make_gripper_rotation(jaw_axis, approach)
        path_clear = check_approach_path_clear(
            depth_map,
            intrinsics,
            grasp_point,
            pregrasp_point,
        )
        score = base_score * (1.0 if path_clear else 0.20)
        return GraspCandidate(
            grasp_type=grasp_type,
            grasp_point_camera_xyz_mm=tuple(float(value) for value in grasp_point),
            pregrasp_point_camera_xyz_mm=tuple(float(value) for value in pregrasp_point),
            approach_vector_camera=tuple(float(value) for value in approach),
            jaw_axis_camera=tuple(float(value) for value in jaw_axis),
            orientation_camera_quaternion_xyzw=rotation_matrix_to_quaternion_xyzw(orientation),
            required_jaw_opening_mm=max(0.0, float(width_mm) + grasp_margin_mm),
            score=float(score),
            path_clear=path_clear,
        )

    center = np.asarray(geometry.center_camera_xyz_mm, dtype=np.float64)
    candidates.append(
        build_candidate(
            "SIDE_GRASP",
            center,
            geometry.width_mm,
            0.85,
        )
    )

    top_points = geometry.top_points_camera_mm
    if top_points.shape[0] >= MIN_RIM_POINT_CLOUD_POINTS:
        top_points64 = top_points.astype(np.float64)
        top_center = np.median(top_points64, axis=0)
        top_width = float(
            np.ptp((top_points64 - top_center) @ jaw_axis)
        )
        candidates.append(
            build_candidate(
                "RIM_GRASP",
                top_center,
                top_width,
                0.70,
            )
        )

    return geometry, candidates


def select_grasp_candidate(
    candidates: Sequence[GraspCandidate],
    max_jaw_opening_mm: float,
) -> Optional[GraspCandidate]:
    """优先选能闭合的候选；若都超宽，保留最佳候选并由上层给出拒绝原因。"""

    if not candidates:
        return None

    feasible = [
        candidate
        for candidate in candidates
        if candidate.required_jaw_opening_mm <= max_jaw_opening_mm
    ]
    clear_feasible = [candidate for candidate in feasible if candidate.path_clear]
    pool = clear_feasible or feasible or list(candidates)
    return max(pool, key=lambda candidate: candidate.score)


def is_point_in_workspace(
    point_xyz_mm: Optional[Sequence[float]],
    minimum_xyz_mm: Optional[Sequence[float]],
    maximum_xyz_mm: Optional[Sequence[float]],
) -> Optional[bool]:
    """检查底座坐标点是否落在配置的轴对齐工作空间盒内。"""

    if (
        point_xyz_mm is None
        or minimum_xyz_mm is None
        or maximum_xyz_mm is None
    ):
        return None
    point = np.asarray(point_xyz_mm, dtype=np.float64)
    minimum = np.asarray(minimum_xyz_mm, dtype=np.float64)
    maximum = np.asarray(maximum_xyz_mm, dtype=np.float64)
    return bool(np.all(point >= minimum) and np.all(point <= maximum))


def parse_xyz_triplet(value: Optional[str], option_name: str) -> Optional[Tuple[float, float, float]]:
    """解析命令行中的 x,y,z 毫米三元组。"""

    if value is None:
        return None
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError(f"{option_name} must be formatted as x,y,z") from error
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{option_name} must contain three finite values")
    return values


# ==============================================================================
# 12. 坐标变换和多帧稳定
# ==============================================================================

def load_extrinsics(path: Optional[str]) -> Optional[Extrinsics]:
    """读取 camera_color_optical_frame -> robot_base 的 JSON 外参。"""

    if not path:
        return None

    resolved_path = os.path.abspath(path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Extrinsics file does not exist: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if bool(data.get("calibration_required", False)):
        raise ValueError(
            "This extrinsics file is an example; replace it with calibrated values "
            "and set calibration_required to false"
        )

    matrix_value = data.get("matrix_base_from_camera")
    if matrix_value is not None:
        matrix = np.asarray(matrix_value, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("matrix_base_from_camera must have shape 4x4")
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
    else:
        rotation_value = data.get(
            "rotation_base_from_camera",
            data.get("rotation"),
        )
        translation_value = data.get(
            "translation_base_mm",
            data.get("translation_mm"),
        )
        if rotation_value is None or translation_value is None:
            raise ValueError(
                "Extrinsics JSON needs matrix_base_from_camera or "
                "rotation_base_from_camera + translation_base_mm"
            )
        rotation = np.asarray(rotation_value, dtype=np.float64)
        translation = np.asarray(translation_value, dtype=np.float64)
        if translation.shape == (3, 1):
            translation = translation[:, 0]
        if translation.shape != (3,):
            raise ValueError("translation_base_mm must contain three values")

    if rotation.shape != (3, 3):
        raise ValueError("rotation_base_from_camera must have shape 3x3")
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        raise ValueError("Extrinsics contain non-finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError("rotation_base_from_camera is not orthonormal")
    if np.linalg.det(rotation) <= 0.0:
        raise ValueError("rotation_base_from_camera must have positive determinant")

    return Extrinsics(
        rotation_base_from_camera=rotation,
        translation_base_mm=translation,
        source_frame=str(data.get("source_frame", "camera_color_optical_frame")),
        target_frame=str(data.get("target_frame", "robot_base")),
    )


def transform_point_to_base(
    extrinsics: Extrinsics,
    point_camera_xyz_mm: Sequence[float],
) -> Tuple[float, float, float]:
    point = (
        extrinsics.rotation_base_from_camera
        @ np.asarray(point_camera_xyz_mm, dtype=np.float64)
        + extrinsics.translation_base_mm
    )
    return tuple(float(value) for value in point)


def transform_vector_to_base(
    extrinsics: Extrinsics,
    vector_camera: Sequence[float],
) -> Tuple[float, float, float]:
    vector = extrinsics.rotation_base_from_camera @ np.asarray(
        vector_camera,
        dtype=np.float64,
    )
    vector = normalize_vector(vector)
    return tuple(float(value) for value in vector)


@dataclass
class _TrackedTarget:
    target_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    missed: int = 0
    samples: Deque[np.ndarray] = field(default_factory=deque)
    position_filter: Optional["TimestampPositionKalman"] = None


class TimestampPositionKalman:
    """以真实时间戳推进的常速度 XYZ 卡尔曼滤波器，坐标单位为 mm。

    该滤波器只产生显示/日志用的平滑坐标；抓取质量和 READY 判定仍由原始
    RGB-D 实测值完成，避免滤波预测掩盖深度异常。
    """

    def __init__(
        self,
        fallback_dt_seconds: float,
        acceleration_sigma_mm_s2: float = 800.0,
    ) -> None:
        self.fallback_dt_seconds = max(1e-3, float(fallback_dt_seconds))
        self.acceleration_sigma_mm_s2 = max(1.0, float(acceleration_sigma_mm_s2))
        self.state: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.timestamp_ns: Optional[int] = None

    def update(
        self,
        measurement_mm: np.ndarray,
        timestamp_ns: int,
        measurement_sigma_mm: float,
    ) -> np.ndarray:
        measurement = np.asarray(measurement_mm, dtype=np.float64).reshape(3)
        if self.state is None or self.covariance is None:
            self.state = np.zeros(6, dtype=np.float64)
            self.state[:3] = measurement
            self.covariance = np.diag([2500.0, 2500.0, 2500.0, 1e6, 1e6, 1e6])
            self.timestamp_ns = int(timestamp_ns)
            return measurement.copy()

        dt = self.fallback_dt_seconds
        if self.timestamp_ns is not None and timestamp_ns > self.timestamp_ns:
            dt = max(1e-3, min((timestamp_ns - self.timestamp_ns) / 1_000_000_000.0, 1.0))
        self.timestamp_ns = int(timestamp_ns)

        transition = np.eye(6, dtype=np.float64)
        transition[:3, 3:] = np.eye(3, dtype=np.float64) * dt
        q = self.acceleration_sigma_mm_s2 ** 2
        process = np.zeros((6, 6), dtype=np.float64)
        process[:3, :3] = np.eye(3) * (q * dt ** 4 / 4.0)
        process[:3, 3:] = np.eye(3) * (q * dt ** 3 / 2.0)
        process[3:, :3] = np.eye(3) * (q * dt ** 3 / 2.0)
        process[3:, 3:] = np.eye(3) * (q * dt ** 2)

        predicted_state = transition @ self.state
        predicted_covariance = transition @ self.covariance @ transition.T + process
        observation = np.zeros((3, 6), dtype=np.float64)
        observation[:, :3] = np.eye(3, dtype=np.float64)
        sigma = max(1.0, float(measurement_sigma_mm))
        measurement_covariance = np.eye(3, dtype=np.float64) * sigma ** 2
        innovation = measurement - observation @ predicted_state
        innovation_covariance = (
            observation @ predicted_covariance @ observation.T + measurement_covariance
        )
        gain = predicted_covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        identity = np.eye(6, dtype=np.float64)
        self.state = predicted_state + gain @ innovation
        # Joseph form prevents accumulated numerical asymmetry/negative variance.
        residual = identity - gain @ observation
        self.covariance = (
            residual @ predicted_covariance @ residual.T
            + gain @ measurement_covariance @ gain.T
        )
        return self.state[:3].copy()


class TemporalStabilityTracker:
    """按 bbox 和类别关联目标，并计算最近 N 帧 XYZ 的离散程度。"""

    def __init__(
        self,
        window_frames: int,
        min_stable_frames: int,
        max_position_std_mm: float,
        fallback_frame_interval_seconds: float = 1.0 / 30.0,
    ) -> None:
        self.window_frames = max(1, int(window_frames))
        self.min_stable_frames = max(1, int(min_stable_frames))
        self.max_position_std_mm = float(max_position_std_mm)
        self.fallback_frame_interval_seconds = max(
            1e-3, float(fallback_frame_interval_seconds)
        )
        self._tracks: Dict[int, _TrackedTarget] = {}
        self._next_target_id = 1

    @staticmethod
    def _bbox_iou(
        first: Tuple[int, int, int, int],
        second: Tuple[int, int, int, int],
    ) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_first = max(1, first[2] - first[0]) * max(1, first[3] - first[1])
        area_second = max(1, second[2] - second[0]) * max(1, second[3] - second[1])
        union = area_first + area_second - intersection
        return float(intersection / max(union, 1))

    @staticmethod
    def _center_distance_ratio(
        first: Tuple[int, int, int, int],
        second: Tuple[int, int, int, int],
    ) -> float:
        first_center = np.asarray(
            ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
        )
        second_center = np.asarray(
            ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
        )
        diagonal = max(
            1.0,
            math.hypot(first[2] - first[0], first[3] - first[1]),
            math.hypot(second[2] - second[0], second[3] - second[1]),
        )
        return float(np.linalg.norm(first_center - second_center) / diagonal)

    def update(self, detections: List[Detection3D]) -> None:
        unmatched_tracks = set(self._tracks.keys())
        unmatched_detections = set(range(len(detections)))
        pairings: List[Tuple[int, int]] = []

        candidates: List[Tuple[float, int, int]] = []
        for target_id, track in self._tracks.items():
            for index, detection in enumerate(detections):
                if track.class_name != detection.class_name:
                    continue
                iou = self._bbox_iou(track.bbox, detection.bbox)
                distance = self._center_distance_ratio(track.bbox, detection.bbox)
                if iou >= 0.10 or distance <= 0.60:
                    candidates.append((max(iou, 1.0 - distance), target_id, index))

        for _, target_id, index in sorted(candidates, reverse=True):
            if target_id not in unmatched_tracks or index not in unmatched_detections:
                continue
            pairings.append((target_id, index))
            unmatched_tracks.remove(target_id)
            unmatched_detections.remove(index)

        for target_id, index in pairings:
            track = self._tracks[target_id]
            detection = detections[index]
            if track.missed > 0:
                track.samples.clear()
            track.missed = 0
            track.bbox = detection.bbox
            detection.target_id = target_id
            if detection.x_mm is not None and detection.y_mm is not None and detection.z_mm is not None:
                track.samples.append(
                    np.asarray(
                        (detection.x_mm, detection.y_mm, detection.z_mm),
                        dtype=np.float64,
                    )
                )
            else:
                # 深度缺失也会打断“连续多帧确认”，不能沿用旧的稳定窗口。
                track.samples.clear()
            self._apply_position_filter(detection, track)
            while len(track.samples) > self.window_frames:
                track.samples.popleft()
            self._apply_track_quality(detection, track)

        for index in unmatched_detections:
            detection = detections[index]
            track = _TrackedTarget(
                target_id=self._next_target_id,
                class_name=detection.class_name,
                bbox=detection.bbox,
            )
            self._next_target_id += 1
            detection.target_id = track.target_id
            track.position_filter = TimestampPositionKalman(
                self.fallback_frame_interval_seconds
            )
            if detection.x_mm is not None and detection.y_mm is not None and detection.z_mm is not None:
                track.samples.append(
                    np.asarray(
                        (detection.x_mm, detection.y_mm, detection.z_mm),
                        dtype=np.float64,
                    )
                )
            self._tracks[track.target_id] = track
            self._apply_position_filter(detection, track)
            self._apply_track_quality(detection, track)

        for target_id in list(unmatched_tracks):
            track = self._tracks[target_id]
            track.missed += 1
            if track.missed > 2:
                del self._tracks[target_id]

    @staticmethod
    def _measurement_sigma_mm(detection: Detection3D) -> float:
        """将当前 RGB-D 质量转成滤波测量噪声，不参与 READY 判定。"""

        dispersion = detection.depth_mad_mm
        if dispersion is None:
            dispersion = detection.depth_std_mm
        if dispersion is None:
            dispersion = 30.0
        valid_ratio = max(0.10, float(detection.depth_valid_ratio))
        confidence_penalty = 1.0 + max(0.0, 0.60 - detection.confidence)
        return float(np.clip((max(15.0, dispersion) / valid_ratio) * confidence_penalty, 15.0, 500.0))

    def _apply_position_filter(
        self,
        detection: Detection3D,
        track: _TrackedTarget,
    ) -> None:
        if (
            detection.x_mm is None
            or detection.y_mm is None
            or detection.z_mm is None
            or track.position_filter is None
        ):
            detection.filtered_camera_xyz_mm = None
            detection.filter_measurement_sigma_mm = None
            return
        sigma = self._measurement_sigma_mm(detection)
        filtered = track.position_filter.update(
            np.asarray((detection.x_mm, detection.y_mm, detection.z_mm)),
            detection.timestamp_ns,
            sigma,
        )
        detection.filtered_camera_xyz_mm = tuple(float(value) for value in filtered)
        detection.filter_measurement_sigma_mm = sigma

    def _apply_track_quality(
        self,
        detection: Detection3D,
        track: _TrackedTarget,
    ) -> None:
        detection.stable_frame_count = len(track.samples)
        if len(track.samples) == 0:
            detection.position_std_mm = None
            detection.stable = False
            return
        positions = np.stack(tuple(track.samples), axis=0)
        axis_std = np.std(positions, axis=0)
        detection.position_std_mm = float(np.linalg.norm(axis_std))
        detection.stable = bool(
            len(track.samples) >= self.min_stable_frames
            and detection.position_std_mm <= self.max_position_std_mm
        )


@dataclass
class _VisualTrackedTarget:
    """仅用于 GUI 的图像特征跟踪状态。"""

    target_id: int
    class_name: str
    detection: Detection3D
    bbox: Tuple[int, int, int, int]
    points: Optional[np.ndarray] = field(default=None, repr=False)
    missed: int = 0
    flow_updated: bool = False


class VisualObjectTracker:
    """用目标框内 LK 光流特征在 YOLO 漏检帧继续跟踪垃圾桶。

    YOLO 检测负责首次建轨、重捕获和消除长期漂移；光流在相邻帧间根据桶身
    的真实图像特征移动框。因此它不会只把旧框固定在原地。
    """

    def __init__(
        self,
        max_missed_frames: int,
        min_confidence: float = VISUAL_TRACK_MIN_CONFIDENCE,
        min_confirmations: int = VISUAL_TRACK_MIN_CONFIRMATIONS,
        confirmation_seconds: float = VISUAL_TRACK_CONFIRMATION_SECONDS,
        smoothing_alpha: float = DISPLAY_TRACK_SMOOTHING_ALPHA,
        min_depth_valid_ratio: float = MIN_TARGET_DEPTH_VALID_RATIO,
        max_depth_mad_mm: float = MAX_TARGET_DEPTH_MAD_MM,
    ) -> None:
        self.max_missed_frames = max(0, int(max_missed_frames))
        self.min_confidence = float(min_confidence)
        self.min_confirmations = max(1, int(min_confirmations))
        self.confirmation_window_ns = max(1, int(confirmation_seconds * 1_000_000_000))
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.01, 1.0))
        self.min_depth_valid_ratio = float(min_depth_valid_ratio)
        self.max_depth_mad_mm = float(max_depth_mad_mm)
        self._tracks: Dict[int, _VisualTrackedTarget] = {}
        self._confirmation_timestamps: Dict[int, Deque[int]] = {}
        self._previous_gray: Optional[np.ndarray] = None

    def _is_confirmation_detection(self, detection: Detection3D) -> bool:
        """只让高置信度、深度有效的垃圾桶检测进入光流候选池。"""

        x1, y1, x2, y2 = detection.bbox
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        aspect_ratio = box_width / box_height
        return bool(
            detection.confidence >= self.min_confidence
            and box_width >= VISUAL_TRACK_MIN_BOX_WIDTH_PX
            and box_height >= VISUAL_TRACK_MIN_BOX_HEIGHT_PX
            and VISUAL_TRACK_MIN_ASPECT_RATIO
            <= aspect_ratio
            <= VISUAL_TRACK_MAX_ASPECT_RATIO
            and detection.x_mm is not None
            and detection.y_mm is not None
            and detection.z_mm is not None
            and detection.depth_valid_ratio >= self.min_depth_valid_ratio
            and detection.depth_mad_mm is not None
            and detection.depth_mad_mm <= self.max_depth_mad_mm
        )

    def _confirmed_detection_indices(
        self,
        detections: Sequence[Detection3D],
    ) -> set[int]:
        """返回在最近确认窗口中出现足够次数的高质量检测。"""

        now_ns = max((detection.timestamp_ns for detection in detections), default=0)
        cutoff_ns = now_ns - self.confirmation_window_ns
        for target_id in list(self._confirmation_timestamps):
            history = self._confirmation_timestamps[target_id]
            while history and history[0] < cutoff_ns:
                history.popleft()
            if not history:
                del self._confirmation_timestamps[target_id]

        confirmed = set()
        for index, detection in enumerate(detections):
            if not self._is_confirmation_detection(detection):
                continue
            history = self._confirmation_timestamps.setdefault(
                detection.target_id,
                deque(),
            )
            history.append(detection.timestamp_ns)
            while history and history[0] < cutoff_ns:
                history.popleft()
            if len(history) >= self.min_confirmations:
                confirmed.add(index)
        return confirmed

    @staticmethod
    def _bbox_iou(
        first: Tuple[int, int, int, int],
        second: Tuple[int, int, int, int],
    ) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_first = max(1, first[2] - first[0]) * max(1, first[3] - first[1])
        area_second = max(1, second[2] - second[0]) * max(1, second[3] - second[1])
        union = area_first + area_second - intersection
        return float(intersection / max(union, 1))

    @staticmethod
    def _center_distance_ratio(
        first: Tuple[int, int, int, int],
        second: Tuple[int, int, int, int],
    ) -> float:
        first_center = np.asarray(
            ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
        )
        second_center = np.asarray(
            ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
        )
        diagonal = max(
            1.0,
            math.hypot(first[2] - first[0], first[3] - first[1]),
            math.hypot(second[2] - second[0], second[3] - second[1]),
        )
        return float(np.linalg.norm(first_center - second_center) / diagonal)

    @staticmethod
    def _clamp_bbox(
        bbox: Tuple[float, float, float, float],
        image_shape: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        image_h, image_w = image_shape
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        x1 = max(0, min(x1, image_w - 1))
        y1 = max(0, min(y1, image_h - 1))
        x2 = max(x1 + 1, min(x2, image_w))
        y2 = max(y1 + 1, min(y2, image_h))
        return x1, y1, x2, y2

    @staticmethod
    def _seed_points(
        gray: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """只在垃圾桶框内提取可跟踪特征，排除框外背景。"""

        x1, y1, x2, y2 = bbox
        mask = np.zeros(gray.shape, dtype=np.uint8)
        # 避开检测框边缘，降低背景特征把框带偏的概率。
        margin = max(2, min(8, min(x2 - x1, y2 - y1) // 8))
        inner_x1 = min(x2 - 1, x1 + margin)
        inner_y1 = min(y2 - 1, y1 + margin)
        inner_x2 = max(inner_x1 + 1, x2 - margin)
        inner_y2 = max(inner_y1 + 1, y2 - margin)
        mask[inner_y1:inner_y2, inner_x1:inner_x2] = 255
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=60,
            qualityLevel=0.01,
            minDistance=3,
            mask=mask,
            blockSize=5,
        )

    def _move_track_with_optical_flow(
        self,
        track: _VisualTrackedTarget,
        current_gray: np.ndarray,
    ) -> bool:
        """把上一帧桶内特征点跟到当前帧，并以中位位移移动检测框。"""

        if self._previous_gray is None or track.points is None or len(track.points) < 3:
            return False

        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            current_gray,
            track.points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_points is None or status is None:
            return False

        valid = status.reshape(-1).astype(bool)
        if errors is not None:
            valid &= np.isfinite(errors.reshape(-1)) & (errors.reshape(-1) <= 25.0)
        valid &= np.isfinite(next_points.reshape(-1, 2)).all(axis=1)
        if int(np.count_nonzero(valid)) < 3:
            return False

        previous_valid = track.points.reshape(-1, 2)[valid]
        next_valid = next_points.reshape(-1, 2)[valid]
        offsets = next_valid - previous_valid
        dx, dy = np.median(offsets, axis=0)
        diagonal = math.hypot(track.bbox[2] - track.bbox[0], track.bbox[3] - track.bbox[1])
        if not np.isfinite((dx, dy)).all() or math.hypot(float(dx), float(dy)) > max(20.0, diagonal * 0.60):
            return False

        x1, y1, x2, y2 = track.bbox
        track.bbox = self._clamp_bbox(
            (x1 + float(dx), y1 + float(dy), x2 + float(dx), y2 + float(dy)),
            current_gray.shape,
        )
        track.points = next_valid.reshape(-1, 1, 2).astype(np.float32)
        return True

    def _fuse_detection_bbox(
        self,
        predicted_bbox: Tuple[int, int, int, int],
        measured_bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        """将光流预测和当前 YOLO 框融合，消除逐帧框边缘抖动。"""

        alpha = self.smoothing_alpha
        fused = tuple(
            (1.0 - alpha) * predicted + alpha * measured
            for predicted, measured in zip(predicted_bbox, measured_bbox)
        )
        return self._clamp_bbox(fused, image_shape)

    def _replace_with_detection(
        self,
        track: _VisualTrackedTarget,
        detection: Detection3D,
        gray: np.ndarray,
    ) -> None:
        # 先前一步已通过 LK 光流把框推进到当前帧时，使用预测与实测融合。
        # 首次建轨或光流无效时保留实测框，避免把错误预测带入显示。
        bbox = (
            self._fuse_detection_bbox(track.bbox, detection.bbox, gray.shape)
            if track.flow_updated
            else detection.bbox
        )
        track.target_id = detection.target_id
        track.class_name = detection.class_name
        track.detection = replace(detection, display_predicted=False)
        track.bbox = bbox
        track.points = self._seed_points(gray, track.bbox)
        track.missed = 0
        track.flow_updated = True

    def update(self, image_bgr: np.ndarray, detections: Sequence[Detection3D]) -> None:
        """以本帧 RGB 图和 YOLO 结果更新所有显示轨迹。"""

        current_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 先按上一帧的目标纹理推进框，再用通过确认门槛的 YOLO 检测纠偏。
        flow_results: Dict[int, bool] = {}
        for track in self._tracks.values():
            track.flow_updated = self._move_track_with_optical_flow(track, current_gray)
            flow_results[track.target_id] = track.flow_updated

        unmatched_track_ids = set(self._tracks)
        unmatched_detection_indices = self._confirmed_detection_indices(detections)

        # 正常情况下 target_id 已由 TemporalStabilityTracker 延续，优先按 ID 关联。
        for index, detection in enumerate(detections):
            if index not in unmatched_detection_indices:
                continue
            track = self._tracks.get(detection.target_id)
            if track is None or track.target_id not in unmatched_track_ids:
                continue
            self._replace_with_detection(track, detection, current_gray)
            unmatched_track_ids.remove(track.target_id)
            unmatched_detection_indices.remove(index)

        # 长时间漏检后 TemporalStabilityTracker 会重建 target_id；按当前框位置重捕获。
        candidates: List[Tuple[float, int, int]] = []
        for target_id in unmatched_track_ids:
            track = self._tracks[target_id]
            for index in unmatched_detection_indices:
                detection = detections[index]
                if track.class_name != detection.class_name:
                    continue
                iou = self._bbox_iou(track.bbox, detection.bbox)
                distance = self._center_distance_ratio(track.bbox, detection.bbox)
                if iou >= 0.05 or distance <= 0.80:
                    candidates.append((max(iou, 1.0 - distance), target_id, index))
        for _, target_id, index in sorted(candidates, reverse=True):
            if target_id not in unmatched_track_ids or index not in unmatched_detection_indices:
                continue
            track = self._tracks[target_id]
            self._replace_with_detection(track, detections[index], current_gray)
            unmatched_track_ids.remove(target_id)
            unmatched_detection_indices.remove(index)

        for index in unmatched_detection_indices:
            detection = detections[index]
            self._tracks[detection.target_id] = _VisualTrackedTarget(
                target_id=detection.target_id,
                class_name=detection.class_name,
                detection=replace(detection, display_predicted=False),
                bbox=detection.bbox,
                points=self._seed_points(current_gray, detection.bbox),
            )

        for target_id in list(unmatched_track_ids):
            track = self._tracks[target_id]
            # 光流失败时不固定显示旧框，立即撤销，避免把错误目标留在画面上。
            if not flow_results.get(target_id, False):
                del self._tracks[target_id]
                continue
            track.missed += 1
            if track.missed > self.max_missed_frames:
                del self._tracks[target_id]

        self._previous_gray = current_gray

    def display_detections(self) -> List[Detection3D]:
        """返回当前应绘制的检测/光流预测框，不改变 JSONL 的真实检测结果。"""

        result: List[Detection3D] = []
        for target_id in sorted(self._tracks):
            track = self._tracks[target_id]
            detection = track.detection
            old_x1, old_y1, old_x2, old_y2 = detection.bbox
            new_x1, new_y1, new_x2, new_y2 = track.bbox
            offset_x = int(round(((new_x1 + new_x2) - (old_x1 + old_x2)) / 2.0))
            offset_y = int(round(((new_y1 + new_y2) - (old_y1 + old_y2)) / 2.0))
            pixel_u = None if detection.pixel_u is None else detection.pixel_u + offset_x
            pixel_v = None if detection.pixel_v is None else detection.pixel_v + offset_y
            result.append(
                replace(
                    detection,
                    bbox=track.bbox,
                    pixel_u=pixel_u,
                    pixel_v=pixel_v,
                    display_predicted=track.missed > 0,
                )
            )
        return result


def combine_detected_and_tracked_for_display(
    detections: Sequence[Detection3D],
    tracked_detections: Sequence[Detection3D],
) -> List[Detection3D]:
    """已确认目标优先显示光流/检测融合框，未确认目标仍显示原始 YOLO 框。"""

    result: List[Detection3D] = []
    remaining_tracked = list(tracked_detections)
    for direct in detections:
        matched_index = next(
            (
                index
                for index, tracked in enumerate(remaining_tracked)
                if direct.class_name == tracked.class_name
                and (
                    direct.target_id == tracked.target_id
                    or VisualObjectTracker._bbox_iou(direct.bbox, tracked.bbox) >= 0.10
                    or VisualObjectTracker._center_distance_ratio(direct.bbox, tracked.bbox)
                    <= 0.60
                )
            ),
            None,
        )
        if matched_index is None:
            result.append(direct)
        else:
            result.append(remaining_tracked.pop(matched_index))
    result.extend(remaining_tracked)
    return result


# ==============================================================================
# 13. YOLO Result + Depth -> Detection3D
# ==============================================================================

def get_class_name(names, class_id: int) -> str:
    """兼容 result.names 为 dict 或 list/tuple 的情况。"""

    if isinstance(names, dict):
        return str(names.get(class_id, class_id))

    if 0 <= class_id < len(names):
        return str(names[class_id])

    return str(class_id)


def is_trash_class(class_name: str) -> bool:
    """判断类别名是否为垃圾桶（忽略大小写与首尾空格）。"""

    return class_name.strip().lower() in TRASH_CLASS_NAMES


def process_yolo_result(
    image: np.ndarray,
    depth_map: np.ndarray,
    result,
    intrinsics: CameraIntrinsics,
    raw_depth_map: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Detection3D]]:
    """
    YOLO11 结果 + 对齐后的 Depth -> 三维检测结果。

    Detect：
        bbox -> center ROI -> depth cluster -> XYZ

    Segment：
        instance mask -> erosion -> depth cluster -> XYZ
    """

    output_image = image.copy()
    detections_3d: List[Detection3D] = []

    image_h, image_w = image.shape[:2]

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return output_image, detections_3d

    xyxy_array = (
        boxes.xyxy
        .detach()
        .cpu()
        .numpy()
    )

    # 对 YOLO11 + Ultralytics 高层 API：
    # boxes.conf 就是最终检测置信度，不要再额外乘一次。
    confidence_array = (
        boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    class_id_array = (
        boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(np.int32)
    )

    masks_array = None

    if result.masks is not None:
        masks_array = (
            result.masks.data
            .detach()
            .cpu()
            .numpy()
        )

    for index in range(len(xyxy_array)):
        x1, y1, x2, y2 = (
            xyxy_array[index]
            .astype(np.int32)
            .tolist()
        )

        x1 = max(0, min(x1, image_w - 1))
        y1 = max(0, min(y1, image_h - 1))
        x2 = max(x1 + 1, min(x2, image_w))
        y2 = max(y1 + 1, min(y2, image_h))

        bbox = (x1, y1, x2, y2)

        confidence = float(confidence_array[index])
        class_id = int(class_id_array[index])
        class_name = get_class_name(result.names, class_id)

        # 只识别垃圾桶：非垃圾桶类别直接跳过。
        if not is_trash_class(class_name):
            continue

        segmentation_mask = None

        # ----------------------------------------------------------------------
        # Seg 模型：优先使用 instance mask
        # ----------------------------------------------------------------------
        if masks_array is not None and index < len(masks_array):
            segmentation_mask = prepare_segmentation_mask(
                masks_array[index],
                (image_h, image_w),
            )

            if np.count_nonzero(segmentation_mask) >= MIN_VALID_DEPTH_POINTS:
                depth_region = segmentation_mask
                depth_source = "seg_mask"
            else:
                depth_region = create_center_roi_mask(
                    (image_h, image_w),
                    bbox,
                )
                depth_source = "center_roi"

        # ----------------------------------------------------------------------
        # Detect 模型：中心 ROI
        # ----------------------------------------------------------------------
        else:
            depth_region = create_center_roi_mask(
                (image_h, image_w),
                bbox,
            )
            depth_source = "center_roi"

        depth_estimate = estimate_depth_from_region(
            depth_map,
            depth_region,
            intrinsics=intrinsics,
            raw_depth_map=raw_depth_map,
        )

        if depth_estimate is not None:
            u = depth_estimate.u
            v = depth_estimate.v
            depth_mm = depth_estimate.depth_mm

            x_mm, y_mm, z_mm = deproject_pixel_to_3d(
                u,
                v,
                depth_mm,
                intrinsics,
            )

            valid_points = depth_estimate.valid_points
            cluster_points = depth_estimate.cluster_points
            region_points = depth_estimate.region_points
            depth_valid_ratio = depth_estimate.valid_ratio
            depth_mad_mm = depth_estimate.depth_mad_mm
            depth_std_mm = depth_estimate.depth_std_mm
            raw_valid_points = depth_estimate.raw_valid_points
            filled_points = depth_estimate.filled_points
            points_camera_mm = depth_estimate.points_camera_mm

        else:
            u = None
            v = None
            depth_mm = None
            x_mm = None
            y_mm = None
            z_mm = None
            valid_points = 0
            cluster_points = 0
            region_points = int(np.count_nonzero(depth_region))
            depth_valid_ratio = 0.0
            depth_mad_mm = None
            depth_std_mm = None
            raw_valid_points = 0
            filled_points = 0
            points_camera_mm = np.empty((0, 3), dtype=np.float32)

        detection = Detection3D(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            bbox=bbox,
            pixel_u=u,
            pixel_v=v,
            depth_mm=depth_mm,
            x_mm=x_mm,
            y_mm=y_mm,
            z_mm=z_mm,
            depth_source=depth_source,
            valid_depth_points=valid_points,
            cluster_depth_points=cluster_points,
            region_depth_points=region_points,
            depth_valid_ratio=depth_valid_ratio,
            depth_mad_mm=depth_mad_mm,
            depth_std_mm=depth_std_mm,
            raw_valid_depth_points=raw_valid_points,
            filled_depth_points=filled_points,
            point_cloud_camera_mm=points_camera_mm,
        )

        detections_3d.append(detection)

    return output_image, detections_3d


def finalize_detection_for_grasp(
    detection: Detection3D,
    frame_id: int,
    timestamp_ns: int,
    environment: EnvironmentQuality,
    quality_config: QualityConfig,
    intrinsics: CameraIntrinsics,
    depth_map: np.ndarray,
    extrinsics: Optional[Extrinsics],
    max_jaw_opening_mm: float,
    pregrasp_distance_mm: float,
    grasp_margin_mm: float,
    min_detection_confidence: float,
    workspace_min_xyz_mm: Optional[Sequence[float]] = None,
    workspace_max_xyz_mm: Optional[Sequence[float]] = None,
) -> None:
    """把三维检测补齐为可解释的抓取候选和安全拒绝结论。"""

    detection.frame_id = int(frame_id)
    detection.timestamp_ns = int(timestamp_ns)

    geometry, candidates = create_grasp_candidates(
        detection.point_cloud_camera_mm
        if detection.point_cloud_camera_mm is not None
        else np.empty((0, 3), dtype=np.float32),
        depth_map,
        intrinsics,
        pregrasp_distance_mm,
        grasp_margin_mm,
    )
    candidate = select_grasp_candidate(candidates, max_jaw_opening_mm)

    if geometry is not None:
        detection.object_width_mm = geometry.width_mm
        detection.object_height_mm = geometry.height_mm
        detection.object_depth_mm = geometry.depth_mm

    if candidate is not None:
        detection.grasp_type = candidate.grasp_type
        detection.required_jaw_opening_mm = candidate.required_jaw_opening_mm
        detection.approach_path_clear = candidate.path_clear
        detection.grasp_point_camera_xyz_mm = candidate.grasp_point_camera_xyz_mm
        detection.pregrasp_point_camera_xyz_mm = candidate.pregrasp_point_camera_xyz_mm
        detection.approach_vector_camera = candidate.approach_vector_camera
        detection.orientation_quaternion_camera_xyzw = (
            candidate.orientation_camera_quaternion_xyzw
        )

        if extrinsics is not None:
            detection.coordinate_frame = extrinsics.target_frame
            if detection.x_mm is not None and detection.y_mm is not None and detection.z_mm is not None:
                detection.position_base_xyz_mm = transform_point_to_base(
                    extrinsics,
                    (detection.x_mm, detection.y_mm, detection.z_mm),
                )
            detection.grasp_point_base_xyz_mm = transform_point_to_base(
                extrinsics,
                candidate.grasp_point_camera_xyz_mm,
            )
            detection.pregrasp_position_base_xyz_mm = transform_point_to_base(
                extrinsics,
                candidate.pregrasp_point_camera_xyz_mm,
            )
            detection.approach_vector_base = transform_vector_to_base(
                extrinsics,
                candidate.approach_vector_camera,
            )
            camera_rotation = make_gripper_rotation(
                np.asarray(candidate.jaw_axis_camera, dtype=np.float64),
                np.asarray(candidate.approach_vector_camera, dtype=np.float64),
            )
            base_rotation = extrinsics.rotation_base_from_camera @ camera_rotation
            detection.orientation_quaternion_xyzw = (
                rotation_matrix_to_quaternion_xyzw(base_rotation)
            )
            center_in_workspace = is_point_in_workspace(
                detection.position_base_xyz_mm
                if detection.position_base_xyz_mm is not None
                else candidate.grasp_point_camera_xyz_mm,
                workspace_min_xyz_mm,
                workspace_max_xyz_mm,
            )
            pregrasp_in_workspace = is_point_in_workspace(
                detection.pregrasp_position_base_xyz_mm,
                workspace_min_xyz_mm,
                workspace_max_xyz_mm,
            )
            if center_in_workspace is None or pregrasp_in_workspace is None:
                detection.within_robot_workspace = None
            else:
                detection.within_robot_workspace = bool(
                    center_in_workspace and pregrasp_in_workspace
                )
        else:
            detection.coordinate_frame = "camera_color_optical_frame"

    reasons: List[str] = []
    if detection.confidence < min_detection_confidence:
        reasons.append("CONFIDENCE_TOO_LOW")

    if detection.depth_mm is None or detection.point_cloud_camera_mm is None:
        reasons.append("DEPTH_UNAVAILABLE")
    elif detection.depth_valid_ratio < quality_config.min_target_depth_valid_ratio:
        reasons.append("DEPTH_VALID_RATIO_TOO_LOW")

    if detection.depth_mad_mm is None:
        reasons.append("DEPTH_VARIANCE_UNAVAILABLE")
    elif detection.depth_mad_mm > quality_config.max_target_depth_mad_mm:
        reasons.append("DEPTH_MAD_TOO_HIGH")

    if detection.cluster_depth_points > 0:
        filled_ratio = detection.filled_depth_points / max(
            detection.cluster_depth_points,
            1,
        )
        if filled_ratio > MAX_FILLED_DEPTH_RATIO:
            reasons.append("DEPTH_HOLE_FILLED_HEAVILY")

    if not detection.stable:
        if detection.stable_frame_count < quality_config.min_stable_frames:
            reasons.append("STABILITY_INSUFFICIENT")
        else:
            reasons.append("POSITION_UNSTABLE")

    if not environment.grasp_safe:
        reasons.extend(
            flag
            for flag in environment.flags
            if flag in HARD_ENVIRONMENT_FLAGS
        )

    if extrinsics is None:
        reasons.append("BASE_EXTRINSICS_UNAVAILABLE")
    elif detection.within_robot_workspace is None:
        reasons.append("ROBOT_WORKSPACE_UNCONFIGURED")
    elif not detection.within_robot_workspace:
        reasons.append("OUT_OF_ROBOT_WORKSPACE")

    if candidate is None:
        reasons.append("NO_GRASP_CANDIDATE")
    else:
        if candidate.required_jaw_opening_mm > max_jaw_opening_mm:
            reasons.append("REQUIRED_JAW_WIDTH_EXCEEDED")
        if not candidate.path_clear:
            reasons.append("APPROACH_PATH_BLOCKED")

    # 质量分数用于排序和论文统计；它不会绕过任何拒绝条件。
    confidence_score = clamp01(detection.confidence)
    depth_score = clamp01(
        (detection.depth_valid_ratio - quality_config.min_target_depth_valid_ratio)
        / max(1.0 - quality_config.min_target_depth_valid_ratio, 1e-6)
    )
    if detection.depth_mad_mm is None:
        consistency_score = 0.0
    else:
        consistency_score = clamp01(
            1.0 - detection.depth_mad_mm
            / max(2.0 * quality_config.max_target_depth_mad_mm, 1.0)
        )
    stability_score = clamp01(
        detection.stable_frame_count
        / max(quality_config.stability_window_frames, 1)
    )
    if detection.position_std_mm is not None:
        stability_score *= clamp01(
            1.0 - detection.position_std_mm
            / max(quality_config.max_position_std_mm, 1e-6)
        )
    environment_score = 1.0 if environment.grasp_safe else 0.0
    candidate_score = candidate.score if candidate is not None else 0.0
    path_score = 1.0 if candidate is not None and candidate.path_clear else 0.0
    detection.grasp_score = float(
        0.20 * confidence_score
        + 0.20 * depth_score
        + 0.15 * consistency_score
        + 0.20 * stability_score
        + 0.10 * environment_score
        + 0.10 * candidate_score
        + 0.05 * path_score
    )

    unique_reasons = list(dict.fromkeys(reasons))
    detection.graspable = not unique_reasons
    if detection.graspable:
        detection.status = "READY"
        detection.reject_reason = "NONE"
    elif "REQUIRED_JAW_WIDTH_EXCEEDED" in unique_reasons:
        detection.status = "NOT_GRASPABLE"
        detection.reject_reason = ";".join(unique_reasons)
    else:
        detection.status = "NOT_READY"
        detection.reject_reason = ";".join(unique_reasons) or "QUALITY_REJECTED"


def format_detection_label(detection: Detection3D) -> List[str]:
    """生成 GUI 标签；相机坐标显示为更易读的 cm。"""

    return ["".join(text for text, _ in format_detection_label_parts(detection))]


def format_detection_label_parts(
    detection: Detection3D,
) -> List[Tuple[str, Tuple[int, int, int]]]:
    """把 XYZ 标签拆为绿色的 X/Y/单位和红色的 Z 深度数字。"""

    if any(value is None for value in (detection.x_mm, detection.y_mm, detection.z_mm)):
        return [("XYZ=N/A", GREEN)]
    return [
        (
            f"XYZ=({detection.x_mm / 10.0:.1f},"
            f"{detection.y_mm / 10.0:.1f},",
            GREEN,
        ),
        (f"{detection.z_mm / 10.0:.1f}", RED),
        (") cm", GREEN),
    ]


def annotate_detection_quality(
    image: np.ndarray,
    detections: Sequence[Detection3D],
) -> None:
    """绘制检测/光流跟踪框、中心红色横杠和大号 XYZ 标注。"""

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        # 黄框表示当前帧由光流跟踪补齐；正常检测框保持类别色。
        color = YELLOW if detection.display_predicted else PALETTE[
            detection.class_id % len(PALETTE)
        ]
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            3,
        )

        # 保留早期“中心红点”的语义，只把点横向扩展成短横杠。
        # 有有效深度时使用实际 XYZ 测量像素；深度无效时退回 bbox 中心，
        # 让画面仍能明确显示检测目标的位置，此时坐标标签会显示 XYZ=N/A。
        image_h, image_w = image.shape[:2]
        if detection.pixel_u is not None and detection.pixel_v is not None:
            center_u = int(detection.pixel_u)
            center_v = int(detection.pixel_v)
        else:
            center_u = int(round((x1 + x2) / 2.0))
            center_v = int(round((y1 + y2) / 2.0))

        center_u = max(0, min(center_u, image_w - 1))
        center_v = max(0, min(center_v, image_h - 1))
        half_width = max(12, min(40, (x2 - x1) // 6))
        line_x1 = max(0, center_u - half_width)
        line_x2 = min(image_w - 1, center_u + half_width)
        cv2.line(
            image,
            (line_x1, center_v),
            (line_x2, center_v),
            RED,
            4,
            cv2.LINE_AA,
        )

        draw_prominent_xyz_label(
            image,
            detection,
            x1,
            max(30, y1 - 10),
        )


def annotate_environment_quality(
    image: np.ndarray,
    quality: EnvironmentQuality,
) -> None:
    """把环境质量指标固定显示在画面左上角。"""

    color = GREEN if quality.grasp_safe else RED
    lines = [
        f"ENV={quality.status} "
        f"bright={quality.brightness_mean:.1f} "
        f"dark={quality.dark_pixel_ratio:.0%} "
        f"over={quality.overexposed_ratio:.0%}",
        f"contrast={quality.contrast_std:.1f} "
        f"depth_valid={quality.depth_valid_ratio:.0%}",
    ]
    draw_multiline_label(image, lines, 8, 48, color)


def environment_to_dict(quality: EnvironmentQuality) -> dict:
    return {
        "brightness_mean": quality.brightness_mean,
        "dark_pixel_ratio": quality.dark_pixel_ratio,
        "overexposed_ratio": quality.overexposed_ratio,
        "contrast_std": quality.contrast_std,
        "depth_valid_ratio": quality.depth_valid_ratio,
        "flags": list(quality.flags),
        "status": quality.status,
        "grasp_safe": quality.grasp_safe,
    }


def detection_to_dict(detection: Detection3D) -> dict:
    """转换成可直接写入 JSONL 的完整抓取结果。"""

    camera_position = None
    if detection.x_mm is not None and detection.y_mm is not None and detection.z_mm is not None:
        camera_position = [detection.x_mm, detection.y_mm, detection.z_mm]

    approach_vector = (
        list(detection.approach_vector_base)
        if detection.approach_vector_base is not None
        else (
            list(detection.approach_vector_camera)
            if detection.approach_vector_camera is not None
            else None
        )
    )
    orientation = (
        list(detection.orientation_quaternion_xyzw)
        if detection.orientation_quaternion_xyzw is not None
        else None
    )
    return {
        "target_id": detection.target_id,
        "timestamp": detection.timestamp_ns / 1_000_000_000.0,
        "timestamp_ns": detection.timestamp_ns,
        "frame_id": detection.frame_id,
        "class_name": detection.class_name,
        "detection_confidence": detection.confidence,
        "position_camera_xyz_mm": camera_position,
        "filtered_position_camera_xyz_mm": (
            list(detection.filtered_camera_xyz_mm)
            if detection.filtered_camera_xyz_mm is not None
            else None
        ),
        "filter_measurement_sigma_mm": detection.filter_measurement_sigma_mm,
        "position_base_xyz_mm": (
            list(detection.position_base_xyz_mm)
            if detection.position_base_xyz_mm is not None
            else None
        ),
        "grasp_point_camera_xyz_mm": (
            list(detection.grasp_point_camera_xyz_mm)
            if detection.grasp_point_camera_xyz_mm is not None
            else None
        ),
        "grasp_point_base_xyz_mm": (
            list(detection.grasp_point_base_xyz_mm)
            if detection.grasp_point_base_xyz_mm is not None
            else None
        ),
        "pregrasp_position_xyz_mm": (
            list(detection.pregrasp_position_base_xyz_mm)
            if detection.pregrasp_position_base_xyz_mm is not None
            else None
        ),
        "pregrasp_position_camera_xyz_mm": (
            list(detection.pregrasp_point_camera_xyz_mm)
            if detection.pregrasp_point_camera_xyz_mm is not None
            else None
        ),
        "approach_vector": approach_vector,
        "approach_vector_camera": (
            list(detection.approach_vector_camera)
            if detection.approach_vector_camera is not None
            else None
        ),
        "approach_vector_base": (
            list(detection.approach_vector_base)
            if detection.approach_vector_base is not None
            else None
        ),
        "orientation_quaternion": orientation,
        "orientation_quaternion_camera": (
            list(detection.orientation_quaternion_camera_xyzw)
            if detection.orientation_quaternion_camera_xyzw is not None
            else None
        ),
        "coordinate_frame": detection.coordinate_frame,
        "within_robot_workspace": detection.within_robot_workspace,
        "object_width_mm": detection.object_width_mm,
        "object_height_mm": detection.object_height_mm,
        "object_depth_mm": detection.object_depth_mm,
        "required_jaw_opening_mm": detection.required_jaw_opening_mm,
        "grasp_type": detection.grasp_type,
        "depth_valid_ratio": detection.depth_valid_ratio,
        "depth_mad_mm": detection.depth_mad_mm,
        "depth_std_mm": detection.depth_std_mm,
        "position_std_mm": detection.position_std_mm,
        "stable_frame_count": detection.stable_frame_count,
        "approach_path_clear": detection.approach_path_clear,
        "grasp_score": detection.grasp_score,
        "graspable": detection.graspable,
        "status": detection.status,
        "reject_reason": detection.reject_reason,
        "depth_source": detection.depth_source,
        "bbox": list(detection.bbox),
    }


def frame_to_dict(
    frame_id: int,
    timestamp_ns: int,
    quality: EnvironmentQuality,
    detections: Sequence[Detection3D],
) -> dict:
    return {
        "frame_id": frame_id,
        "timestamp": timestamp_ns / 1_000_000_000.0,
        "timestamp_ns": timestamp_ns,
        "environment": environment_to_dict(quality),
        "detections": [detection_to_dict(detection) for detection in detections],
    }


def build_d435i_config(
    rs,
    serial: Optional[str],
    color_width: Optional[int],
    color_height: Optional[int],
    depth_width: Optional[int],
    depth_height: Optional[int],
    fps: int,
):
    """创建 D435i RGB-D 配置；深度会在主循环中对齐到彩色图像。"""

    color_width = color_width or DEFAULT_COLOR_WIDTH
    color_height = color_height or DEFAULT_COLOR_HEIGHT
    depth_width = depth_width or DEFAULT_DEPTH_WIDTH
    depth_height = depth_height or DEFAULT_DEPTH_HEIGHT

    config = rs.config()
    if serial:
        config.enable_device(serial)

    config.enable_stream(
        rs.stream.depth,
        depth_width,
        depth_height,
        rs.format.z16,
        fps,
    )
    config.enable_stream(
        rs.stream.color,
        color_width,
        color_height,
        rs.format.bgr8,
        fps,
    )

    print(
        "[Camera] D435i requested: "
        f"color={color_width}x{color_height}, "
        f"depth={depth_width}x{depth_height}, fps={fps}"
    )
    return config


def try_set_realsense_option(
    sensor,
    option,
    value: float,
    label: str,
) -> bool:
    """尽力设置 RealSense 选项；不同固件不支持时只给出可读警告。"""

    if option is None:
        return False
    try:
        if not sensor.supports(option):
            return False
        sensor.set_option(option, float(value))
        print(f"[Camera] {label}={value}")
        return True
    except Exception as error:
        print(f"[Camera] {label} not applied: {error}")
        return False


def configure_realsense_sensors(
    rs,
    profile,
    auto_exposure: bool,
    max_gain: Optional[float],
    laser_power: Optional[float],
) -> None:
    """配置自动曝光、最大增益、发射器和激光功率。"""

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    color_sensor = None
    try:
        color_sensor = device.first_color_sensor()
    except Exception:
        for sensor in device.query_sensors():
            try:
                if "RGB" in sensor.get_info(rs.camera_info.name).upper():
                    color_sensor = sensor
                    break
            except Exception:
                continue

    sensors = [sensor for sensor in (depth_sensor, color_sensor) if sensor is not None]
    enable_auto_exposure = getattr(rs.option, "enable_auto_exposure", None)
    auto_gain_limit = getattr(rs.option, "auto_gain_limit", None)
    laser_power_option = getattr(rs.option, "laser_power", None)
    emitter_option = getattr(rs.option, "emitter_enabled", None)

    for sensor in sensors:
        if auto_exposure:
            try_set_realsense_option(
                sensor,
                enable_auto_exposure,
                1.0,
                "enable_auto_exposure",
            )
        if max_gain is not None:
            try_set_realsense_option(
                sensor,
                auto_gain_limit,
                max_gain,
                "auto_gain_limit",
            )

    if laser_power is not None:
        try_set_realsense_option(
            depth_sensor,
            laser_power_option,
            laser_power,
            "laser_power",
        )

    # 开启投影器作为默认策略；强光时仍依赖质量拒绝机制，不假设软件能恢复被淹没的纹理。
    try_set_realsense_option(
        depth_sensor,
        emitter_option,
        1.0,
        "emitter_enabled",
    )


class RealSenseDepthPostProcessor:
    """按 RealSense 推荐顺序执行视差、空间、时间、深度和孔洞滤波。"""

    def __init__(self, rs) -> None:
        self._rs = rs
        self._to_disparity = rs.disparity_transform(True)
        self._spatial = rs.spatial_filter()
        self._temporal = rs.temporal_filter()
        self._to_depth = rs.disparity_transform(False)
        self._hole_filling = rs.hole_filling_filter()

        self._set_filter_option(
            self._spatial,
            getattr(rs.option, "filter_magnitude", None),
            2.0,
            "spatial.filter_magnitude",
        )
        self._set_filter_option(
            self._spatial,
            getattr(rs.option, "filter_smooth_alpha", None),
            0.5,
            "spatial.filter_smooth_alpha",
        )
        self._set_filter_option(
            self._spatial,
            getattr(rs.option, "filter_smooth_delta", None),
            20.0,
            "spatial.filter_smooth_delta",
        )
        self._set_filter_option(
            self._temporal,
            getattr(rs.option, "filter_smooth_alpha", None),
            0.4,
            "temporal.filter_smooth_alpha",
        )
        self._set_filter_option(
            self._temporal,
            getattr(rs.option, "filter_smooth_delta", None),
            20.0,
            "temporal.filter_smooth_delta",
        )
        self._set_filter_option(
            self._hole_filling,
            getattr(rs.option, "holes_fill", None),
            1.0,
            "hole_filling.mode",
        )

    @staticmethod
    def _set_filter_option(filter_object, option, value: float, label: str) -> None:
        if option is None:
            return
        try:
            filter_object.set_option(option, float(value))
        except Exception as error:
            print(f"[Depth] {label} not applied: {error}")

    def process(self, depth_frame):
        filtered = self._to_disparity.process(depth_frame)
        filtered = self._spatial.process(filtered)
        filtered = self._temporal.process(filtered)
        filtered = self._to_depth.process(filtered)
        filtered = self._hole_filling.process(filtered)
        return filtered


class OfflineDepthPostProcessor:
    """DB3 回放的保守 3x3 中值平滑，不做孔洞填充，也不伪造有效率。"""

    def process(self, depth_raw: np.ndarray) -> np.ndarray:
        if depth_raw.dtype != np.uint16:
            depth_raw = np.clip(depth_raw, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        smoothed = cv2.medianBlur(depth_raw, 3)
        smoothed[(depth_raw == 0) | (smoothed == 0)] = 0
        return smoothed


# ==============================================================================
# 14. Console output
# ==============================================================================

def print_detections(
    detections: List[Detection3D],
    environment: Optional[EnvironmentQuality] = None,
    frame_id: Optional[int] = None,
    timestamp_ns: Optional[int] = None,
) -> None:
    """把环境质量和完整抓取状态打印到终端。"""

    if environment is not None:
        print(
            f"[Environment] frame={frame_id} timestamp_ns={timestamp_ns} "
            f"status={environment.status} "
            f"brightness={environment.brightness_mean:.1f} "
            f"dark={environment.dark_pixel_ratio:.1%} "
            f"over={environment.overexposed_ratio:.1%} "
            f"contrast={environment.contrast_std:.1f} "
            f"depth_valid={environment.depth_valid_ratio:.1%}"
        )
    if not detections:
        print("[Detection] No objects.")
        return

    print("-" * 120)

    for detection in detections:
        if detection.z_mm is None:
            print(
                f"target={detection.target_id} {detection.class_name:<15} "
                f"conf={detection.confidence:.3f} "
                f"status={detection.status} XYZ=N/A "
                f"reason={detection.reject_reason}"
            )
            continue

        base_position = (
            "(" + ", ".join(f"{value:.1f}" for value in detection.position_base_xyz_mm) + ")"
            if detection.position_base_xyz_mm is not None
            else "N/A"
        )
        grasp_point = (
            "(" + ", ".join(f"{value:.1f}" for value in detection.grasp_point_base_xyz_mm) + ")"
            if detection.grasp_point_base_xyz_mm is not None
            else "N/A"
        )
        print(
            f"target={detection.target_id} {detection.class_name:<15} "
            f"frame={detection.frame_id} conf={detection.confidence:.3f} "
            f"status={detection.status} "
            f"cam_xyz=("
            f"{detection.x_mm:.1f}, "
            f"{detection.y_mm:.1f}, "
            f"{detection.z_mm:.1f}) mm  "
            f"base_xyz={base_position} grasp={grasp_point} "
            f"grasp={detection.grasp_type or 'N/A'} "
            f"jaw={detection.required_jaw_opening_mm or 0.0:.1f} mm "
            f"score={detection.grasp_score:.2f} "
            f"valid={detection.depth_valid_ratio:.1%} "
            f"MAD={detection.depth_mad_mm or 0.0:.1f} mm "
            f"std={detection.position_std_mm or 0.0:.1f} mm "
            f"depth_points="
            f"{detection.cluster_depth_points}/"
            f"{detection.valid_depth_points} "
            f"source={detection.depth_source} "
            f"reason={detection.reject_reason}"
        )


# ==============================================================================
# 14. Main
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Intel RealSense D435i + YOLO11 RGB-D 3D object detection"
    )

    # --------------------------------------------------------------------------
    # YOLO
    # --------------------------------------------------------------------------
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="YOLO model path (default: bundled garbage-bin best.pt)",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD,
        help="YOLO confidence threshold",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=NMS_IOU_THRESHOLD,
        help="YOLO NMS IoU threshold",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=YOLO_IMAGE_SIZE,
        help="YOLO inference image size",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="auto / cpu / 0 / 1 ...",
    )

    parser.add_argument(
        "--max_det",
        type=int,
        default=MAX_DETECTIONS,
        help="Maximum detections per frame",
    )

    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help=(
            "Comma-separated class names to keep (e.g. 'trash'). "
            "Defaults to the built-in trash bin class list."
        ),
    )

    parser.add_argument(
        "--no_warmup",
        action="store_true",
        help="Skip the initial YOLO warm-up inference",
    )
    parser.add_argument(
        "--adaptive_rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply environment-aware Gamma/CLAHE only for low-light/low-contrast frames",
    )

    # --------------------------------------------------------------------------
    # Camera
    # --------------------------------------------------------------------------
    parser.add_argument("--color_width", type=int, default=None)
    parser.add_argument("--color_height", type=int, default=None)
    parser.add_argument("--depth_width", type=int, default=None)
    parser.add_argument("--depth_height", type=int, default=None)
    parser.add_argument(
        "--rs_serial",
        type=str,
        default=None,
        help="Optional D435i serial number when more than one RealSense is connected",
    )
    parser.add_argument(
        "--db3",
        type=str,
        default=None,
        help=(
            "RealSense Viewer .db3 recording to play back. "
            "When set, no physical camera or ROS 2 installation is required."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="D435i RGB and depth stream rate",
    )
    parser.add_argument(
        "--auto_exposure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable RealSense auto exposure when using a physical camera",
    )
    parser.add_argument(
        "--exposure_settle_frames",
        type=int,
        default=5,
        help="Ignore READY during this many initial live-camera frames",
    )
    parser.add_argument(
        "--max_gain",
        type=float,
        default=DEFAULT_MAX_GAIN,
        help="RealSense auto gain limit; use --max_gain -1 to leave it unchanged",
    )
    parser.add_argument(
        "--laser_power",
        type=float,
        default=None,
        help="Optional RealSense projector laser power; leave unset to preserve device setting",
    )
    parser.add_argument(
        "--depth_filters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SDK depth filters for live camera, conservative median filter for DB3",
    )

    # --------------------------------------------------------------------------
    # 抓取质量 / 坐标
    # --------------------------------------------------------------------------
    parser.add_argument(
        "--extrinsics",
        type=str,
        default=None,
        help="JSON camera_color_optical_frame -> robot_base extrinsics",
    )
    parser.add_argument(
        "--max_jaw_opening_mm",
        type=float,
        default=DEFAULT_MAX_JAW_OPENING_MM,
        help="Maximum gripper opening used for grasp rejection",
    )
    parser.add_argument(
        "--pregrasp_distance_mm",
        type=float,
        default=DEFAULT_PREGRASP_DISTANCE_MM,
        help="Distance from pre-grasp point to grasp point",
    )
    parser.add_argument(
        "--grasp_margin_mm",
        type=float,
        default=DEFAULT_GRASP_MARGIN_MM,
        help="Safety margin added to estimated object width",
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=None,
        help="Minimum confidence for READY; default is max(--conf, 0.20)",
    )
    parser.add_argument(
        "--workspace_min_xyz_mm",
        type=str,
        default=None,
        help="Minimum robot-base workspace as x,y,z in mm; required for READY",
    )
    parser.add_argument(
        "--workspace_max_xyz_mm",
        type=str,
        default=None,
        help="Maximum robot-base workspace as x,y,z in mm; required for READY",
    )
    parser.add_argument("--min_target_depth_valid_ratio", type=float, default=MIN_TARGET_DEPTH_VALID_RATIO)
    parser.add_argument("--max_target_depth_mad_mm", type=float, default=MAX_TARGET_DEPTH_MAD_MM)
    parser.add_argument("--stability_window_frames", type=int, default=STABILITY_WINDOW_FRAMES)
    parser.add_argument("--min_stable_frames", type=int, default=MIN_STABLE_FRAMES)
    parser.add_argument("--max_position_std_mm", type=float, default=MAX_POSITION_STD_MM)
    parser.add_argument(
        "--display_track_max_missed_frames",
        type=int,
        default=DISPLAY_TRACK_MAX_MISSED_FRAMES,
        help=(
            "Keep a confirmed GUI optical-flow track through this many "
            "consecutive YOLO missed frames; 0 disables tracking"
        ),
    )
    parser.add_argument(
        "--visual_track_min_confidence",
        type=float,
        default=VISUAL_TRACK_MIN_CONFIDENCE,
        help="Minimum YOLO confidence for an optical-flow track confirmation",
    )
    parser.add_argument(
        "--visual_track_min_confirmations",
        type=int,
        default=VISUAL_TRACK_MIN_CONFIRMATIONS,
        help="Required high-quality detections within the confirmation window",
    )
    parser.add_argument(
        "--visual_track_confirmation_seconds",
        type=float,
        default=VISUAL_TRACK_CONFIRMATION_SECONDS,
        help="Seconds used to confirm a target before optical-flow tracking",
    )
    parser.add_argument(
        "--display_track_smoothing_alpha",
        type=float,
        default=DISPLAY_TRACK_SMOOTHING_ALPHA,
        help=(
            "YOLO measurement weight for a confirmed display track; "
            "smaller is smoother, 1.0 disables box smoothing"
        ),
    )
    parser.add_argument("--low_light_brightness", type=float, default=LOW_LIGHT_BRIGHTNESS)
    parser.add_argument("--dark_pixel_ratio", type=float, default=DARK_PIXEL_RATIO)
    parser.add_argument("--overexposed_ratio", type=float, default=OVEREXPOSED_RATIO)
    parser.add_argument("--low_contrast_std", type=float, default=LOW_CONTRAST_STD)
    parser.add_argument("--min_global_depth_valid_ratio", type=float, default=MIN_GLOBAL_DEPTH_VALID_RATIO)
    parser.add_argument("--fog_suspected_depth_ratio", type=float, default=FOG_SUSPECTED_DEPTH_RATIO)

    # --------------------------------------------------------------------------
    # Output
    # --------------------------------------------------------------------------
    parser.add_argument(
        "--print_result",
        action="store_true",
        help="Print 3D detections to terminal",
    )

    parser.add_argument(
        "--print_interval",
        type=float,
        default=1.0,
        help="Terminal print interval in seconds",
    )

    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Disable cv2 GUI for SSH/headless running",
    )

    parser.add_argument(
        "--show_runtime_osd",
        action="store_true",
        help="Show FPS/inference/device text in the GUI (off by default)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print additional first-frame debugging information",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Write one structured RGB-D frame result per line to this JSONL file",
    )
    parser.add_argument(
        "--out_video",
        type=str,
        default=None,
        help="Optional annotated MP4 output path; includes smoothed display boxes",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Stop after N frames; 0 means play until the source ends",
    )

    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be a positive integer")
    if args.max_frames < 0:
        parser.error("--max_frames must be >= 0")
    if args.print_interval < 0:
        parser.error("--print_interval must be >= 0")
    if not 0.0 < args.display_track_smoothing_alpha <= 1.0:
        parser.error("--display_track_smoothing_alpha must be in (0, 1]")
    if args.max_jaw_opening_mm <= 0:
        parser.error("--max_jaw_opening_mm must be positive")
    if args.pregrasp_distance_mm <= 0:
        parser.error("--pregrasp_distance_mm must be positive")
    if args.grasp_margin_mm < 0:
        parser.error("--grasp_margin_mm must be >= 0")
    if args.max_gain == 0 or args.max_gain < -1:
        parser.error("--max_gain must be -1 or a positive value")
    if args.exposure_settle_frames < 0:
        parser.error("--exposure_settle_frames must be >= 0")
    for option_name in (
        "dark_pixel_ratio",
        "overexposed_ratio",
        "min_target_depth_valid_ratio",
        "min_global_depth_valid_ratio",
        "fog_suspected_depth_ratio",
    ):
        option_value = getattr(args, option_name)
        if not 0.0 <= option_value <= 1.0:
            parser.error(f"--{option_name} must be between 0 and 1")
    if args.stability_window_frames < 1 or args.min_stable_frames < 1:
        parser.error("stability frame counts must be positive")
    if args.min_stable_frames > args.stability_window_frames:
        parser.error("--min_stable_frames cannot exceed --stability_window_frames")
    if args.display_track_max_missed_frames < 0:
        parser.error("--display_track_max_missed_frames must be >= 0")
    if not 0.0 <= args.visual_track_min_confidence <= 1.0:
        parser.error("--visual_track_min_confidence must be between 0 and 1")
    if args.visual_track_min_confirmations < 1:
        parser.error("--visual_track_min_confirmations must be positive")
    if args.visual_track_confirmation_seconds <= 0.0:
        parser.error("--visual_track_confirmation_seconds must be positive")

    # --------------------------------------------------------------------------
    # 类别过滤：只识别垃圾桶
    # --------------------------------------------------------------------------
    if args.classes is not None:
        TRASH_CLASS_NAMES = {
            name.strip().lower()
            for name in args.classes.split(",")
            if name.strip()
        }

    quality_config = QualityConfig(
        low_light_brightness=args.low_light_brightness,
        dark_pixel_ratio=args.dark_pixel_ratio,
        overexposed_ratio=args.overexposed_ratio,
        low_contrast_std=args.low_contrast_std,
        min_global_depth_valid_ratio=args.min_global_depth_valid_ratio,
        fog_suspected_depth_ratio=args.fog_suspected_depth_ratio,
        min_target_depth_valid_ratio=args.min_target_depth_valid_ratio,
        max_target_depth_mad_mm=args.max_target_depth_mad_mm,
        stability_window_frames=args.stability_window_frames,
        min_stable_frames=args.min_stable_frames,
        max_position_std_mm=args.max_position_std_mm,
    )
    min_detection_confidence = (
        max(args.conf, CONF_THRESHOLD)
        if args.min_detection_confidence is None
        else args.min_detection_confidence
    )
    if not 0.0 <= min_detection_confidence <= 1.0:
        parser.error("--min_detection_confidence must be between 0 and 1")
    try:
        workspace_min_xyz_mm = parse_xyz_triplet(
            args.workspace_min_xyz_mm,
            "--workspace_min_xyz_mm",
        )
        workspace_max_xyz_mm = parse_xyz_triplet(
            args.workspace_max_xyz_mm,
            "--workspace_max_xyz_mm",
        )
    except ValueError as error:
        parser.error(str(error))
    if (workspace_min_xyz_mm is None) != (workspace_max_xyz_mm is None):
        parser.error(
            "--workspace_min_xyz_mm and --workspace_max_xyz_mm must be supplied together"
        )
    if (
        workspace_min_xyz_mm is not None
        and workspace_max_xyz_mm is not None
        and not np.all(np.asarray(workspace_min_xyz_mm) <= np.asarray(workspace_max_xyz_mm))
    ):
        parser.error("workspace minimum values must not exceed maximum values")

    try:
        extrinsics = load_extrinsics(args.extrinsics)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[Error] Invalid extrinsics: {error}")
        return 1

    print(
        "[Quality] Thresholds: "
        f"target_depth_valid>={quality_config.min_target_depth_valid_ratio:.2f}, "
        f"target_MAD<={quality_config.max_target_depth_mad_mm:.1f} mm, "
        f"stable_frames>={quality_config.min_stable_frames}/"
        f"{quality_config.stability_window_frames}, "
        f"position_std<={quality_config.max_position_std_mm:.1f} mm, "
        f"jaw<={args.max_jaw_opening_mm:.1f} mm"
    )
    if extrinsics is None:
        print(
            "[Quality] No camera-to-base extrinsics supplied; "
            "base pose and READY are disabled."
        )
    elif workspace_min_xyz_mm is None:
        print(
            "[Quality] No robot workspace bounds supplied; "
            "READY is disabled until bounds are configured."
        )
    else:
        print(
            f"[Quality] Coordinate transform: "
            f"{extrinsics.source_frame} -> {extrinsics.target_frame}"
        )

    # --------------------------------------------------------------------------
    # Resolve device
    # --------------------------------------------------------------------------
    device = resolve_device(args.device)

    print_runtime_info(device)

    # CUDA 性能设置：固定输入尺寸时通常有利。
    if device != "cpu" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # --------------------------------------------------------------------------
    # GUI：尽可能早地创建窗口
    # --------------------------------------------------------------------------
    display_enabled = init_display(
        enabled=not args.no_display
    )

    # --------------------------------------------------------------------------
    # Validate model
    # --------------------------------------------------------------------------
    print("=" * 80)
    print("[YOLO] Loading model")
    print("=" * 80)
    print(f"[YOLO] Model: {args.model}")

    if not os.path.isfile(args.model):
        print("[Error] YOLO model file does not exist:")
        print(f"        {args.model}")
        print()
        print("Please check --model or DEFAULT_MODEL.")
        return 1

    show_status(
        display_enabled,
        [
            "OpenCV GUI OK.",
            "Loading YOLO model...",
            os.path.basename(args.model),
        ],
    )

    model = YOLO(args.model)

    print(f"[YOLO] Task: {getattr(model, 'task', 'unknown')}")
    print(f"[YOLO] Device requested/resolved: {args.device} -> {device}")
    print()

    # --------------------------------------------------------------------------
    # YOLO warm-up
    # --------------------------------------------------------------------------
    if not args.no_warmup:
        yolo_warmup(
            model=model,
            device=device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            display_enabled=display_enabled,
        )
    else:
        print("[YOLO] Warm-up skipped by --no_warmup")

    # --------------------------------------------------------------------------
    # Camera
    # --------------------------------------------------------------------------
    print("=" * 80)
    print("[Camera] Initializing")
    print("=" * 80)

    show_status(
        display_enabled,
        [
            "YOLO ready.",
            f"Device: {device}",
            "Opening RealSense Viewer DB3..." if args.db3 else "Starting D435i camera...",
        ],
    )

    pipeline = None
    db3_reader: Optional[RealSenseViewerDb3Reader] = None
    pipeline_started = False
    depth_scale_mm = None
    depth_postprocessor = None
    offline_depth_postprocessor = None
    json_handle = None
    video_writer = None
    tracker = TemporalStabilityTracker(
        quality_config.stability_window_frames,
        quality_config.min_stable_frames,
        quality_config.max_position_std_mm,
        1.0 / args.fps,
    )
    visual_tracker = VisualObjectTracker(
        max_missed_frames=args.display_track_max_missed_frames,
        min_confidence=args.visual_track_min_confidence,
        min_confirmations=args.visual_track_min_confirmations,
        confirmation_seconds=args.visual_track_confirmation_seconds,
        smoothing_alpha=args.display_track_smoothing_alpha,
        min_depth_valid_ratio=quality_config.min_target_depth_valid_ratio,
        max_depth_mad_mm=quality_config.max_target_depth_mad_mm,
    )
    processed_frames = 0
    ready_results = 0
    rejected_results = 0
    exposure_settle_remaining = 0

    try:
        if args.jsonl:
            json_path = os.path.abspath(args.jsonl)
            json_parent = os.path.dirname(json_path)
            if json_parent:
                os.makedirs(json_parent, exist_ok=True)
            json_handle = open(json_path, "w", encoding="utf-8")
            print(f"[Output] JSONL: {json_path}")

        if args.db3:
            db3_reader = RealSenseViewerDb3Reader(args.db3)
            depth_scale_mm = db3_reader.depth_scale_mm
            intrinsics: Optional[CameraIntrinsics] = db3_reader.color_intrinsics
            print(f"[Playback] RealSense Viewer DB3: {db3_reader.path}")
            print(
                "[Playback] Frames: "
                f"color={db3_reader.color_intrinsics.width}x"
                f"{db3_reader.color_intrinsics.height}, "
                f"depth={db3_reader.depth_intrinsics.width}x"
                f"{db3_reader.depth_intrinsics.height}"
            )
            print("[Playback] Depth-to-color alignment enabled from DB3 calibration.")
            if args.depth_filters:
                offline_depth_postprocessor = OfflineDepthPostProcessor()
                print("[Depth] DB3 mode: conservative median filter enabled; no hole filling.")
            exposure_settle_remaining = 0
        else:
            try:
                import pyrealsense2 as rs
            except (ImportError, RuntimeError) as error:
                print(f"[Error] D435i runtime setup failed: {error}")
                print("Run setup_windows.bat to install pyrealsense2.")
                return 1

            pipeline = rs.pipeline()
            align_filter = rs.align(rs.stream.color)
            config = build_d435i_config(
                rs,
                args.rs_serial,
                args.color_width,
                args.color_height,
                args.depth_width,
                args.depth_height,
                args.fps,
            )
            profile = pipeline.start(config)
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale_mm = float(depth_sensor.get_depth_scale()) * 1000.0
            configure_realsense_sensors(
                rs,
                profile,
                args.auto_exposure,
                None if args.max_gain < 0 else args.max_gain,
                args.laser_power,
            )
            if args.depth_filters:
                depth_postprocessor = RealSenseDepthPostProcessor(rs)
                print(
                    "[Depth] SDK filters: disparity -> spatial -> temporal "
                    "-> depth -> hole filling."
                )
            exposure_settle_remaining = args.exposure_settle_frames
            if exposure_settle_remaining:
                print(
                    f"[Camera] Exposure settling: first "
                    f"{exposure_settle_remaining} frames cannot be READY."
                )
            device_name = profile.get_device().get_info(rs.camera_info.name)
            print(f"[Camera] RealSense device: {device_name}")
            if "d435i" not in device_name.lower():
                print("[Warning] Selected RealSense device is not reported as D435i.")
            pipeline_started = True

            print("[Camera] Pipeline started.")
            print("[Camera] Depth-to-color alignment enabled.")
            intrinsics = None

        print(f"[Camera] D435i depth scale: {depth_scale_mm:.6f} mm/unit")
        print()

        previous_time = time.perf_counter()
        last_print_time = 0.0

        first_real_inference = True
        frame_index = 0

        while True:
            frame_index += 1

            # ==================================================================
            # 1. RGB-D Frames
            # ==================================================================
            if db3_reader is not None:
                packet = db3_reader.next_frame()
                if packet is None:
                    print(f"[Playback] Completed {db3_reader.frame_count} frame pairs.")
                    break
                image_bgr = packet.color_bgr
                raw_depth = packet.depth_raw
                timestamp_ns = packet.timestamp_ns
                processed_depth_raw = (
                    offline_depth_postprocessor.process(raw_depth)
                    if offline_depth_postprocessor is not None
                    else raw_depth
                )
            else:
                frames = pipeline.wait_for_frames(5000)

                if not frames:
                    continue

                # ==============================================================
                # 2. Depth-to-color alignment
                # ==============================================================
                frames = align_filter.process(frames)

                if not frames:
                    continue

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()

                if not (bool(color_frame) and bool(depth_frame)):
                    continue

                # ==============================================================
                # 3. RGB -> OpenCV
                # ==============================================================
                image_bgr = np.asanyarray(color_frame.get_data()).copy()
                timestamp_ns = int(time.time() * 1_000_000_000)

                # ==============================================================
                # 4. Depth
                # ==============================================================
                raw_depth = np.asanyarray(depth_frame.get_data()).copy()
                if depth_postprocessor is not None:
                    processed_depth_raw = np.asanyarray(
                        depth_postprocessor.process(depth_frame).get_data()
                    ).copy()
                else:
                    processed_depth_raw = raw_depth

            raw_depth_map = raw_depth.astype(np.float32) * depth_scale_mm
            depth_map = processed_depth_raw.astype(np.float32) * depth_scale_mm
            raw_depth_map = np.where(
                (raw_depth_map > MIN_DEPTH_MM) & (raw_depth_map < MAX_DEPTH_MM),
                raw_depth_map,
                0.0,
            ).astype(np.float32)
            depth_map = np.where(
                (depth_map > MIN_DEPTH_MM) & (depth_map < MAX_DEPTH_MM),
                depth_map,
                0.0,
            ).astype(np.float32)

            # 对齐后必须与 RGB 尺寸一致，否则不能直接按相同像素索引取深度。
            if depth_map.shape != image_bgr.shape[:2]:
                print(
                    "[Warning] RGB/Depth shape mismatch after alignment: "
                    f"RGB={image_bgr.shape[:2]}, "
                    f"Depth={depth_map.shape}"
                )
                continue

            # ==================================================================
            # 5. Camera intrinsics
            # ==================================================================
            if intrinsics is None:
                intrinsics = get_realsense_color_intrinsics(color_frame)

                print("=" * 80)
                print("[Camera] RGB Intrinsics")
                print("=" * 80)
                print(
                    f"resolution = "
                    f"{intrinsics.width}x{intrinsics.height}"
                )
                print(f"fx = {intrinsics.fx:.4f}")
                print(f"fy = {intrinsics.fy:.4f}")
                print(f"cx = {intrinsics.cx:.4f}")
                print(f"cy = {intrinsics.cy:.4f}")
                print()

            environment_quality = compute_environment_quality(
                image_bgr,
                raw_depth_map,
                quality_config,
            )
            if exposure_settle_remaining > 0:
                settling_flags = tuple(
                    dict.fromkeys(
                        ("EXPOSURE_SETTLING",) + environment_quality.flags
                    )
                )
                environment_quality = EnvironmentQuality(
                    brightness_mean=environment_quality.brightness_mean,
                    dark_pixel_ratio=environment_quality.dark_pixel_ratio,
                    overexposed_ratio=environment_quality.overexposed_ratio,
                    contrast_std=environment_quality.contrast_std,
                    depth_valid_ratio=environment_quality.depth_valid_ratio,
                    flags=settling_flags,
                    status="|".join(settling_flags),
                    grasp_safe=False,
                )
                exposure_settle_remaining -= 1
            inference_image = enhance_rgb_for_environment(
                image_bgr,
                environment_quality,
                enabled=args.adaptive_rgb,
            )

            # ==================================================================
            # 6. 在第一次真实 YOLO 推理前先把原始相机画面显示出来
            #
            # 如果这里已经能看到画面，而随后卡住，问题就在 YOLO/CUDA，
            # 而不是 OpenCV GUI 或相机。
            # ==================================================================
            if display_enabled and first_real_inference:
                preview = image_bgr.copy()

                cv2.putText(
                    preview,
                    "Camera OK - starting first YOLO inference...",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    YELLOW,
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(WINDOW_NAME, preview)
                cv2.waitKey(1)

            # ==================================================================
            # 7. YOLO11 inference
            # ==================================================================
            if first_real_inference or args.debug:
                print(
                    "[DEBUG] YOLO input: "
                    f"shape={inference_image.shape}, "
                    f"dtype={inference_image.dtype}, "
                    f"device={device}"
                )

                if first_real_inference:
                    print("[DEBUG] First real YOLO inference starting...")

            inference_start = time.perf_counter()

            results = model.predict(
                source=inference_image,
                imgsz=args.imgsz,

                # Ultralytics 内部完成保持纵横比的 LetterBox。
                rect=False,

                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                device=device,
                verbose=False,

                # Seg 模型让 mask 直接映射回原图尺寸。
                retina_masks=True,
            )

            if device != "cpu" and torch.cuda.is_available():
                # 用于让 inference_ms 更接近真实 GPU 完成时间。
                torch.cuda.synchronize()

            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000.0

            if first_real_inference:
                print(
                    "[DEBUG] First real YOLO inference finished: "
                    f"{inference_ms:.1f} ms"
                )

            result = results[0]

            # ==================================================================
            # 8. YOLO + Depth -> XYZ
            # ==================================================================
            output_image, detections = process_yolo_result(
                image_bgr,
                depth_map,
                result,
                intrinsics,
                raw_depth_map=raw_depth_map,
            )

            for detection in detections:
                detection.frame_id = frame_index
                detection.timestamp_ns = timestamp_ns
            tracker.update(detections)
            # tracker.update 会更新稳定性字段，再计算最终抓取质量。
            for detection in detections:
                finalize_detection_for_grasp(
                    detection,
                    frame_id=frame_index,
                    timestamp_ns=timestamp_ns,
                    environment=environment_quality,
                    quality_config=quality_config,
                    intrinsics=intrinsics,
                    depth_map=depth_map,
                    extrinsics=extrinsics,
                    max_jaw_opening_mm=args.max_jaw_opening_mm,
                    pregrasp_distance_mm=args.pregrasp_distance_mm,
                    grasp_margin_mm=args.grasp_margin_mm,
                    min_detection_confidence=min_detection_confidence,
                    workspace_min_xyz_mm=workspace_min_xyz_mm,
                    workspace_max_xyz_mm=workspace_max_xyz_mm,
                )

            # 仅 GUI 使用光流预测框；终端和 JSONL 仍只记录本帧 YOLO 实测结果。
            visual_tracker.update(image_bgr, detections)
            annotate_detection_quality(
                output_image,
                combine_detected_and_tracked_for_display(
                    detections,
                    visual_tracker.display_detections(),
                ),
            )

            processed_frames += 1
            ready_results += sum(
                1 for detection in detections if detection.status == "READY"
            )
            rejected_results += sum(
                1 for detection in detections if detection.status != "READY"
            )

            if json_handle is not None:
                json_handle.write(
                    json.dumps(
                        frame_to_dict(
                            frame_index,
                            timestamp_ns,
                            environment_quality,
                            detections,
                        ),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                json_handle.flush()

            if first_real_inference:
                print(
                    "[DEBUG] First post-process finished. "
                    f"objects={len(detections)}"
                )

            # ==================================================================
            # 9. FPS
            # ==================================================================
            current_time = time.perf_counter()

            frame_delta = max(
                current_time - previous_time,
                1e-6,
            )

            fps = 1.0 / frame_delta
            previous_time = current_time

            # ==================================================================
            # 10. OSD（默认关闭，保持画面只显示红框和 XYZ）
            # ==================================================================
            if args.show_runtime_osd:
                cv2.putText(
                    output_image,
                    (
                        f"Infer: {inference_ms:.1f} ms"
                        f" | FPS: {fps:.1f}"
                        f" | Objects: {len(detections)}"
                        f" | Device: {device}"
                    ),
                    (10, 22),
                    FONT_FACE,
                    FONT_SCALE,
                    RED,
                    FONT_THICKNESS,
                    cv2.LINE_AA,
                )

            # 标注视频写出的画面与 GUI 完全一致，包括确认后的平滑跟踪框；
            # JSONL 仍只保存当前帧原始 YOLO + RGB-D 测量，便于实验复核。
            if args.out_video:
                if video_writer is None:
                    video_path = os.path.abspath(args.out_video)
                    video_parent = os.path.dirname(video_path)
                    if video_parent:
                        os.makedirs(video_parent, exist_ok=True)
                    height, width = output_image.shape[:2]
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(args.fps),
                        (width, height),
                    )
                    if not video_writer.isOpened():
                        video_writer.release()
                        video_writer = None
                        raise RuntimeError(f"Could not open annotated video for writing: {video_path}")
                    print(f"[Output] Annotated video: {video_path} ({width}x{height}@{args.fps}fps)")
                video_writer.write(output_image)

            # ==================================================================
            # 11. Terminal structured output
            # ==================================================================
            if args.print_result:
                wall_time = time.time()

                if (
                    wall_time - last_print_time
                    >= args.print_interval
                ):
                    print_detections(
                        detections,
                        environment=environment_quality,
                        frame_id=frame_index,
                        timestamp_ns=timestamp_ns,
                    )
                    last_print_time = wall_time

            # ==================================================================
            # 12. GUI
            # ==================================================================
            if display_enabled:
                if first_real_inference:
                    print("[DEBUG] Calling cv2.imshow() with processed frame...")

                cv2.imshow(
                    WINDOW_NAME,
                    output_image,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (
                    ESC_KEY,
                    ord("q"),
                    ord("Q"),
                ):
                    print("[Info] Exit key pressed.")
                    break

            if args.max_frames and processed_frames >= args.max_frames:
                print(f"[Info] Reached --max_frames={args.max_frames}.")
                break

            first_real_inference = False

    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user.")

    except Exception as error:
        print("\n[Error] Unhandled exception:")
        print(f"        {type(error).__name__}: {error}")
        print(
            "[Error] Check D435i USB connection, serial number, and "
            "the requested resolution/FPS with rs-enumerate-devices -c."
        )
        return 1

    finally:
        if display_enabled:
            cv2.destroyAllWindows()

        if pipeline_started:
            try:
                pipeline.stop()
                print("[Camera] Pipeline stopped.")
            except Exception as error:
                print(f"[Camera] pipeline.stop() failed: {error}")

        if db3_reader is not None:
            db3_reader.close()

        if json_handle is not None:
            json_handle.close()

        if video_writer is not None:
            video_writer.release()

        print(
            f"[Summary] processed_frames={processed_frames} "
            f"ready_results={ready_results} "
            f"rejected_results={rejected_results}"
        )

    return 0


# ==============================================================================
# Entry
# ==============================================================================

if __name__ == "__main__":
    sys.exit(main())
