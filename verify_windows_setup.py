#!/usr/bin/env python3
"""Verify the Windows runtime and bundled model without requiring a camera."""

from pathlib import Path

import cv2
import numpy
import pyrealsense2 as rs
import torch
import ultralytics
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "bin_v3_yolo11s_960_best.pt"

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"Ultralytics: {ultralytics.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {numpy.__version__}")

model = YOLO(str(MODEL))
print(f"Model: {MODEL}")
print(f"Model task: {model.task}")
print(f"Model classes: {model.names}")

devices = rs.context().query_devices()
print(f"Connected RealSense devices: {len(devices)}")
for index, device in enumerate(devices):
    print(f"  {index}: {device.get_info(rs.camera_info.name)}")

if not torch.cuda.is_available():
    print("WARNING: CUDA is unavailable; detection will run on CPU.")
if len(devices) == 0:
    print("WARNING: no RealSense camera is currently connected.")

print("Verification completed.")
