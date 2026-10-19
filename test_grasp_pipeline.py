"""Fast regression tests for the environment/quality/grasp pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import d435i_yolo11_3d_detection as pipeline


class GraspPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.intrinsics = pipeline.CameraIntrinsics(
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
        )

    def test_environment_flags_are_explainable(self) -> None:
        dark_image = np.full((20, 20, 3), 20, dtype=np.uint8)
        invalid_depth = np.zeros((20, 20), dtype=np.float32)
        quality = pipeline.compute_environment_quality(
            dark_image,
            invalid_depth,
            pipeline.QualityConfig(),
        )
        self.assertIn("LOW_LIGHT", quality.flags)
        self.assertIn("DEPTH_UNRELIABLE", quality.flags)
        self.assertFalse(quality.grasp_safe)

    def test_depth_estimate_contains_quality_and_point_cloud(self) -> None:
        depth = np.zeros((480, 640), dtype=np.float32)
        raw_depth = np.zeros_like(depth)
        region = np.zeros_like(depth, dtype=bool)
        region[200:260, 290:350] = True
        depth[region] = 1000.0
        raw_depth[region] = 1000.0
        estimate = pipeline.estimate_depth_from_region(
            depth,
            region,
            intrinsics=self.intrinsics,
            raw_depth_map=raw_depth,
        )
        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertAlmostEqual(estimate.valid_ratio, 1.0)
        self.assertAlmostEqual(estimate.depth_mad_mm, 0.0)
        self.assertEqual(estimate.points_camera_mm.shape[1], 3)

    def test_pca_grasp_and_transform(self) -> None:
        x_values = np.linspace(-80.0, 80.0, 20)
        y_values = np.linspace(-120.0, 120.0, 20)
        xx, yy = np.meshgrid(x_values, y_values)
        points = np.column_stack(
            (
                xx.ravel(),
                yy.ravel(),
                np.full(xx.size, 1000.0),
            )
        )
        depth = np.full((480, 640), 1000.0, dtype=np.float32)
        geometry, candidates = pipeline.create_grasp_candidates(
            points,
            depth,
            self.intrinsics,
            pregrasp_distance_mm=100.0,
            grasp_margin_mm=10.0,
        )
        self.assertIsNotNone(geometry)
        self.assertTrue(candidates)
        self.assertIn(candidates[0].grasp_type, {"SIDE_GRASP", "RIM_GRASP"})

        rotation = np.eye(3)
        extrinsics = pipeline.Extrinsics(
            rotation_base_from_camera=rotation,
            translation_base_mm=np.asarray((10.0, 20.0, 30.0)),
        )
        self.assertEqual(
            pipeline.transform_point_to_base(extrinsics, (1.0, 2.0, 3.0)),
            (11.0, 22.0, 33.0),
        )
        self.assertEqual(
            pipeline.rotation_matrix_to_quaternion_xyzw(rotation),
            (0.0, 0.0, 0.0, 1.0),
        )

    def test_tracker_requires_consecutive_stable_frames(self) -> None:
        tracker = pipeline.TemporalStabilityTracker(5, 5, 30.0)
        detections = []
        for frame_index in range(5):
            detection = pipeline.Detection3D(
                class_id=1,
                class_name="garbage bin",
                confidence=0.9,
                bbox=(100, 100, 200, 300),
                pixel_u=150,
                pixel_v=200,
                depth_mm=1000.0,
                x_mm=10.0 + frame_index * 0.5,
                y_mm=20.0,
                z_mm=1000.0,
                depth_source="center_roi",
            )
            detections = [detection]
            tracker.update(detections)
        self.assertEqual(detections[0].stable_frame_count, 5)
        self.assertTrue(detections[0].stable)
        self.assertEqual(detections[0].target_id, 1)
        self.assertIsNotNone(detections[0].filtered_camera_xyz_mm)
        # 平滑结果仅用于展示和日志，原始测量值不应被覆盖。
        self.assertEqual(detections[0].x_mm, 12.0)

    def test_timestamp_kalman_uses_measurement_time(self) -> None:
        tracker = pipeline.TimestampPositionKalman(1.0 / 30.0)
        first = tracker.update(np.asarray((0.0, 0.0, 1000.0)), 0, 20.0)
        second = tracker.update(
            np.asarray((100.0, 0.0, 1000.0)),
            1_000_000_000,
            20.0,
        )
        self.assertEqual(tuple(first), (0.0, 0.0, 1000.0))
        self.assertGreater(second[0], 50.0)
        self.assertLess(second[0], 100.0)

    def test_finalizer_produces_ready_with_calibrated_pose(self) -> None:
        x_values = np.linspace(-30.0, 30.0, 10)
        y_values = np.linspace(-60.0, 60.0, 10)
        xx, yy = np.meshgrid(x_values, y_values)
        points = np.column_stack(
            (
                xx.ravel(),
                yy.ravel(),
                np.full(xx.size, 1000.0),
            )
        ).astype(np.float32)
        detection = pipeline.Detection3D(
            class_id=1,
            class_name="garbage bin",
            confidence=0.9,
            bbox=(280, 180, 360, 300),
            pixel_u=320,
            pixel_v=240,
            depth_mm=1000.0,
            x_mm=0.0,
            y_mm=0.0,
            z_mm=1000.0,
            depth_source="center_roi",
            valid_depth_points=100,
            cluster_depth_points=100,
            region_depth_points=100,
            depth_valid_ratio=1.0,
            depth_mad_mm=0.0,
            depth_std_mm=0.0,
            raw_valid_depth_points=100,
            point_cloud_camera_mm=points,
            stable_frame_count=5,
            stable=True,
        )
        quality = pipeline.compute_environment_quality(
            np.full((480, 640, 3), 128, dtype=np.uint8),
            np.full((480, 640), 1000, dtype=np.float32),
            pipeline.QualityConfig(),
        )
        extrinsics = pipeline.Extrinsics(
            rotation_base_from_camera=np.eye(3),
            translation_base_mm=np.zeros(3),
        )
        pipeline.finalize_detection_for_grasp(
            detection,
            frame_id=1,
            timestamp_ns=2,
            environment=quality,
            quality_config=pipeline.QualityConfig(),
            intrinsics=self.intrinsics,
            depth_map=np.full((480, 640), 1000, dtype=np.float32),
            extrinsics=extrinsics,
            max_jaw_opening_mm=1000.0,
            pregrasp_distance_mm=100.0,
            grasp_margin_mm=10.0,
            min_detection_confidence=0.2,
            workspace_min_xyz_mm=(-2000.0, -2000.0, 0.0),
            workspace_max_xyz_mm=(2000.0, 2000.0, 3000.0),
        )
        self.assertEqual(detection.status, "READY")
        self.assertTrue(detection.graspable)
        self.assertIsNotNone(detection.position_base_xyz_mm)
        self.assertIsNotNone(detection.orientation_quaternion_xyzw)

    def test_jsonl_result_is_serializable(self) -> None:
        quality = pipeline.compute_environment_quality(
            np.full((10, 10, 3), 128, dtype=np.uint8),
            np.full((10, 10), 1000, dtype=np.float32),
            pipeline.QualityConfig(),
        )
        result = pipeline.frame_to_dict(1, 2, quality, [])
        encoded = json.dumps(result, allow_nan=False)
        self.assertIn('"frame_id": 1', encoded)

    def test_gui_uses_colored_bbox_red_center_bar_and_xyz_cm(self) -> None:
        detection = pipeline.Detection3D(
            class_id=1,
            class_name="garbage bin",
            confidence=0.9,
            bbox=(20, 50, 80, 100),
            pixel_u=50,
            pixel_v=75,
            depth_mm=1000.0,
            x_mm=1.2,
            y_mm=-3.4,
            z_mm=1000.0,
            depth_source="center_roi",
        )
        self.assertEqual(
            pipeline.format_detection_label(detection),
            ["XYZ=(0.1,-0.3,100.0) cm"],
        )
        self.assertEqual(
            pipeline.format_detection_label_parts(detection),
            [
                ("XYZ=(0.1,-0.3,", pipeline.GREEN),
                ("100.0", pipeline.RED),
                (") cm", pipeline.GREEN),
            ],
        )
        image = np.zeros((140, 120, 3), dtype=np.uint8)
        pipeline.annotate_detection_quality(image, [detection])
        self.assertEqual(tuple(image[99, 20]), pipeline.PALETTE[1])
        self.assertNotEqual(tuple(image[100, 50]), pipeline.RED)
        self.assertEqual(tuple(image[75, 50]), pipeline.RED)
        self.assertEqual(tuple(image[75, 38]), pipeline.RED)

        detection.depth_mm = None
        detection.x_mm = None
        detection.y_mm = None
        detection.z_mm = None
        self.assertEqual(pipeline.format_detection_label(detection), ["XYZ=N/A"])

        detection.pixel_u = None
        detection.pixel_v = None
        no_depth_image = np.zeros((140, 120, 3), dtype=np.uint8)
        pipeline.annotate_detection_quality(no_depth_image, [detection])
        self.assertEqual(tuple(no_depth_image[75, 50]), pipeline.RED)

    def test_visual_tracker_moves_box_with_target_features(self) -> None:
        first = np.zeros((160, 200, 3), dtype=np.uint8)
        rng = np.random.default_rng(42)
        first[45:115, 55:125] = rng.integers(
            0,
            256,
            size=(70, 70, 3),
            dtype=np.uint8,
        )
        detection = pipeline.Detection3D(
            class_id=1,
            class_name="garbage bin",
            confidence=0.9,
            bbox=(50, 40, 130, 120),
            pixel_u=90,
            pixel_v=80,
            depth_mm=1000.0,
            x_mm=0.0,
            y_mm=0.0,
            z_mm=1000.0,
            depth_source="center_roi",
            target_id=1,
            depth_valid_ratio=1.0,
            depth_mad_mm=0.0,
        )
        tracker = pipeline.VisualObjectTracker(max_missed_frames=3)
        tracker.update(first, [detection])
        self.assertEqual(tracker.display_detections(), [])
        direct_display = pipeline.combine_detected_and_tracked_for_display(
            [detection],
            tracker.display_detections(),
        )
        self.assertEqual(direct_display, [detection])
        tracker.update(first, [pipeline.replace(detection, timestamp_ns=33_000_000)])
        self.assertEqual(tracker.display_detections(), [])
        tracker.update(first, [pipeline.replace(detection, timestamp_ns=66_000_000)])
        self.assertEqual(len(tracker.display_detections()), 1)
        translation = np.float32(((1.0, 0.0, 8.0), (0.0, 1.0, 4.0)))
        second = cv2.warpAffine(first, translation, (200, 160))
        tracker.update(second, [])

        displayed = tracker.display_detections()
        self.assertEqual(len(displayed), 1)
        self.assertTrue(displayed[0].display_predicted)
        self.assertAlmostEqual(displayed[0].bbox[0] - detection.bbox[0], 8, delta=2)
        self.assertAlmostEqual(displayed[0].bbox[1] - detection.bbox[1], 4, delta=2)

    def test_confirmed_display_prefers_smoothed_track_box(self) -> None:
        direct = pipeline.Detection3D(
            class_id=1,
            class_name="garbage bin",
            confidence=0.9,
            bbox=(100, 100, 200, 300),
            pixel_u=150,
            pixel_v=200,
            depth_mm=1000.0,
            x_mm=0.0,
            y_mm=0.0,
            z_mm=1000.0,
            depth_source="center_roi",
            target_id=1,
        )
        smoothed = pipeline.replace(direct, bbox=(105, 103, 205, 303))
        displayed = pipeline.combine_detected_and_tracked_for_display(
            [direct], [smoothed]
        )
        self.assertEqual(displayed, [smoothed])

    def test_visual_tracker_rejects_low_confidence_or_invalid_candidate_shape(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        tracker = pipeline.VisualObjectTracker(max_missed_frames=3)
        low_confidence = pipeline.Detection3D(
            class_id=1,
            class_name="garbage bin",
            confidence=0.44,
            bbox=(20, 20, 80, 80),
            pixel_u=50,
            pixel_v=50,
            depth_mm=1000.0,
            x_mm=0.0,
            y_mm=0.0,
            z_mm=1000.0,
            depth_source="center_roi",
            target_id=1,
            depth_valid_ratio=1.0,
            depth_mad_mm=0.0,
        )
        for timestamp_ns in (0, 33_000_000, 66_000_000):
            tracker.update(
                image,
                [pipeline.replace(low_confidence, timestamp_ns=timestamp_ns)],
            )
        self.assertEqual(tracker.display_detections(), [])

        too_thin = pipeline.replace(
            low_confidence,
            confidence=0.90,
            bbox=(20, 10, 38, 90),
            target_id=2,
        )
        for timestamp_ns in (100_000_000, 133_000_000, 166_000_000):
            tracker.update(
                image,
                [pipeline.replace(too_thin, timestamp_ns=timestamp_ns)],
            )
        self.assertEqual(tracker.display_detections(), [])

    def test_example_extrinsics_cannot_be_used_as_calibration(self) -> None:
        example = {
            "calibration_required": True,
            "translation_base_mm": [0.0, 0.0, 0.0],
            "rotation_base_from_camera": np.eye(3).tolist(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.json"
            path.write_text(json.dumps(example), encoding="utf-8")
            with self.assertRaises(ValueError):
                pipeline.load_extrinsics(str(path))


if __name__ == "__main__":
    unittest.main()
