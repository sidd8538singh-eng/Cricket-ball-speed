"""
Unit tests specifically for the manual eight-point calibration module.
Tests coordinate transformations, clicking sequencing, duplicate rejection,
bounds validation, undo, reset, point replacement, Line A-B reference,
structured JSON saving/loading, video hash locking, and resolution mismatch detection.
"""

import unittest
import numpy as np
import tempfile
import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from calibration.world_coordinates import WorldPitchGeometry, CALIBRATION_POINT_KEYS
from calibration.point_selector import CalibrationPointManager
from calibration.coordinate_manager import CoordinateManager
from calibration.camera_model import CameraModel
from calibration.camera_calibration import CameraCalibrator
from calibration.manual_calibration_ui import (
    ManualCalibrationUI,
    save_calibration_file,
    load_calibration_file,
    CalibrationResolutionMismatchError,
    CalibrationLockError
)


class TestManualCalibration(unittest.TestCase):
    def setUp(self):
        self.w = 1920
        self.h = 1080
        self.geom = WorldPitchGeometry()
        self.coord_mgr = CoordinateManager(self.w, self.h)

        # Canonical synthetic ground truth points for testing
        # Perspective image: near stumps lower in image, far stumps higher
        self.valid_points = {
            "near_top": (960.0, 720.0),
            "near_base": (960.0, 850.0),
            "far_top": (960.0, 360.0),
            "far_base": (960.0, 410.0),
            "pitch_left_near": (650.0, 850.0),
            "pitch_right_near": (1270.0, 850.0),
            "pitch_left_far": (820.0, 410.0),
            "pitch_right_far": (1100.0, 410.0),
        }

    def test_coordinate_transformations_zoom_pan(self):
        """Tests that display coordinate conversions with zoom and pan map back to exact original pixels."""
        # Test across diverse zoom scales (0.5x, 1.0x, 2.5x, 5.0x) and pan offsets
        test_scales = [0.5, 1.0, 2.5, 5.0]
        test_coords = [(100.0, 150.0), (960.0, 540.0), (1800.0, 1000.0)]

        for scale in test_scales:
            pan_x = -200.0 * scale
            pan_y = -150.0 * scale
            for orig_x, orig_y in test_coords:
                disp_x, disp_y = self.coord_mgr.original_to_display(orig_x, orig_y, scale, pan_x, pan_y)
                recov_x, recov_y = self.coord_mgr.display_to_original(disp_x, disp_y, scale, pan_x, pan_y)
                self.assertAlmostEqual(orig_x, recov_x, delta=0.5)
                self.assertAlmostEqual(orig_y, recov_y, delta=0.5)

    def test_eight_point_click_sequence(self):
        """Verifies strict sequential ordering from 1 to 8."""
        pm = CalibrationPointManager(self.w, self.h)

        for idx, key in enumerate(CALIBRATION_POINT_KEYS):
            self.assertEqual(pm.next_required_point_key, key)
            pt = self.valid_points[key]
            added_key = pm.add_point(pt[0], pt[1])
            self.assertEqual(added_key, key)

        self.assertTrue(pm.is_complete())
        self.assertIsNone(pm.next_required_point_key)

    def test_out_of_bounds_rejection(self):
        """Verifies clicks outside image dimensions are strictly rejected."""
        pm = CalibrationPointManager(self.w, self.h)

        # Negative X
        with self.assertRaises(ValueError) as ctx:
            pm.add_point(-10.0, 500.0)
        self.assertIn("outside image width", str(ctx.exception))

        # X exceeding width
        with self.assertRaises(ValueError) as ctx:
            pm.add_point(2000.0, 500.0)
        self.assertIn("outside image width", str(ctx.exception))

        # Y exceeding height
        with self.assertRaises(ValueError) as ctx:
            pm.add_point(500.0, 1200.0)
        self.assertIn("outside image height", str(ctx.exception))

    def test_duplicate_point_rejection(self):
        """Verifies clicking on or very close to an existing point is rejected."""
        pm = CalibrationPointManager(self.w, self.h)
        pm.add_point(960.0, 720.0) # near_top

        # Attempt to click near_base right on top of near_top (within 3 pixels)
        with self.assertRaises(ValueError) as ctx:
            pm.add_point(961.0, 721.0)
        self.assertIn("too close", str(ctx.exception))

    def test_perspective_geometry_rejection(self):
        """Verifies obviously invalid perspective selections are rejected."""
        pm = CalibrationPointManager(self.w, self.h)
        pm.add_point(960.0, 720.0) # near_top at y=720

        # Attempt to set near_base above near_top (e.g. y=600 in image)
        with self.assertRaises(ValueError) as ctx:
            pm.add_point(960.0, 600.0)
        self.assertIn("near_base must be positioned below near_top", str(ctx.exception))

    def test_undo_and_reset(self):
        """Verifies undo and reset behaviors."""
        pm = CalibrationPointManager(self.w, self.h)
        pm.add_point(960.0, 720.0) # near_top
        pm.add_point(960.0, 850.0) # near_base

        self.assertEqual(pm.next_required_point_key, "far_top")

        # Undo near_base
        undone = pm.undo()
        self.assertEqual(undone, "near_base")
        self.assertEqual(pm.next_required_point_key, "near_base")
        self.assertIsNone(pm.points["near_base"])

        # Reset all
        pm.add_point(960.0, 850.0)
        pm.reset()
        self.assertIsNone(pm.points["near_top"])
        self.assertEqual(pm.next_required_point_key, "near_top")
        self.assertFalse(pm.is_complete())

    def test_individual_point_replacement(self):
        """Verifies replacing a specific point without resetting all others."""
        pm = CalibrationPointManager(self.w, self.h)
        for k in CALIBRATION_POINT_KEYS:
            pm.add_point(self.valid_points[k][0], self.valid_points[k][1])
        self.assertTrue(pm.is_complete())

        # Replace far_top (index 3)
        pm.set_point_explicit("far_top", 965.0, 362.0)
        self.assertEqual(pm.points["far_top"], (965.0, 362.0))
        # Ensure other points remain unchanged
        self.assertEqual(pm.points["near_top"], (960.0, 720.0))
        self.assertTrue(pm.is_complete())

    def test_line_a_b_connection(self):
        """
        Requirement: 'Draw the imaginary line connecting the centres of near_base and far_base.'
        Verifies line coordinates connecting near_base and far_base centres.
        """
        pm = CalibrationPointManager(self.w, self.h)
        pm.add_point(self.valid_points["near_top"][0], self.valid_points["near_top"][1])
        pm.add_point(self.valid_points["near_base"][0], self.valid_points["near_base"][1])
        pm.add_point(self.valid_points["far_top"][0], self.valid_points["far_top"][1])
        pm.add_point(self.valid_points["far_base"][0], self.valid_points["far_base"][1])

        pt_a = pm.points["near_base"] # Point A
        pt_b = pm.points["far_base"]  # Point B

        self.assertIsNotNone(pt_a)
        self.assertIsNotNone(pt_b)

        # Midpoint of imaginary line A-B
        mid_x = (pt_a[0] + pt_b[0]) / 2.0
        mid_y = (pt_a[1] + pt_b[1]) / 2.0
        self.assertAlmostEqual(mid_x, 960.0)
        self.assertAlmostEqual(mid_y, (850.0 + 410.0) / 2.0)

    def test_save_and_load_calibration_file(self):
        """Tests saving and loading structured JSON calibration file."""
        calibrator = CameraCalibrator(self.w, self.h, self.geom)
        cam_model, reproj_report = calibrator.calibrate(self.valid_points)

        with tempfile.TemporaryDirectory() as tmp_dir:
            calib_file = os.path.join(tmp_dir, "test_calib.json")
            video_hash = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
            video_path = "sample_video.mp4"

            save_calibration_file(
                calibration_path=calib_file,
                video_hash=video_hash,
                video_path=video_path,
                frame_idx=15,
                frame_timestamp=0.25,
                points_dict=self.valid_points,
                camera_model=cam_model,
                reproj_dict=reproj_report.to_dict(),
                geometry=self.geom
            )

            self.assertTrue(os.path.isfile(calib_file))

            # Load and validate
            loaded = load_calibration_file(
                calib_file,
                expected_video_hash=video_hash,
                expected_resolution=(self.w, self.h)
            )

            self.assertEqual(loaded["calibration_version"], "1.0.0")
            self.assertEqual(loaded["video_identity"], video_hash)
            self.assertEqual(loaded["selected_frame_index"], 15)
            self.assertEqual(loaded["video_resolution"], [self.w, self.h])

            # Exact coordinates preserved without modification
            pts_loaded = loaded["calibration_points_pixel"]
            for k in CALIBRATION_POINT_KEYS:
                self.assertAlmostEqual(pts_loaded[k][0], self.valid_points[k][0])
                self.assertAlmostEqual(pts_loaded[k][1], self.valid_points[k][1])

    def test_resolution_mismatch_detection(self):
        """Verifies resolution mismatch error is raised when video resolution differs from calibration."""
        calibrator = CameraCalibrator(self.w, self.h, self.geom)
        cam_model, reproj_report = calibrator.calibrate(self.valid_points)

        with tempfile.TemporaryDirectory() as tmp_dir:
            calib_file = os.path.join(tmp_dir, "calib_1080p.json")
            save_calibration_file(
                calibration_path=calib_file,
                video_hash="hash_12345",
                video_path="video.mp4",
                frame_idx=0,
                frame_timestamp=0.0,
                points_dict=self.valid_points,
                camera_model=cam_model,
                reproj_dict=reproj_report.to_dict(),
                geometry=self.geom
            )

            # Attempt to load with 720p resolution (1280x720)
            with self.assertRaises(CalibrationResolutionMismatchError) as ctx:
                load_calibration_file(
                    calib_file,
                    expected_resolution=(1280, 720)
                )
            self.assertIn("Video resolution mismatch", str(ctx.exception))

    def test_video_lock_hash_mismatch(self):
        """Verifies CalibrationLockError is raised when video hash does not match."""
        calibrator = CameraCalibrator(self.w, self.h, self.geom)
        cam_model, reproj_report = calibrator.calibrate(self.valid_points)

        with tempfile.TemporaryDirectory() as tmp_dir:
            calib_file = os.path.join(tmp_dir, "calib_locked.json")
            save_calibration_file(
                calibration_path=calib_file,
                video_hash="hash_AAA",
                video_path="video.mp4",
                frame_idx=0,
                frame_timestamp=0.0,
                points_dict=self.valid_points,
                camera_model=cam_model,
                reproj_dict=reproj_report.to_dict(),
                geometry=self.geom
            )

            with self.assertRaises(CalibrationLockError) as ctx:
                load_calibration_file(
                    calib_file,
                    expected_video_hash="hash_BBB"
                )
            self.assertIn("Calibration hash mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
