"""
Unit and Integration Tests for Automatic Stump Calibration, DRS Line,
Pitch Geometry, and Calibration Manager.
"""

import unittest
import numpy as np
import cv2
import os
import sys

# Ensure d:/final and d:/final/cricket_ball_speed are in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CBS_DIR = os.path.join(ROOT_DIR, "cricket_ball_speed")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if CBS_DIR not in sys.path:
    sys.path.insert(0, CBS_DIR)

from calibration.drs_line import DRSLine, DRSLineValidator
from calibration.automatic_stump_calibration import (
    AutomaticStumpDetector, AutomaticStumpDetectionResult, WicketObservation
)
from calibration.pitch_geometry import PitchGeometryEngine, PitchCorners
from calibration.calibration_manager import CalibrationManager, CalibrationResult
from calibration.calibration_storage import CalibrationStorage, StoredCalibration
from calibration.world_coordinates import WorldPitchGeometry


class TestDRSLineAndCalibration(unittest.TestCase):

    def test_drs_line_geometry_and_projections(self):
        """Verifies DRSLine length, midpoint, lateral distance, and longitudinal projection."""
        p_near = (640.0, 680.0)
        p_far = (640.0, 360.0)

        drs = DRSLine(
            near_point=p_near,
            far_point=p_far,
            real_distance_m=20.12,
            is_valid=True
        )

        self.assertAlmostEqual(drs.pixel_length, 320.0, places=1)
        self.assertEqual(drs.midpoint, (640.0, 520.0))
        self.assertAlmostEqual(drs.angle_deg, -90.0, places=1) # Pointing straight up in image coordinates

        # Point on the line: lateral distance should be 0
        self.assertAlmostEqual(drs.lateral_distance_px((640.0, 500.0)), 0.0, places=2)

        # Point 50px to the right
        self.assertAlmostEqual(abs(drs.lateral_distance_px((690.0, 500.0))), 50.0, places=1)

        # Longitudinal projection fraction
        self.assertAlmostEqual(drs.longitudinal_projection_px(p_near), 0.0, places=2)
        self.assertAlmostEqual(drs.longitudinal_projection_px(p_far), 1.0, places=2)
        self.assertAlmostEqual(drs.longitudinal_projection_px(drs.midpoint), 0.5, places=2)

    def test_drs_line_validation_rules(self):
        """Tests perspective and geometric rejection in DRSLineValidator."""
        # 1. Normal valid setup
        drs_valid = DRSLineValidator.validate(
            near_base=(640.0, 650.0),
            far_base=(640.0, 350.0),
            image_width=1280,
            image_height=720
        )
        self.assertTrue(drs_valid.is_valid)
        self.assertFalse(drs_valid.is_provisional)
        self.assertGreater(drs_valid.confidence, 0.7)

        # 2. Inverted perspective (far wicket lower in frame than near wicket)
        drs_inverted = DRSLineValidator.validate(
            near_base=(640.0, 350.0),
            far_base=(640.0, 650.0),
            image_width=1280,
            image_height=720
        )
        self.assertFalse(drs_inverted.is_valid)
        self.assertTrue(drs_inverted.is_provisional)
        self.assertTrue(any("Inverted perspective" in note for note in drs_inverted.validation_notes))

        # 3. Too short pixel length (< 50px)
        drs_short = DRSLineValidator.validate(
            near_base=(640.0, 400.0),
            far_base=(640.0, 380.0),
            image_width=1280,
            image_height=720,
            min_pixel_length=50.0
        )
        self.assertFalse(drs_short.is_valid)
        self.assertTrue(any("below minimum threshold" in note for note in drs_short.validation_notes))

    def test_wicket_classification_and_disambiguation(self):
        """Tests separation of near from far wickets based on perspective and apparent scale."""
        detector = AutomaticStumpDetector(model_path="models/non_existent.pt")

        # Synthetic candidates: Candidate 1 (near: lower y, taller), Candidate 2 (far: higher y, smaller)
        cand1 = WicketObservation(
            stump_type="unassigned",
            bbox=[600.0, 520.0, 680.0, 680.0],
            confidence=0.92,
            base_center=(640.0, 680.0),
            top_center=(640.0, 520.0),
            height_px=160.0,
            width_px=80.0
        )
        cand2 = WicketObservation(
            stump_type="unassigned",
            bbox=[625.0, 380.0, 655.0, 430.0],
            confidence=0.88,
            base_center=(640.0, 430.0),
            top_center=(640.0, 380.0),
            height_px=50.0,
            width_px=30.0
        )

        res = detector._classify_wickets([cand1, cand2], 1280, 720, frame_index=0, pitch_bbox=None)

        self.assertEqual(res.status, "DRS_LINE_GENERATED")
        self.assertIsNotNone(res.near_wicket)
        self.assertIsNotNone(res.far_wicket)
        self.assertEqual(res.near_wicket.stump_type, "near")
        self.assertEqual(res.far_wicket.stump_type, "far")
        self.assertIsNotNone(res.drs_line)
        self.assertTrue(res.drs_line.is_valid)
        self.assertEqual(res.drs_line.near_point, (640.0, 680.0))
        self.assertEqual(res.drs_line.far_point, (640.0, 430.0))

    def test_pitch_geometry_estimation_and_alignment(self):
        """Tests pitch corner estimation from bounding box and DRS line cross-check."""
        engine = PitchGeometryEngine(model_path=None)

        drs = DRSLine(
            near_point=(640.0, 680.0),
            far_point=(640.0, 380.0),
            real_distance_m=20.12,
            is_valid=True
        )

        # Synthetic pitch bbox: [x1, y1, x2, y2]
        pitch_bbox = [450.0, 360.0, 830.0, 700.0]
        corners = engine._estimate_corners(pitch_bbox, (720, 1280, 3), drs_line=drs)

        self.assertTrue(corners.is_valid)
        # Verify lateral order: left corner x < right corner x
        self.assertLess(corners.pitch_left_near[0], corners.pitch_right_near[0])
        self.assertLess(corners.pitch_left_far[0], corners.pitch_right_far[0])
        # Verify vertical order: near corners y > far corners y
        self.assertGreater(corners.pitch_left_near[1], corners.pitch_left_far[1])

    def test_calibration_manager_4point_solver(self):
        """Tests 4-point Stump + DRS calibration solving focal length and camera pose."""
        mgr = CalibrationManager(
            stumps_model_path="models/non_existent.pt",
            pitch_model_path=None
        )

        pts = {
            "near_base": (640.0, 680.0),
            "near_top": (640.0, 520.0),
            "far_base": (640.0, 420.0),
            "far_top": (640.0, 400.0)
        }

        calib = mgr.solve_calibration(
            image_points=pts,
            image_width=1280,
            image_height=720,
            initial_focal_length_px=1000.0
        )

        self.assertTrue(calib.success)
        self.assertEqual(calib.status, "AUTOMATIC_CALIBRATED")
        self.assertIsNotNone(calib.camera_model)
        self.assertIsNotNone(calib.reprojection_report)
        self.assertLess(calib.reprojection_report.mean_error_px, 1.0)

        # Physical camera sanity
        c_world = calib.camera_model.optical_centre
        # Camera is behind near stumps (Y < 0.5m)
        self.assertLess(c_world[1], 0.5)
        # Camera is elevated above ground (Z > 0.3m)
        self.assertGreater(c_world[2], 0.3)

    def test_calibration_storage_video_isolation(self):
        """Verifies calibration is locked to SHA-256 hash and not cross-contaminated."""
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        storage = CalibrationStorage(tmp_dir)

        calib_rec = StoredCalibration(
            entity_id="abc123hash",
            entity_type="video",
            calibration_version="2.0.0",
            created_at="2026-11-06T18:00:00",
            image_width=1920,
            image_height=1080,
            calibration_method="Stump + DRS Calibration",
            points_dict={"near_base": [960, 950]},
            drs_line={"near_point": [960, 950], "far_point": [960, 450]},
            pitch_corners=None,
            camera_parameters={"width": 1920, "height": 1080},
            reprojection_report={"mean_error_px": 0.42},
            physical_dimensions={"wicket_distance_m": 20.12},
            is_valid=True
        )

        saved_path = storage.save_calibration(calib_rec)
        self.assertTrue(os.path.isfile(saved_path))

        # Retrieve with matching hash
        loaded = storage.load_calibration_for_video("abc123hash")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.image_width, 1920)
        self.assertEqual(loaded.calibration_method, "Stump + DRS Calibration")

        # Non-matching hash should return None (isolation guarantee)
        mismatch = storage.load_calibration_for_video("different_hash_xyz")
        self.assertIsNone(mismatch)


if __name__ == "__main__":
    unittest.main()
