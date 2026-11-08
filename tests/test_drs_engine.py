"""
Tests for Zoota-style DRS Engine and Calibration.
"""

import unittest
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.drs_engine import DRSCalibrationEngine, DRSLine, DetectedWicket
from cricket_ball_speed.calibration.world_coordinates import WorldPitchGeometry


class TestDRSEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DRSCalibrationEngine()
        self.w = 1280
        self.h = 720

    def test_model_loading(self):
        """Verifies stumps model is loaded successfully if ultralytics is available."""
        try:
            import ultralytics
            self.assertTrue(self.engine.is_stumps_loaded)
        except ImportError:
            self.skipTest("ultralytics not installed in this environment")

    def test_stump_drs_only_calibration(self):
        """
        Tests Stump + DRS Calibration using 4 known points generated from ground truth camera.
        """
        # Ground truth camera: f=1200, cx=640, cy=360, camera at (0, -2.0, 2.0)
        fx, fy = 1200.0, 1200.0
        cx, cy = 640.0, 360.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # Look down pitch towards (0, 10, 0)
        # Rotation with ~10 deg down tilt
        pitch = np.radians(8.5)
        R_pitch = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)]
        ])
        R_w2c = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ R_pitch.T
        rvec, _ = cv2.Rodrigues(R_w2c)
        C = np.array([0.0, -2.0, 2.0])
        tvec = -R_w2c @ C

        obj_pts = np.array([
            [0.0, 0.0, 0.0],      # near_base
            [0.0, 0.0, 0.711],    # near_top
            [0.0, 20.12, 0.0],    # far_base
            [0.0, 20.12, 0.711],  # far_top
        ], dtype=np.float64)

        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
        proj_2d = proj.reshape((-1, 2))

        img_points = {
            "near_base": (float(proj_2d[0, 0]), float(proj_2d[0, 1])),
            "near_top": (float(proj_2d[1, 0]), float(proj_2d[1, 1])),
            "far_base": (float(proj_2d[2, 0]), float(proj_2d[2, 1])),
            "far_top": (float(proj_2d[3, 0]), float(proj_2d[3, 1])),
        }

        result = self.engine.calibrate(img_points, self.w, self.h, initial_focal_length_px=1200.0)
        self.assertTrue(result["success"])
        self.assertEqual(result["method"], "Stump + DRS Calibration")
        self.assertLess(result["reprojection_error_px"], 2.0)
        self.assertIsNotNone(result["camera_parameters"])
        self.assertIn("homography_w2i", result["camera_parameters"])

        # Check estimated camera height is near 2.0m
        cam_height = result["camera_parameters"]["camera_height_m"]
        self.assertAlmostEqual(cam_height, 2.0, delta=0.4)

    def test_stump_drs_pitch_calibration(self):
        """
        Tests 8-point calibration with Pitch and DRS line.
        """
        fx, fy = 1200.0, 1200.0
        cx, cy = 640.0, 360.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        pitch = np.radians(8.5)
        R_pitch = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)]
        ])
        R_w2c = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ R_pitch.T
        rvec, _ = cv2.Rodrigues(R_w2c)
        C = np.array([0.0, -2.0, 2.0])
        tvec = -R_w2c @ C

        geom = WorldPitchGeometry()
        world_pts = geom.get_world_coordinates()

        img_points = {}
        for k, pt in world_pts.items():
            proj, _ = cv2.projectPoints(pt.reshape((1, 3)), rvec, tvec, K, None)
            p = proj.reshape((2,))
            img_points[k] = (float(p[0]), float(p[1]))

        result = self.engine.calibrate(img_points, self.w, self.h, initial_focal_length_px=1200.0)
        self.assertTrue(result["success"])
        self.assertEqual(result["method"], "Stump + DRS + Pitch Calibration")
        self.assertLess(result["reprojection_error_px"], 2.0)
        self.assertIn("homography_w2i", result["camera_parameters"])


if __name__ == "__main__":
    unittest.main()
