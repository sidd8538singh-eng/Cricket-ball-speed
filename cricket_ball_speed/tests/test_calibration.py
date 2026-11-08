"""
Unit tests for camera calibration, homography, camera model, and validation.
Tests synthetic geometry with known ground truth, crease-to-wicket offset handling,
ground-plane homography airborne rejection, constraint sufficiency, and visual diagnostics.
"""

import unittest
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from calibration.world_coordinates import WorldPitchGeometry, CALIBRATION_POINT_KEYS
from calibration.point_selector import CalibrationPointManager
from calibration.camera_model import CameraModel
from calibration.homography import GroundPlaneHomography
from calibration.camera_calibration import CameraCalibrator
from calibration.calibration_validator import CalibrationValidator
from calibration.reprojection import compute_reprojection_report
from calibration.visual_diagnostics import (
    render_calibration_diagnostic_image,
    generate_calibration_numerical_report
)


class TestCalibration(unittest.TestCase):
    def setUp(self):
        self.geom = WorldPitchGeometry()
        self.w = 1920
        self.h = 1080

        # Construct a synthetic camera with ground truth pose
        # Camera 2m high, 2m behind near stumps (X=0, Y=-2.0, Z=2.0)
        # Pitch center is at X=0, Y=10.06, Z=0.0
        fx = 1800.0
        fy = 1800.0
        cx = self.w / 2.0
        cy = self.h / 2.0
        self.K_gt = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Camera optical center in world: C = [0.0, -2.0, 2.0]
        pitch_rad = np.radians(8.0)
        R_pitch = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0.0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])
        R_w2c = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]
        ]) @ R_pitch.T

        rvec_gt, _ = cv2.Rodrigues(R_w2c)
        C_gt = np.array([0.0, -2.0, 2.0], dtype=np.float64)
        tvec_gt = -R_w2c @ C_gt

        self.gt_camera = CameraModel(
            camera_matrix=self.K_gt,
            dist_coeffs=np.zeros(5),
            rvec=rvec_gt,
            tvec=tvec_gt,
            image_width=self.w,
            image_height=self.h
        )

        # Generate synthetic 2D projection points for the 8 canonical world points
        self.world_pts_dict = self.geom.get_world_coordinates()
        self.img_pts_dict = {}
        for k, pt3d in self.world_pts_dict.items():
            u, v = self.gt_camera.project_point(pt3d)
            self.img_pts_dict[k] = (u, v)

    def test_crease_vs_wicket_offsets(self):
        """
        Requirement: 'Carefully account for the fact that the distance between bowling creases
        is not automatically the distance between wicket-base centres.'
        """
        # Crease length is 20.12m. Near wickets 0.05m behind crease, far wickets 0.08m behind crease.
        geom_offset = WorldPitchGeometry(
            pitch_length_crease_to_crease_m=20.12,
            near_wicket_offset_from_crease_m=0.05,
            far_wicket_offset_from_crease_m=0.08,
            pitch_corner_reference="crease"
        )
        # Expected wicket distance L = 20.12 + 0.08 - 0.05 = 20.15m
        self.assertAlmostEqual(geom_offset.wicket_distance_m, 20.15, places=3)
        self.assertAlmostEqual(geom_offset.L, 20.15, places=3)

        coords = geom_offset.get_world_coordinates()
        # Near base is origin (0, 0, 0)
        np.testing.assert_allclose(coords["near_base"], [0.0, 0.0, 0.0])
        # Far base is at Y = L = 20.15m
        np.testing.assert_allclose(coords["far_base"], [0.0, 20.15, 0.0])
        # Near crease pitch corners are at Y = -near_offset = -0.05m
        self.assertAlmostEqual(coords["pitch_left_near"][1], -0.05, places=3)
        self.assertAlmostEqual(coords["pitch_right_near"][1], -0.05, places=3)
        # Far crease pitch corners are at Y = 20.12 - 0.05 = 20.07m
        self.assertAlmostEqual(coords["pitch_left_far"][1], 20.07, places=3)
        self.assertAlmostEqual(coords["pitch_right_far"][1], 20.07, places=3)

    def test_homography_airborne_rejection(self):
        """
        Requirement: 'Do not apply ground-plane homography to airborne ball positions.'
        """
        ground_3d = self.geom.get_ground_corners_3d()
        ground_2d = np.array([
            self.img_pts_dict["pitch_left_near"],
            self.img_pts_dict["pitch_right_near"],
            self.img_pts_dict["pitch_right_far"],
            self.img_pts_dict["pitch_left_far"]
        ])

        homography = GroundPlaneHomography(ground_3d, ground_2d)

        # Valid projection of ground point (Z = 0)
        u_p, v_p = homography.world_to_image(0.0, 10.06, z=0.0)
        self.assertGreater(u_p, 0.0)

        # Attempt to project airborne ball at Z = 1.8m -> MUST RAISE ERROR
        with self.assertRaises(ValueError) as ctx:
            homography.world_to_image(0.0, 10.06, z=1.8)
        self.assertIn("cannot project 3D point with height", str(ctx.exception))

    def test_constraint_sufficiency_checker(self):
        """
        Requirement: 'Check whether the available points uniquely and reliably constrain
        the desired parameters. If they do not, explicitly report what additional information is required.'
        """
        calibrator = CameraCalibrator(self.w, self.h, self.geom)

        # 1. Valid points pass sufficiency check
        is_ok, err = calibrator.check_constraint_sufficiency(self.img_pts_dict)
        self.assertTrue(is_ok)
        self.assertIsNone(err)

        # 2. Degenerate points: collapse near_top to near_base (height span = 0)
        degen_pts = dict(self.img_pts_dict)
        degen_pts["near_top"] = (degen_pts["near_base"][0], degen_pts["near_base"][1] - 2.0)
        is_ok2, err2 = calibrator.check_constraint_sufficiency(degen_pts)
        self.assertFalse(is_ok2)
        self.assertIn("Underconstrained vertical geometry", err2)

    def test_camera_calibration_solver_with_uncertainty(self):
        """
        Tests joint non-linear optimization with uncertainty and covariance reporting.
        """
        calibrator = CameraCalibrator(self.w, self.h, self.geom)
        solved_cam, reproj_report, uncertainty_rep = calibrator.calibrate(self.img_pts_dict, return_uncertainty=True)

        self.assertTrue(reproj_report.is_acceptable)
        self.assertLess(reproj_report.mean_error_px, 1.0)
        self.assertAlmostEqual(solved_cam.camera_height_m, 2.0, delta=0.2)
        self.assertAlmostEqual(solved_cam.camera_distance_behind_near_stumps_m, 2.0, delta=0.2)

        # Verify uncertainty metrics
        self.assertTrue(uncertainty_rep.is_well_constrained)
        self.assertLess(uncertainty_rep.jacobian_condition_number, 50000.0)
        self.assertGreater(uncertainty_rep.sigma_focal_length_px, 0.0)
        self.assertLess(uncertainty_rep.sigma_focal_length_px, 100.0)
        self.assertLess(uncertainty_rep.sigma_camera_height_m, 0.20)

    def test_visual_diagnostics_rendering(self):
        """Tests visual calibration diagnostics image generation."""
        calibrator = CameraCalibrator(self.w, self.h, self.geom)
        solved_cam, reproj_report, uncertainty_rep = calibrator.calibrate(self.img_pts_dict, return_uncertainty=True)

        frame = np.full((self.h, self.w, 3), 40, dtype=np.uint8)
        diag_img = render_calibration_diagnostic_image(
            frame, solved_cam, reproj_report, self.img_pts_dict, self.geom
        )

        self.assertEqual(diag_img.shape, (self.h, self.w, 3))
        # Image should not be empty
        self.assertGreater(np.mean(diag_img), 5.0)

        # Test numerical report
        num_report = generate_calibration_numerical_report(solved_cam, reproj_report, self.geom, uncertainty_rep)
        self.assertIn("camera_parameters", num_report)
        self.assertIn("reprojection_report", num_report)
        self.assertIn("uncertainty_report", num_report)


if __name__ == "__main__":
    unittest.main()
