"""
Comprehensive Unit Tests for 3D Ball Reconstruction, Camera Rays, Depth Estimation,
and Trajectory Fitting.
Tests known 3D ballistic ground-truth trajectories, heteroskedastic depth cues,
tiny-object depth handling (2-5 px), outlier rejection, and unobservable depth rejection (NOT_MEASURED).
"""

import unittest
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from calibration.camera_model import CameraModel
from reconstruction.camera_rays import CameraRayBuilder, Ray3D, triangulate_two_rays
from reconstruction.depth_estimation import RayDepthEstimator
from reconstruction.trajectory_fitting import TrajectoryFitter, TrajectoryFitResult
from ball_processing.ball_observations import BallObservation


class TestReconstruction(unittest.TestCase):
    def setUp(self):
        w, h = 1920, 1080
        fx = fy = 1800.0
        cx = w / 2.0
        cy = h / 2.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # Fixed camera placed behind bowler: C = (0.0, -2.0, 2.0)
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
        rvec, _ = cv2.Rodrigues(R_w2c)
        C = np.array([0.0, -2.0, 2.0])
        tvec = -R_w2c @ C

        self.camera = CameraModel(K, np.zeros(5), rvec, tvec, w, h)

    # -------------------------------------------------------------------------
    # 1. Camera Rays & Triangulation Tests
    # -------------------------------------------------------------------------

    def test_camera_ray_builder_and_point_projection(self):
        """Verifies camera ray back-projection matches projection."""
        ray_builder = CameraRayBuilder(self.camera)
        u_target, v_target = 960.0, 540.0 # Optical center
        ray = ray_builder.build_ray(u_target, v_target)

        # Camera center must match ray origin
        np.testing.assert_allclose(ray.origin, self.camera.optical_centre, atol=1e-4)

        # Point along ray should project back to exact (u, v)
        pt_along = ray.point_at_distance(10.0)
        u_proj, v_proj = self.camera.project_point(pt_along)
        self.assertAlmostEqual(u_proj, u_target, delta=0.1)
        self.assertAlmostEqual(v_proj, v_target, delta=0.1)

    def test_triangulate_two_rays(self):
        """Tests ray intersection / triangulation between two simulated cameras."""
        target = np.array([1.0, 5.0, 1.0])
        c1 = np.array([0.0, 0.0, 0.0])
        c2 = np.array([2.0, 0.0, 0.0])

        ray1 = Ray3D(c1, target - c1)
        ray2 = Ray3D(c2, target - c2)

        midpoint, dist = triangulate_two_rays(ray1, ray2)
        self.assertLess(dist, 1e-4)
        np.testing.assert_allclose(midpoint, target, atol=1e-3)

    # -------------------------------------------------------------------------
    # 2. Heteroskedastic Apparent Diameter Depth Cue Tests
    # -------------------------------------------------------------------------

    def test_heteroskedastic_apparent_diameter_depth_and_variance(self):
        """
        Verifies that apparent diameter depth variance is appropriately scaled
        by inverse diameter squared (Taylor series propagation).
        """
        depth_est = RayDepthEstimator(self.camera, ball_diameter_m=0.0715, pixel_diameter_noise_std=0.75)

        # 1. Near bowler: d_px = 30 px (distance ~ 4.3m)
        depth_near, var_near = depth_est.estimate_depth_and_variance(30.0)
        sigma_near = np.sqrt(var_near)
        self.assertAlmostEqual(depth_near, (1800.0 * 0.0715) / 30.0, places=2)
        self.assertLess(sigma_near, 0.30) # High precision near camera

        # 2. Distant ball: d_px = 4 px (distance ~ 32m)
        depth_far, var_far = depth_est.estimate_depth_and_variance(4.0)
        sigma_far = np.sqrt(var_far)
        self.assertAlmostEqual(depth_far, (1800.0 * 0.0715) / 4.0, places=1)
        self.assertGreater(sigma_far, 5.0) # High uncertainty at distance

        # Uncertainty at 4px must be over 20x larger than at 30px
        self.assertGreater(sigma_far / sigma_near, 20.0)

    # -------------------------------------------------------------------------
    # 3. Known Synthetic Ballistic Trajectory Speed Recovery
    # -------------------------------------------------------------------------

    def test_trajectory_fitter_recovers_known_ground_truth_speed(self):
        """
        RIGOROUS SCIENTIFIC TEST:
        Simulate a 135.0 km/h delivery with known ballistic equations.
        Verify that TrajectoryFitter recovers velocity, release position, and speed within 1.0 km/h.
        """
        # Ground truth state at release (t0 = 0.0):
        # 135 km/h = 37.5 m/s
        gt_x0 = 0.05
        gt_y0 = 0.40
        gt_z0 = 2.15
        gt_vx0 = 0.15
        gt_vy0 = 37.50 # m/s
        gt_vz0 = -2.20 # m/s
        g = 9.80665

        gt_speed_m_s = np.sqrt(gt_vx0**2 + gt_vy0**2 + gt_vz0**2)
        gt_speed_km_h = gt_speed_m_s * 3.6

        # 15 frames at 120 FPS (dt = 0.00833s)
        dt_frame = 1.0 / 120.0
        observations = []

        for i in range(15):
            t = i * dt_frame
            x_t = gt_x0 + gt_vx0 * t
            y_t = gt_y0 + gt_vy0 * t
            z_t = gt_z0 + gt_vz0 * t - 0.5 * g * (t ** 2)

            u, v = self.camera.project_point(np.array([x_t, y_t, z_t]))

            # Non-constant apparent diameter: decreases as ball flies away
            dist_cam = float(np.linalg.norm(np.array([x_t, y_t, z_t]) - self.camera.optical_centre))
            diam_px = (1800.0 * 0.0715) / dist_cam

            half_d = diam_px / 2.0
            bbox = [u - half_d, v - half_d, u + half_d, v + half_d]
            obs = BallObservation.from_bbox(
                frame_idx=i,
                timestamp_sec=t,
                bbox=bbox,
                confidence=0.92,
                detection_source="synthetic_gt"
            )
            observations.append(obs)

        # Run non-linear ballistic trajectory fit
        fitter = TrajectoryFitter(self.camera, gravity_m_s2=g)
        fit_result = fitter.fit_trajectory(observations, t0_release=0.0)

        self.assertTrue(fit_result.is_converged)
        self.assertTrue(fit_result.is_valid)
        self.assertEqual(fit_result.status, "MEASURED")
        self.assertLess(fit_result.rmse_pixels, 0.5)

        # Speed recovery accuracy: must match within 1.0 km/h of 135 km/h
        fitted_speed_kmh = fit_result.release_speed_km_h
        speed_error = abs(fitted_speed_kmh - gt_speed_km_h)
        self.assertLess(speed_error, 1.0, f"Speed error {speed_error:.2f} km/h exceeds tolerance.")

        # Release position accuracy
        self.assertAlmostEqual(fit_result.z0, gt_z0, delta=0.05)
        self.assertAlmostEqual(fit_result.y0, gt_y0, delta=0.10)

        # Check speed uncertainty
        self.assertGreater(fit_result.uncertainty_speed_kmh, 0.0)
        self.assertLess(fit_result.uncertainty_speed_kmh, 5.0)

    # -------------------------------------------------------------------------
    # 4. Outlier Observation Rejection During Trajectory Fitting
    # -------------------------------------------------------------------------

    def test_outlier_rejection_during_trajectory_fit(self):
        """
        Tests that an isolated corrupted detection (e.g. 15px off ray) is detected
        and rejected as an outlier, and the clean inliers fit the true trajectory.
        """
        gt_x0, gt_y0, gt_z0 = 0.0, 0.5, 2.1
        gt_vx0, gt_vy0, gt_vz0 = 0.0, 36.0, -1.8
        g = 9.80665

        observations = []
        dt_frame = 1.0 / 120.0

        for i in range(12):
            t = i * dt_frame
            x_t = gt_x0 + gt_vx0 * t
            y_t = gt_y0 + gt_vy0 * t
            z_t = gt_z0 + gt_vz0 * t - 0.5 * g * (t ** 2)

            u, v = self.camera.project_point(np.array([x_t, y_t, z_t]))
            dist_cam = float(np.linalg.norm(np.array([x_t, y_t, z_t]) - self.camera.optical_centre))
            diam_px = (1800.0 * 0.0715) / dist_cam

            # Corrupt frame 6 with an image displacement glitch (+18 px in v)
            if i == 6:
                v += 18.0

            half_d = diam_px / 2.0
            obs = BallObservation.from_bbox(
                frame_idx=i,
                timestamp_sec=t,
                bbox=[u - half_d, v - half_d, u + half_d, v + half_d],
                confidence=0.90
            )
            observations.append(obs)

        fitter = TrajectoryFitter(self.camera, gravity_m_s2=g)
        fit_result = fitter.fit_trajectory(observations, t0_release=0.0)

        self.assertTrue(fit_result.is_valid)
        self.assertGreaterEqual(fit_result.outliers_rejected_count, 1)
        self.assertLess(fit_result.rmse_pixels, 1.0)
        self.assertAlmostEqual(fit_result.vy0, gt_vy0, delta=1.5)

    # -------------------------------------------------------------------------
    # 5. Depth Unobservability & Insufficient Observations (NOT_MEASURED)
    # -------------------------------------------------------------------------

    def test_insufficient_observations_returns_not_measured(self):
        """
        REQUIREMENT: 'If depth is not sufficiently observable, return NOT_MEASURED.'
        'Do not force an unreliable 3D trajectory.'
        """
        fitter = TrajectoryFitter(self.camera)

        # Only 2 observations provided
        obs_sparse = [
            BallObservation.from_bbox(0, 0.0, [950.0, 500.0, 970.0, 520.0], 0.85),
            BallObservation.from_bbox(1, 0.00833, [952.0, 505.0, 972.0, 525.0], 0.85)
        ]

        result = fitter.fit_trajectory(obs_sparse, t0_release=0.0)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, "NOT_MEASURED")
        self.assertIsNotNone(result.rejection_reason)
        self.assertIn("INSUFFICIENT_OBSERVATIONS", result.rejection_reason)

    # -------------------------------------------------------------------------
    # 6. Observed vs Predicted Point Classification
    # -------------------------------------------------------------------------

    def test_observed_vs_predicted_point_classification(self):
        """
        Verifies fitted 3D points are strictly classified as OBSERVED for valid inliers.
        """
        observations = []
        for i in range(8):
            t = i * 0.00833
            pos = np.array([0.0, 0.5 + 35.0 * t, 2.1 - 2.0 * t - 0.5 * 9.80665 * (t**2)])
            u, v = self.camera.project_point(pos)
            dist_cam = float(np.linalg.norm(pos - self.camera.optical_centre))
            diam_px = (1800.0 * 0.0715) / dist_cam
            half_d = diam_px / 2.0
            obs = BallObservation.from_bbox(i, t, [u - half_d, v - half_d, u + half_d, v + half_d], 0.90)
            observations.append(obs)

        fitter = TrajectoryFitter(self.camera)
        result = fitter.fit_trajectory(observations, t0_release=0.0)

        self.assertEqual(len(result.fitted_3d_points), 8)
        for pt in result.fitted_3d_points:
            self.assertEqual(pt.point_type, "OBSERVED")
            self.assertTrue(pt.is_real_detection)
            self.assertGreater(pt.depth_from_camera_m, 0.0)
            self.assertGreater(pt.depth_uncertainty_m, 0.0)


if __name__ == "__main__":
    unittest.main()
