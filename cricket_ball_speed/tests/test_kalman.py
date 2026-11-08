"""
Comprehensive Unit Tests for Ball Observation Refinement, Kalman Filtering,
Adaptive Gating, Data Association, and Track Lifecycle Management.
Includes synthetic noisy trajectory tests with known ground-truth positions and velocities.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from filtering.kalman_2d import Kalman2D
from filtering.kalman_3d import Kalman3D
from filtering.data_association import DataAssociator
from filtering.prediction_manager import PredictionManager, TrackStatePoint
from ball_processing.ball_observations import BallObservation
from ball_processing.outlier_rejection import OutlierRejector


class TestKalmanSubsystem(unittest.TestCase):

    # -------------------------------------------------------------------------
    # 1. 2D Kalman Filter: State Transition & Covariance Propagation
    # -------------------------------------------------------------------------

    def test_kalman_2d_covariance_propagation_nonuniform_dt(self):
        """
        Validates 2D discrete continuous white noise acceleration (CWNA) covariance propagation
        across nonuniform frame intervals dt.
        """
        kf = Kalman2D(initial_u=500.0, initial_v=300.0, process_noise_std=100.0, measurement_noise_std=2.0)
        p_initial = kf.covariance.copy()

        # Step 1: Predict with dt = 0.0081 sec (nonuniform high-speed interval)
        dt1 = 0.0081
        kf.predict(dt1)
        p_pred1 = kf.covariance

        # Covariance must grow during prediction
        self.assertGreater(p_pred1[0, 0], p_initial[0, 0])
        self.assertGreater(p_pred1[1, 1], p_initial[1, 1])
        # P must be symmetric positive definite
        self.assertTrue(np.allclose(p_pred1, p_pred1.T))
        eigenvals = np.linalg.eigvals(p_pred1)
        self.assertTrue(np.all(eigenvals > 0.0))

        # Step 2: Measurement update with high confidence (0.95)
        kf.update(504.0, 301.5, confidence=0.95)
        p_upd1 = kf.covariance

        # Measurement update must contract uncertainty
        self.assertLess(p_upd1[0, 0], p_pred1[0, 0])
        self.assertLess(p_upd1[1, 1], p_pred1[1, 1])

        # Step 3: Predict with larger dt = 0.0167 sec (dropped frame interval)
        dt2 = 0.0167
        kf.predict(dt2)
        p_pred2 = kf.covariance
        self.assertGreater(p_pred2[0, 0], p_upd1[0, 0])

    def test_kalman_2d_adaptive_gating_and_mahalanobis(self):
        """
        Tests adaptive Mahalanobis distance calculation and gating against predicted covariance.
        """
        kf = Kalman2D(initial_u=100.0, initial_v=100.0, measurement_noise_std=2.0)
        kf.predict(0.01)

        # 1. Detection at exact predicted position should have Mahalanobis distance ~ 0
        d_m_zero, _, _ = kf.compute_mahalanobis(100.0, 100.0, confidence=1.0)
        self.assertLess(d_m_zero, 0.2)

        # 2. Close valid detection should pass gate (threshold = 3.5)
        in_gate, d_m_valid, _ = kf.is_within_gate(102.0, 101.5, confidence=0.85, gate_threshold=3.5)
        self.assertTrue(in_gate)
        self.assertLess(d_m_valid, 3.5)

        # 3. Distant false positive (clutter 80px away) must fail gate
        in_gate_bad, d_m_bad, _ = kf.is_within_gate(180.0, 180.0, confidence=0.85, gate_threshold=3.5)
        self.assertFalse(in_gate_bad)
        self.assertGreater(d_m_bad, 5.0)

        # 4. Low-confidence detection inflates measurement noise R, increasing innovation variance
        d_m_high_conf, _, S_high = kf.compute_mahalanobis(105.0, 105.0, confidence=1.0)
        d_m_low_conf, _, S_low = kf.compute_mahalanobis(105.0, 105.0, confidence=0.2)
        self.assertGreater(S_low[0, 0], S_high[0, 0])

    # -------------------------------------------------------------------------
    # 2. Synthetic Noisy 2D Trajectory with Known Ground Truth
    # -------------------------------------------------------------------------

    def test_kalman_2d_synthetic_noisy_trajectory(self):
        """
        RIGOROUS TEST: Simulates a known cricket ball 2D flight path:
        True initial pos: (200.0, 300.0) px
        True velocities: Vu = 550.0 px/s, Vv = 180.0 px/s
        Adds Gaussian sensor measurement noise N(0, 2.5^2).
        Verifies filter convergence to ground truth within theoretical 2-sigma bounds.
        """
        np.random.seed(42)
        true_u0, true_v0 = 200.0, 300.0
        true_vu, true_vv = 550.0, 180.0

        kf = Kalman2D(initial_u=true_u0, initial_v=true_v0, process_noise_std=60.0, measurement_noise_std=2.5)

        # Nonuniform timestamps (120 FPS nominal with realistic camera jitter)
        dts = [0.0082, 0.0084, 0.0083, 0.0081, 0.0085, 0.0082, 0.0083, 0.0084] * 4
        cur_u, cur_v = true_u0, true_v0

        residuals_pos = []
        residuals_vel = []

        for dt in dts:
            cur_u += true_vu * dt
            cur_v += true_vv * dt

            # Add Gaussian noise
            meas_u = cur_u + np.random.normal(0, 2.5)
            meas_v = cur_v + np.random.normal(0, 2.5)

            kf.predict(dt)
            kf.update(meas_u, meas_v, confidence=0.90)

            est_u, est_v = kf.position
            est_vu, est_vv = kf.velocity

            residuals_pos.append(np.hypot(est_u - cur_u, est_v - cur_v))
            residuals_vel.append(np.hypot(est_vu - true_vu, est_vv - true_vv))

        # Positional RMSE after convergence should be smaller than raw measurement noise
        late_pos_rmse = float(np.mean(residuals_pos[15:]))
        self.assertLess(late_pos_rmse, 2.0)

        # Velocity estimates should converge close to ground truth (550, 180)
        final_vu, final_vv = kf.velocity
        self.assertAlmostEqual(final_vu, true_vu, delta=35.0)
        self.assertAlmostEqual(final_vv, true_vv, delta=30.0)

        # Verify uncertainties are reported
        sig_u, sig_v = kf.get_position_uncertainty()
        sig_vu, sig_vv = kf.get_velocity_uncertainty()
        self.assertGreater(sig_u, 0.0)
        self.assertGreater(sig_vu, 0.0)

    # -------------------------------------------------------------------------
    # 3. 3D World-Space Kalman Filter: Physical Ballistic Motion
    # -------------------------------------------------------------------------

    def test_kalman_3d_ballistics_known_trajectory(self):
        """
        RIGOROUS TEST: 3D cricket bowling trajectory with known physical ground truth:
        Release point: (X=0.0, Y=1.8, Z=2.1) metres
        Release velocity: (VX=0.2, VY=38.0, VZ=0.8) m/s  (Speed ~ 136.8 km/h)
        Gravity: g = 9.80665 m/s^2 acts downward in Z.
        """
        np.random.seed(123)
        init_xyz = np.array([0.0, 1.8, 2.1])
        true_vel = np.array([0.2, 38.0, 0.8])
        true_speed_km_h = np.linalg.norm(true_vel) * 3.6  # ~136.85 km/h

        kf3d = Kalman3D(
            initial_xyz=init_xyz,
            initial_vxyz=np.array([0.0, 30.0, 0.0]), # Slightly off initial guess
            process_noise_std=0.8
        )

        dt = 0.00833 # 120 FPS
        cur_pos = init_xyz.copy()
        cur_vel = true_vel.copy()

        for _ in range(25):
            # True ballistic propagation
            cur_pos[0] += cur_vel[0] * dt
            cur_pos[1] += cur_vel[1] * dt
            cur_pos[2] += cur_vel[2] * dt - 0.5 * 9.80665 * (dt ** 2)
            cur_vel[2] -= 9.80665 * dt

            # Synthetic noisy 3D triangulation observation (std: 1.5cm lateral, 5cm depth)
            noise = np.random.normal(0, [0.015, 0.05, 0.015])
            obs_xyz = cur_pos + noise

            kf3d.predict(dt)
            kf3d.update(obs_xyz, measurement_cov_diag=[0.015**2, 0.05**2, 0.015**2])

        # Verify estimated position matches ground truth
        est_pos = kf3d.position_xyz
        np.testing.assert_allclose(est_pos, cur_pos, atol=0.04)

        # Verify estimated speed converges to true speed within 1.5 km/h
        est_speed_km_h = kf3d.speed_km_h
        current_true_speed_km_h = np.linalg.norm(cur_vel) * 3.6
        self.assertAlmostEqual(est_speed_km_h, current_true_speed_km_h, delta=1.5)

        # Check speed uncertainty is reported and non-zero
        speed_unc = kf3d.get_speed_uncertainty()
        self.assertGreater(speed_unc, 0.0)
        self.assertLess(speed_unc, 5.0)

    # -------------------------------------------------------------------------
    # 4. Data Association: Clutter & Distractor Rejection
    # -------------------------------------------------------------------------

    def test_data_association_clutter_rejection(self):
        """
        Tests that DataAssociator correctly matches true ball candidate while rejecting
        isolated clutter and false positives, recording explicit rejection reasons.
        """
        kf = Kalman2D(initial_u=300.0, initial_v=400.0)
        kf.predict(0.01)

        associator = DataAssociator(mahalanobis_gate_threshold=3.5, max_pixel_distance=80.0)

        # True candidate near predicted position (302, 401)
        true_cand = BallObservation.from_bbox(1, 0.01, [296.0, 395.0, 308.0, 407.0], 0.88, "tile_x0_y0")

        # Distant clutter candidate (e.g. bowler hand or footwear at 390, 420)
        clutter_cand = BallObservation.from_bbox(1, 0.01, [380.0, 410.0, 400.0, 430.0], 0.75, "tile_x0_y0")

        # Another candidate outside Mahalanobis gate
        gate_outlier = BallObservation.from_bbox(1, 0.01, [325.0, 425.0, 335.0, 435.0], 0.70, "tile_x0_y0")

        candidates = [clutter_cand, true_cand, gate_outlier]
        matched, unmatched = associator.associate(kf, candidates)

        # True candidate must be matched
        self.assertIs(matched, true_cand)

        # Clutter candidates must be rejected with explicit reasons
        self.assertEqual(len(unmatched), 2)
        for cand in unmatched:
            self.assertEqual(cand.acceptance_status, "REJECTED")
            self.assertIsNotNone(cand.rejection_reason)
            self.assertTrue(len(cand.rejection_reason) > 5)

    # -------------------------------------------------------------------------
    # 5. Prediction Manager: Strict Provenance & Gap Enforcement
    # -------------------------------------------------------------------------

    def test_prediction_manager_strict_provenance(self):
        """
        CRITICAL ARCHITECTURAL TEST:
        'Never call a predicted position a real detection.'
        'Never allow a long prediction-only gap to create a fake continuous trajectory.'
        """
        pm = PredictionManager(max_consecutive_predictions=3, min_hits_to_confirm=2)

        # Frame 0: Initialization
        obs0 = BallObservation.from_bbox(0, 0.0, [100.0, 200.0, 114.0, 214.0], 0.85)
        pm.initialize_track(obs0)
        self.assertTrue(pm.history[0].is_real_detection)
        self.assertEqual(pm.history[0].point_type, "OBSERVED")

        # Frame 1: Real detection present
        obs1 = BallObservation.from_bbox(1, 0.00833, [105.0, 202.0, 119.0, 216.0], 0.85)
        pt1 = pm.step(1, 0.00833, [obs1])
        self.assertTrue(pt1.is_real_detection)
        self.assertEqual(pt1.point_type, "FILTERED")

        # Frame 2: Missing observation (Prediction 1)
        pt2 = pm.step(2, 0.01667, [])
        self.assertIsNotNone(pt2)
        self.assertFalse(pt2.is_real_detection) # MUST BE STRICTLY FALSE
        self.assertEqual(pt2.point_type, "PREDICTED")
        self.assertIsNone(pt2.observation)

        # Frame 3: Missing observation (Prediction 2)
        pt3 = pm.step(3, 0.02500, [])
        self.assertFalse(pt3.is_real_detection)

        # Frame 4: Missing observation (Prediction 3)
        pt4 = pm.step(4, 0.03333, [])
        self.assertFalse(pt4.is_real_detection)

        # Frame 5: Missing observation (Prediction 4 > max_pred_gap=3) -> Track MUST terminate
        pt5 = pm.step(5, 0.04167, [])
        self.assertIsNone(pt5)
        self.assertFalse(pm.is_active)

        # Verify get_real_detected_points returns ONLY points 0 and 1
        real_pts = pm.get_real_detected_points()
        self.assertEqual(len(real_pts), 2)
        for rp in real_pts:
            self.assertTrue(rp.is_real_detection)

        # Verify get_predicted_points returns exactly 3 predictions
        pred_pts = pm.get_predicted_points()
        self.assertEqual(len(pred_pts), 3)
        for pp in pred_pts:
            self.assertFalse(pp.is_real_detection)

    # -------------------------------------------------------------------------
    # 6. Outlier Rejector: Kinematic & Morphological Constraints
    # -------------------------------------------------------------------------

    def test_outlier_rejector_with_nonuniform_dt(self):
        """
        Tests outlier rejection with dynamic timestamp delta scaling and rejection auditing.
        """
        rejector = OutlierRejector(
            max_interframe_displacement_px=100.0,
            nominal_dt_sec=0.00833,
            max_velocity_px_s=15000.0
        )

        obs_prev = BallObservation.from_bbox(1, 0.0, [500.0, 500.0, 514.0, 514.0], 0.85)
        obs_prev.mark_accepted()

        # Valid small displacement: 10px in 8.33ms -> v ~ 1200 px/s
        obs_valid = BallObservation.from_bbox(2, 0.00833, [508.0, 506.0, 522.0, 520.0], 0.85)

        # Kinematic jump: 300px in 8.33ms -> v ~ 36000 px/s (physically impossible)
        obs_jump = BallObservation.from_bbox(2, 0.00833, [750.0, 680.0, 764.0, 694.0], 0.85)

        # Sudden diameter explosion (e.g. 14px -> 48px)
        obs_fat = BallObservation.from_bbox(2, 0.00833, [505.0, 505.0, 553.0, 553.0], 0.85)

        accepted = rejector.filter_candidates([obs_valid, obs_jump, obs_fat], last_accepted=obs_prev, dt_sec=0.00833)

        self.assertIn(obs_valid, accepted)
        self.assertNotIn(obs_jump, accepted)
        self.assertNotIn(obs_fat, accepted)

        self.assertEqual(obs_jump.acceptance_status, "REJECTED")
        self.assertIn("Kinematic jump", obs_jump.rejection_reason)

        self.assertEqual(obs_fat.acceptance_status, "REJECTED")
        self.assertIn("diameter change ratio", obs_fat.rejection_reason)


if __name__ == "__main__":
    unittest.main()
