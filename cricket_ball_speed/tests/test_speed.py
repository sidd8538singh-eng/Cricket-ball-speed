"""
Unit tests for speed calculation, C1-C2 baseline, quality control, and radar benchmarks.
Includes tests using synthetic trajectories with known physical velocities.
"""

import unittest
import numpy as np
import tempfile
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from speed.velocity_estimator import VelocityEstimator
from speed.release_speed import compute_release_speed, ReleaseSpeedResult
from speed.c1_c2_speed import C1C2SpeedCalculator, C1C2SpeedResult
from speed.framewise_speed import FramewiseSpeedCalculator, FramewiseSpeedReport
from speed.speed_validator import SpeedValidator
from speed.speed_orchestrator import BowlingSpeedEngine, BowlingSpeedOutput
from reconstruction.trajectory_fitting import TrajectoryFitResult
from reconstruction.ball_3d import Ball3DPoint
from validation.quality_control import QualityControlPipeline
from validation.radar_validation import RadarValidator
from calibration.reprojection import ReprojectionReport
from release.release_detector import ReleaseEvent
from output.excel_report import ExcelReportWriter


class TestSpeed(unittest.TestCase):
    def test_velocity_estimator_conversion(self):
        """Verifies exact 3.6 conversion factor between m/s and km/h."""
        # A ball moving at 40 m/s along Y
        m_s, km_h = VelocityEstimator.vector_to_speed(0.0, 40.0, 0.0)
        self.assertAlmostEqual(m_s, 40.0)
        self.assertAlmostEqual(km_h, 144.0)

        # 3D vector: vx=3, vy=4, vz=0 -> norm = 5 m/s -> 18.0 km/h
        m_s, km_h = VelocityEstimator.vector_to_speed(3.0, 4.0, 0.0)
        self.assertAlmostEqual(m_s, 5.0)
        self.assertAlmostEqual(km_h, 18.0)

    def test_flight_angles_elevation_azimuth(self):
        """Verifies elevation and azimuth angle computations."""
        # Ball released downward along Y axis: vy=35.0, vz=-2.0, vx=0.5
        elev, azim = VelocityEstimator.compute_trajectory_angles_deg(0.5, 35.0, -2.0)
        self.assertLess(elev, 0.0)  # Downward elevation
        self.assertAlmostEqual(elev, np.degrees(np.arctan2(-2.0, np.hypot(0.5, 35.0))), delta=0.01)
        self.assertGreater(azim, 0.0)  # Slight drift towards off-side
        self.assertAlmostEqual(azim, np.degrees(np.arctan2(0.5, 35.0)), delta=0.01)

    def test_synthetic_trajectory_instantaneous_release_speed(self):
        """
        Tests instantaneous 3D release speed calculation against a synthetic trajectory
        with known physical velocities (e.g. 135 km/h delivery).
        """
        # Ground-truth: 135.0 km/h = 37.5 m/s
        vx_true = 0.2
        vy_true = 37.45
        vz_true = -1.95
        speed_true_ms = float(np.sqrt(vx_true**2 + vy_true**2 + vz_true**2))
        speed_true_kmh = speed_true_ms * 3.6

        fit_res = TrajectoryFitResult(
            t0=0.1500,
            x0=-0.12,
            y0=0.65,
            z0=2.15,
            vx0=vx_true,
            vy0=vy_true,
            vz0=vz_true,
            rmse_pixels=1.1,
            uncertainty_speed_kmh=2.4,
            covariance_matrix=np.eye(6),
            fitted_3d_points=[],
            is_converged=True,
            is_valid=True,
            status="MEASURED"
        )

        res = compute_release_speed(fit_res)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.status, "MEASURED")
        self.assertAlmostEqual(res.speed_m_s, speed_true_ms, delta=0.01)
        self.assertAlmostEqual(res.speed_km_h, speed_true_kmh, delta=0.02)
        self.assertAlmostEqual(res.speed_km_h, res.speed_m_s * 3.6, delta=1e-5)
        self.assertEqual(res.velocity_vector_m_s, (vx_true, vy_true, vz_true))
        self.assertEqual(res.c1_release_pos_m, (-0.12, 0.65, 2.15))
        self.assertGreater(len(res.calculation_logs), 3)

    def test_dynamic_c2_selection_not_fixed_frame(self):
        """
        Verifies that C2 is selected dynamically based on physical 3D distance targeting ~6.5m,
        and NOT by a fixed frame number, across different frame rates (60fps vs 120fps).
        """
        calc = C1C2SpeedCalculator(target_distance_m=6.5, min_distance_m=4.5, max_distance_m=8.5)
        c1_pos = np.array([0.0, 0.5, 2.1])
        t1 = 0.0

        # Case A: 60 fps (dt = 0.0167s), speed = 35 m/s (~126 km/h)
        # At 35 m/s, 6.5m is reached around t = 6.5 / 35 = 0.1857s -> ~11 frames
        pts_60fps = []
        for f in range(1, 30):
            t = f / 60.0
            p = np.array([0.0, 0.5 + 35.0 * t, 2.1 - 0.5 * 9.8 * t**2])
            pts_60fps.append(Ball3DPoint(f, t, p, np.zeros(3), "fitted"))

        res_60 = calc.compute_c1_c2(c1_pos, t1, pts_60fps)
        self.assertTrue(res_60.is_valid)
        self.assertAlmostEqual(res_60.distance_3d_m, 6.5, delta=0.6)
        # Verify C2 frame index is around 11
        self.assertIn(res_60.c2_frame_index, [10, 11, 12])

        # Case B: 120 fps (dt = 0.00833s), same speed = 35 m/s
        # At 120 fps, 6.5m is reached around t = 0.1857s -> ~22 frames (DIFFERENT frame number!)
        pts_120fps = []
        for f in range(1, 60):
            t = f / 120.0
            p = np.array([0.0, 0.5 + 35.0 * t, 2.1 - 0.5 * 9.8 * t**2])
            pts_120fps.append(Ball3DPoint(f, t, p, np.zeros(3), "fitted"))

        res_120 = calc.compute_c1_c2(c1_pos, t1, pts_120fps)
        self.assertTrue(res_120.is_valid)
        self.assertAlmostEqual(res_120.distance_3d_m, 6.5, delta=0.4)
        # Dynamic selection correctly picked frame ~22, proving it does not use a fixed frame number
        self.assertIn(res_120.c2_frame_index, [21, 22, 23])
        self.assertNotEqual(res_60.c2_frame_index, res_120.c2_frame_index)

    def test_c1_c2_insufficient_baseline_returns_not_measured(self):
        """Verifies that if trajectory reaches only 2 metres, C1-C2 returns NOT_MEASURED."""
        calc = C1C2SpeedCalculator(target_distance_m=6.5, min_distance_m=4.0)
        c1_pos = np.array([0.0, 0.5, 2.1])
        t1 = 0.0

        # Trajectory only has 3 points reaching 2.0 metres
        short_pts = [
            Ball3DPoint(1, 0.02, np.array([0.0, 1.0, 2.05]), np.zeros(3), "fitted"),
            Ball3DPoint(2, 0.04, np.array([0.0, 1.8, 2.00]), np.zeros(3), "fitted"),
            Ball3DPoint(3, 0.06, np.array([0.0, 2.5, 1.95]), np.zeros(3), "fitted"),  # Dist = 2.0m from C1
        ]
        res = calc.compute_c1_c2(c1_pos, t1, short_pts)
        self.assertFalse(res.is_valid)
        self.assertEqual(res.status, "NOT_MEASURED")
        self.assertIn("BASELINE_TOO_SHORT", res.rejection_reason)

    def test_framewise_speed_and_deceleration_profile(self):
        """Tests framewise speed profiles, accelerations, and deceleration reporting."""
        # 4 points moving at ~35 m/s down pitch with slight drag deceleration
        dt = 0.02
        points = [
            Ball3DPoint(0, 0.00, np.array([0.0, 0.0, 2.0]), np.zeros(3), "fitted"),
            Ball3DPoint(1, 0.02, np.array([0.0, 0.70, 1.99]), np.zeros(3), "fitted"),  # 35 m/s
            Ball3DPoint(2, 0.04, np.array([0.0, 1.39, 1.96]), np.zeros(3), "fitted"),  # 34.5 m/s
            Ball3DPoint(3, 0.06, np.array([0.0, 2.07, 1.91]), np.zeros(3), "fitted"),  # 34 m/s
        ]
        report = FramewiseSpeedCalculator.compute_profile(points, window_size=3)
        self.assertEqual(len(report.records), 3)
        self.assertGreater(report.mean_speed_km_h, 120.0)
        self.assertLess(report.mean_speed_km_h, 130.0)
        self.assertGreater(report.mean_deceleration_m_s2, 0.0)  # Slowing down
        # Check backward-compatible method
        legacy_records = FramewiseSpeedCalculator.compute_framewise_speeds(points, window_size=3)
        self.assertEqual(len(legacy_records), 3)

    def test_speed_validator_bounds_and_aerodynamics(self):
        """Tests physical limits, uncertainty bounds, and aerodynamic drag consistency."""
        validator = SpeedValidator()

        # Valid delivery: 138 km/h, C1-C2 = 134 km/h (normal aerodynamic drag loss)
        valid_rel = ReleaseSpeedResult(
            speed_km_h=138.0,
            speed_m_s=38.33,
            uncertainty_km_h=2.8,
            ci_95_lower_km_h=132.5,
            ci_95_upper_km_h=143.5,
            velocity_vector_m_s=(0.0, 38.3, -1.8),
            c1_release_pos_m=(0.0, 0.5, 2.1),
            release_timestamp_sec=0.0
        )
        valid_c1c2 = C1C2SpeedResult(
            c1_pos_m=(0.0, 0.5, 2.1),
            c2_pos_m=(0.0, 7.0, 1.8),
            t1_sec=0.0,
            t2_sec=0.174,
            distance_3d_m=6.51,
            time_delta_sec=0.174,
            average_speed_km_h=134.7,
            average_speed_m_s=37.4,
            is_valid=True,
            status="MEASURED"
        )
        is_ok, reason = validator.validate(valid_rel, valid_c1c2)
        self.assertTrue(is_ok)
        self.assertIsNone(reason)

        # Unphysical aerodynamic acceleration: C1-C2 speed is 155 km/h while release is 138 km/h (+17 km/h!)
        unphysical_c1c2 = C1C2SpeedResult(
            c1_pos_m=(0.0, 0.5, 2.1),
            c2_pos_m=(0.0, 7.0, 1.8),
            t1_sec=0.0,
            t2_sec=0.151,
            distance_3d_m=6.51,
            time_delta_sec=0.151,
            average_speed_km_h=155.2,
            average_speed_m_s=43.1,
            is_valid=True,
            status="MEASURED"
        )
        is_ok, reason = validator.validate(valid_rel, unphysical_c1c2)
        self.assertFalse(is_ok)
        self.assertIn("AERODYNAMIC_INCONSISTENCY_ACCELERATION", reason)

    def test_bowling_speed_engine_full_workflow(self):
        """Tests complete orchestration of release speed, C1-C2, framewise, and validation."""
        engine = BowlingSpeedEngine()

        # Construct synthetic delivery: 140.0 km/h (38.89 m/s)
        vx = 0.3
        vy = 38.85
        vz = -1.75
        fit_res = TrajectoryFitResult(
            t0=0.05,
            x0=0.05,
            y0=0.45,
            z0=2.12,
            vx0=vx,
            vy0=vy,
            vz0=vz,
            rmse_pixels=0.9,
            uncertainty_speed_kmh=1.8,
            covariance_matrix=np.eye(6),
            fitted_3d_points=[],
            is_converged=True,
            is_valid=True,
            status="MEASURED"
        )

        pts = []
        for i in range(1, 25):
            t = 0.05 + i * 0.01
            p = np.array([0.05 + vx * (t - 0.05), 0.45 + vy * (t - 0.05), 2.12 + vz * (t - 0.05)])
            pts.append(Ball3DPoint(i, t, p, np.zeros(3), "fitted"))

        output = engine.compute(fit_res, pts)
        self.assertEqual(output.status, "MEASURED")
        self.assertTrue(output.is_measured)
        self.assertAlmostEqual(output.release_speed_km_h, 140.0, delta=0.5)
        self.assertIsNotNone(output.c1_c2_speed_km_h)
        self.assertAlmostEqual(output.c1_c2_speed_km_h, 140.0, delta=1.0)
        self.assertIsNotNone(output.framewise_report)
        self.assertGreater(len(output.master_logs), 5)

    def test_bowling_speed_engine_invalid_trajectory_returns_not_measured(self):
        """Tests that invalid trajectory returns status='NOT_MEASURED' without raising."""
        engine = BowlingSpeedEngine()
        invalid_fit = TrajectoryFitResult(
            t0=0.0, x0=0, y0=0, z0=0, vx0=0, vy0=0, vz0=0,
            rmse_pixels=10.0, uncertainty_speed_kmh=20.0,
            covariance_matrix=np.eye(6), fitted_3d_points=[],
            is_converged=False, is_valid=False, status="NOT_MEASURED", rejection_reason="FIT_RESIDUAL_TOO_HIGH"
        )
        out = engine.compute(invalid_fit, [])
        self.assertEqual(out.status, "NOT_MEASURED")
        self.assertFalse(out.is_measured)
        self.assertIn("FIT_RESIDUAL_TOO_HIGH", out.rejection_reason)
        self.assertEqual(out.release_speed_km_h, 0.0)

    def test_quality_control_pipeline(self):
        """Tests QC pipeline auditing passing vs failing deliveries."""
        qc = QualityControlPipeline(min_ball_detections=5)

        reproj = ReprojectionReport(
            per_point_errors={}, mean_error_px=2.1, median_error_px=2.0,
            max_error_px=3.5, worst_point="near_top", confidence_score=0.92, is_acceptable=True
        )
        release_ev = ReleaseEvent(0, 0.0, (500, 300), 0.85, (500, 300), 10, 800.0)
        traj_res = TrajectoryFitResult(0.0, 0, 0.5, 2.1, 0, 37.5, -2.0, 1.2, 3.2, np.eye(6), [], True)
        rel_speed = ReleaseSpeedResult(135.0, 37.5, 3.2, 128.7, 141.3, (0, 37.5, -2), (0, 0.5, 2.1), 0.0)

        # Passing delivery
        res_pass = qc.audit(
            video_fps=120.0,
            is_camera_stable=True,
            reprojection_report=reproj,
            num_ball_detections=10,
            release_event=release_ev,
            trajectory_result=traj_res,
            release_speed_result=rel_speed
        )
        self.assertEqual(res_pass.status, "MEASURED")
        self.assertTrue(res_pass.is_measured)

        # Failing delivery: camera shake detected
        res_fail = qc.audit(
            video_fps=120.0,
            is_camera_stable=False, # UNSTABLE
            reprojection_report=reproj,
            num_ball_detections=10,
            release_event=release_ev,
            trajectory_result=traj_res,
            release_speed_result=rel_speed
        )
        self.assertEqual(res_fail.status, "NOT_MEASURED")
        self.assertFalse(res_fail.is_measured)
        self.assertIn("CAMERA_UNSTABLE", res_fail.rejection_reason)

    def test_radar_validation_metrics(self):
        """Tests independent radar evaluation: MAE, RMSE, Bias, and +-2km/h percentage."""
        radar = RadarValidator()

        # Add 4 test comparisons
        radar.add_comparison("deliv1.mp4", 136.2, 135.0, is_measured=True) # +1.2
        radar.add_comparison("deliv2.mp4", 140.5, 141.0, is_measured=True) # -0.5
        radar.add_comparison("deliv3.mp4", 128.0, 126.5, is_measured=True) # +1.5
        radar.add_comparison("deliv4.mp4", None, 130.0, is_measured=False, rejection_reason="CAMERA_UNSTABLE")

        summary = radar.compute_summary()
        self.assertEqual(summary.total_deliveries, 4)
        self.assertEqual(summary.measured_deliveries, 3)
        self.assertEqual(summary.not_measured_deliveries, 1)
        self.assertAlmostEqual(summary.coverage_percentage, 75.0)

        # All 3 measured are within +- 2 km/h
        self.assertAlmostEqual(summary.within_2kmh_percentage, 100.0)
        self.assertLess(summary.mae_kmh, 1.5)

    def test_excel_report_writer(self):
        """Tests Excel workbook writing and formatting."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            excel_path = os.path.join(tmp_dir, "test_report.xlsx")
            writer = ExcelReportWriter(excel_path)
            writer.add_delivery_row({
                "filename": "ball_01.mp4",
                "calculated_speed": 138.4,
                "radar_speed": 137.5,
                "c1_c2_speed": 134.2,
                "calibration_status": "VALID",
                "reprojection_error_px": 1.45,
                "detected_frames": 12,
                "predicted_frames": 1,
                "release_confidence": 0.88,
                "trajectory_rmse_px": 0.95,
                "speed_uncertainty_kmh": 2.5,
                "final_status": "MEASURED",
            })
            saved_file = writer.save()
            self.assertTrue(os.path.isfile(saved_file))
            self.assertGreater(os.path.getsize(saved_file), 1000)


if __name__ == "__main__":
    unittest.main()
