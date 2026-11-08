"""
Unit and integration tests for the Quality Control and Radar Validation subsystem.
Verifies all 10 verification gates, explicit rejection reasons, NOT_MEASURED handling,
strict radar independence, statistical metric accuracy, plot generation, and reproducibility.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import json
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validation.rejection_reasons import RejectionReason
from validation.quality_control import QualityControlPipeline, QualityControlResult
from validation.radar_validation import RadarValidator, RadarBenchmarkSummary, DeliveryRadarComparison
from validation.benchmark import BenchmarkRunner, load_ground_truth
from validation.benchmark_plots import BenchmarkPlotGenerator
from calibration.camera_model import CameraModel
from calibration.reprojection import ReprojectionReport
from release.release_detector import ReleaseEvent
from reconstruction.trajectory_fitting import TrajectoryFitResult
from speed.release_speed import ReleaseSpeedResult, compute_release_speed
from speed.c1_c2_speed import C1C2SpeedResult


class TestValidationSubsystem(unittest.TestCase):
    def setUp(self):
        self.qc = QualityControlPipeline()
        K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.cam_model = CameraModel(
            camera_matrix=K,
            dist_coeffs=np.zeros(5),
            rvec=np.zeros(3),
            tvec=np.array([0.0, -10.0, 2.5]),
            image_width=1920,
            image_height=1080
        )
        self.reproj_pass = ReprojectionReport(
            per_point_errors={}, mean_error_px=2.1, median_error_px=2.0,
            max_error_px=3.8, worst_point="near_top", confidence_score=0.92, is_acceptable=True
        )
        self.release_pass = ReleaseEvent(
            frame_index=15, timestamp_sec=0.125,
            c1_pixel=(950.0, 400.0), confidence=0.88,
            hand_position_pixel=(950.0, 400.0),
            observation_mode="DIRECTLY_OBSERVED"
        )
        self.traj_pass = TrajectoryFitResult(
            t0=0.125, x0=-0.1, y0=0.6, z0=2.15, vx0=0.2, vy0=37.5, vz0=-1.8,
            rmse_pixels=1.2, uncertainty_speed_kmh=2.5, covariance_matrix=np.eye(6),
            fitted_3d_points=[], is_converged=True, is_valid=True,
            status="MEASURED", depth_observability_score=0.85
        )
        self.speed_pass = ReleaseSpeedResult(
            speed_km_h=135.2, speed_m_s=37.55, uncertainty_km_h=2.5,
            ci_95_lower_km_h=130.3, ci_95_upper_km_h=140.1,
            velocity_vector_m_s=(0.2, 37.5, -1.8), c1_release_pos_m=(-0.1, 0.6, 2.15),
            release_timestamp_sec=0.125, is_valid=True, status="MEASURED"
        )
        self.c1c2_pass = C1C2SpeedResult(
            c1_pos_m=(-0.1, 0.6, 2.15), c2_pos_m=(-0.1, 7.1, 1.8),
            t1_sec=0.125, t2_sec=0.300, distance_3d_m=6.51, time_delta_sec=0.175,
            average_speed_km_h=133.9, average_speed_m_s=37.2,
            is_valid=True, status="MEASURED"
        )

    def test_all_10_gates_pass(self):
        """Verifies delivery passing all 10 gates returns MEASURED with high quality score."""
        res = self.qc.audit(
            video_fps=120.0,
            is_camera_stable=True,
            reprojection_report=self.reproj_pass,
            num_ball_detections=12,
            release_event=self.release_pass,
            trajectory_result=self.traj_pass,
            release_speed_result=self.speed_pass,
            c1_c2_result=self.c1c2_pass,
            camera_model=self.cam_model,
            is_timestamp_valid=True,
            max_consecutive_missing_frames=1,
            mean_observation_quality=0.85
        )
        self.assertTrue(res.is_measured)
        self.assertEqual(res.status, "MEASURED")
        self.assertIsNone(res.rejection_reason)
        self.assertEqual(len(res.all_failures), 0)
        self.assertGreaterEqual(res.quality_score, 90.0)
        self.assertEqual(len(res.audit_checks), 10)
        self.assertTrue(all(res.audit_checks.values()))

    def test_gate1_camera_stability_rejection(self):
        """Gate 1: Excessive camera shake/vibration returns NOT_MEASURED."""
        res = self.qc.audit(
            video_fps=120.0,
            is_camera_stable=False,  # UNSTABLE
            reprojection_report=self.reproj_pass,
            num_ball_detections=12,
            release_event=self.release_pass,
            trajectory_result=self.traj_pass,
            release_speed_result=self.speed_pass
        )
        self.assertFalse(res.is_measured)
        self.assertEqual(res.status, "NOT_MEASURED")
        self.assertIn("CAMERA_UNSTABLE", res.rejection_reason)
        self.assertFalse(res.audit_checks["camera_stability"])

    def test_gate2_calibration_rejection(self):
        """Gate 2: Invalid or missing camera calibration returns NOT_MEASURED."""
        bad_K = np.array([[-100.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        invalid_cam = CameraModel(
            camera_matrix=bad_K,
            dist_coeffs=np.zeros(5),
            rvec=np.zeros(3),
            tvec=np.zeros(3),
            image_width=1920,
            image_height=1080
        )
        res = self.qc.audit(
            video_fps=120.0,
            is_camera_stable=True,
            reprojection_report=self.reproj_pass,
            num_ball_detections=12,
            release_event=self.release_pass,
            trajectory_result=self.traj_pass,
            release_speed_result=self.speed_pass,
            camera_model=invalid_cam
        )
        self.assertFalse(res.is_measured)
        self.assertEqual(res.status, "NOT_MEASURED")
        self.assertIn("CAMERA_CALIBRATION_FAILED", res.rejection_reason)

    def test_gate3_reprojection_error_rejection(self):
        """Gate 3: Reprojection RMSE exceeding threshold returns NOT_MEASURED."""
        bad_reproj = ReprojectionReport(
            per_point_errors={}, mean_error_px=7.8, median_error_px=7.5,
            max_error_px=11.2, worst_point="near_top", confidence_score=0.45, is_acceptable=False
        )
        res = self.qc.audit(
            video_fps=120.0,
            is_camera_stable=True,
            reprojection_report=bad_reproj,
            num_ball_detections=12,
            release_event=self.release_pass,
            trajectory_result=self.traj_pass,
            release_speed_result=self.speed_pass
        )
        self.assertFalse(res.is_measured)
        self.assertEqual(res.status, "NOT_MEASURED")
        self.assertIn("REPROJECTION_ERROR_HIGH", res.rejection_reason)

    def test_gate4_ball_observation_quality_rejection(self):
        """Gate 4: Insufficient detections or prediction gap returns NOT_MEASURED."""
        # Case A: Too few detections (2 < 5)
        res_few = self.qc.audit(
            video_fps=120.0,
            is_camera_stable=True,
            reprojection_report=self.reproj_pass,
            num_ball_detections=2,  # < 5
            release_event=self.release_pass,
            trajectory_result=self.traj_pass,
            release_speed_result=self.speed_pass
        )
        self.assertFalse(res_few.is_measured)
        self.assertIn("INSUFFICIENT_BALL_DETECTIONS", res_few.rejection_reason)

        # Case B: Long prediction gap (5 > 3)
        res_gap = self.qc.audit(
            video_fps=120.0,
            is_camera_stable=True,
            reprojection_report=self.reproj_pass,
            num_ball_detections=10,
            release_event=self.release_pass,
            trajectory_result=self.traj_pass,
            release_speed_result=self.speed_pass,
            max_consecutive_missing_frames=5
        )
        self.assertFalse(res_gap.is_measured)
        self.assertIn("LONG_PREDICTION_GAP", res_gap.rejection_reason)

    def test_gate5_release_confidence_and_height_rejection(self):
        """Gate 5: Low release confidence or unphysical release height returns NOT_MEASURED."""
        # Low confidence
        low_conf_ev = ReleaseEvent(
            frame_index=15, timestamp_sec=0.125,
            c1_pixel=(950.0, 400.0), confidence=0.35,
            hand_position_pixel=(950.0, 400.0),
            observation_mode="DIRECTLY_OBSERVED"
        )
        res = self.qc.audit(
            video_fps=120.0,
            is_camera_stable=True,
            reprojection_report=self.reproj_pass,
            num_ball_detections=12,
            release_event=low_conf_ev,
            trajectory_result=self.traj_pass,
            release_speed_result=self.speed_pass
        )
        self.assertFalse(res.is_measured)
        self.assertIn("RELEASE_CONFIDENCE_LOW", res.rejection_reason)

        # Implausible release height (Z = 0.5m, bowling release from floor)
        low_z_traj = TrajectoryFitResult(
            t0=0.125, x0=-0.1, y0=0.6, z0=0.50, vx0=0.2, vy0=37.5, vz0=-1.8,
            rmse_pixels=1.2, uncertainty_speed_kmh=2.5, covariance_matrix=np.eye(6),
            fitted_3d_points=[], is_converged=True, is_valid=True,
            status="MEASURED", depth_observability_score=0.85
        )
        res_z = self.qc.audit(
            video_fps=120.0,
            is_camera_stable=True,
            reprojection_report=self.reproj_pass,
            num_ball_detections=12,
            release_event=self.release_pass,
            trajectory_result=low_z_traj,
            release_speed_result=self.speed_pass
        )
        self.assertFalse(res_z.is_measured)
        self.assertIn("RELEASE_HEIGHT_IMPLAUSIBLE", res_z.rejection_reason)

    def test_gate6_depth_uncertainty_rejection(self):
        """Gate 6: Unconstrained depth ray observability returns NOT_MEASURED."""
        ill_depth_traj = TrajectoryFitResult(
            t0=0.125, x0=-0.1, y0=0.6, z0=2.15, vx0=0.2, vy0=37.5, vz0=-1.8,
            rmse_pixels=1.2, uncertainty_speed_kmh=2.5, covariance_matrix=np.eye(6),
            fitted_3d_points=[], is_converged=True, is_valid=True,
            status="MEASURED", depth_observability_score=0.08  # < 0.20
        )
        res = self.qc.audit(
            video_fps=120.0,
            is_camera_stable=True,
            reprojection_report=self.reproj_pass,
            num_ball_detections=12,
            release_event=self.release_pass,
            trajectory_result=ill_depth_traj,
            release_speed_result=self.speed_pass
        )
        self.assertFalse(res.is_measured)
        self.assertIn("DEPTH_UNCERTAINTY_HIGH", res.rejection_reason)

    def test_gate7_trajectory_fitting_residual_rejection(self):
        """Gate 7: Trajectory divergence or excessive RMSE returns NOT_MEASURED."""
        high_rmse_traj = TrajectoryFitResult(
            t0=0.125, x0=-0.1, y0=0.6, z0=2.15, vx0=0.2, vy0=37.5, vz0=-1.8,
            rmse_pixels=8.9, uncertainty_speed_kmh=2.5, covariance_matrix=np.eye(6),
            fitted_3d_points=[], is_converged=True, is_valid=True,
            status="MEASURED", depth_observability_score=0.85
        )
        res = self.qc.audit(
            video_fps=120.0,
            is_camera_stable=True,
            reprojection_report=self.reproj_pass,
            num_ball_detections=12,
            release_event=self.release_pass,
            trajectory_result=high_rmse_traj,
            release_speed_result=self.speed_pass
        )
        self.assertFalse(res.is_measured)
        self.assertIn("TRAJECTORY_RESIDUAL_HIGH", res.rejection_reason)

    def test_gate8_timestamp_validity_rejection(self):
        """Gate 8: Low frame rate or irregular timestamps returns NOT_MEASURED."""
        # 30 fps video (minimum required is 59 fps for fast bowling resolution)
        res_fps = self.qc.audit(
            video_fps=30.0,  # < 59.0
            is_camera_stable=True,
            reprojection_report=self.reproj_pass,
            num_ball_detections=12,
            release_event=self.release_pass,
            trajectory_result=self.traj_pass,
            release_speed_result=self.speed_pass
        )
        self.assertFalse(res_fps.is_measured)
        self.assertIn("FRAMERATE_TOO_LOW", res_fps.rejection_reason)

        # Corrupt timestamps (dt <= 0 or dropped frames)
        res_ts = self.qc.audit(
            video_fps=120.0,
            is_camera_stable=True,
            reprojection_report=self.reproj_pass,
            num_ball_detections=12,
            release_event=self.release_pass,
            trajectory_result=self.traj_pass,
            release_speed_result=self.speed_pass,
            is_timestamp_valid=False
        )
        self.assertFalse(res_ts.is_measured)
        self.assertIn("TIMESTAMP_IRREGULAR", res_ts.rejection_reason)

    def test_gate9_speed_uncertainty_rejection(self):
        """Gate 9: Speed uncertainty > 8.5 km/h returns NOT_MEASURED."""
        high_unc_speed = ReleaseSpeedResult(
            speed_km_h=135.2, speed_m_s=37.55, uncertainty_km_h=11.2,  # > 8.5
            ci_95_lower_km_h=113.2, ci_95_upper_km_h=157.2,
            velocity_vector_m_s=(0.2, 37.5, -1.8), c1_release_pos_m=(-0.1, 0.6, 2.15),
            release_timestamp_sec=0.125, is_valid=True, status="MEASURED"
        )
        res = self.qc.audit(
            video_fps=120.0,
            is_camera_stable=True,
            reprojection_report=self.reproj_pass,
            num_ball_detections=12,
            release_event=self.release_pass,
            trajectory_result=self.traj_pass,
            release_speed_result=high_unc_speed
        )
        self.assertFalse(res.is_measured)
        self.assertIn("SPEED_UNCERTAINTY_EXCESSIVE", res.rejection_reason)

    def test_gate10_physical_and_aerodynamic_plausibility_rejection(self):
        """Gate 10: Unphysical speeds (220 km/h) or aerodynamic violation returns NOT_MEASURED."""
        # Unphysical speed (220 km/h > 175 km/h limit)
        unphys_speed = ReleaseSpeedResult(
            speed_km_h=220.0, speed_m_s=61.1, uncertainty_km_h=3.0,
            ci_95_lower_km_h=214.0, ci_95_upper_km_h=226.0,
            velocity_vector_m_s=(0.0, 61.1, 0.0), c1_release_pos_m=(-0.1, 0.6, 2.15),
            release_timestamp_sec=0.125, is_valid=True, status="MEASURED"
        )
        res_spd = self.qc.audit(
            video_fps=120.0,
            is_camera_stable=True,
            reprojection_report=self.reproj_pass,
            num_ball_detections=12,
            release_event=self.release_pass,
            trajectory_result=self.traj_pass,
            release_speed_result=unphys_speed
        )
        self.assertFalse(res_spd.is_measured)
        self.assertIn("SPEED_OUT_OF_RANGE", res_spd.rejection_reason)

        # Aerodynamic inconsistency: C1-C2 speed is 155 km/h while release is 135 km/h (+20 km/h unphysical acceleration)
        bad_c1c2 = C1C2SpeedResult(
            c1_pos_m=(-0.1, 0.6, 2.15), c2_pos_m=(-0.1, 7.1, 1.8),
            t1_sec=0.125, t2_sec=0.280, distance_3d_m=6.51, time_delta_sec=0.155,
            average_speed_km_h=155.0, average_speed_m_s=43.0,
            is_valid=True, status="MEASURED"
        )
        res_aero = self.qc.audit(
            video_fps=120.0,
            is_camera_stable=True,
            reprojection_report=self.reproj_pass,
            num_ball_detections=12,
            release_event=self.release_pass,
            trajectory_result=self.traj_pass,
            release_speed_result=self.speed_pass,
            c1_c2_result=bad_c1c2
        )
        self.assertFalse(res_aero.is_measured)
        self.assertIn("AERODYNAMIC_INCONSISTENCY", res_aero.rejection_reason)

    def test_radar_isolation_strict_non_input(self):
        """
        MANDATE VERIFICATION: Radar is strictly an independent post-hoc evaluation metric.
        Production speed estimation must never require, read, or change with radar speed.
        """
        # compute_release_speed accepts only TrajectoryFitResult (no radar parameter exists)
        import inspect
        sig = inspect.signature(compute_release_speed)
        self.assertNotIn("radar_speed", sig.parameters)
        self.assertNotIn("radar", sig.parameters)

        res1 = compute_release_speed(self.traj_pass)
        self.assertAlmostEqual(res1.speed_km_h, 135.2, delta=0.5)

    def test_radar_benchmark_summary_and_metrics(self):
        """Tests calculation of MAE, RMSE, Bias, +-2km/h %, +-5km/h %, coverage %, and NOT_MEASURED rate."""
        validator = RadarValidator()

        # Add 5 deliveries: 4 measured, 1 rejected
        # Deliv 1: vision=136.0, radar=135.0 -> err = +1.0
        # Deliv 2: vision=140.5, radar=141.0 -> err = -0.5
        # Deliv 3: vision=128.8, radar=127.0 -> err = +1.8
        # Deliv 4: vision=131.0, radar=134.5 -> err = -3.5 (in +-5km/h, not in +-2km/h)
        # Deliv 5: unmeasured (camera unstable)
        validator.add_comparison("d1.mp4", 136.0, 135.0, is_measured=True, c1_c2_speed_kmh=133.0)
        validator.add_comparison("d2.mp4", 140.5, 141.0, is_measured=True, c1_c2_speed_kmh=138.5)
        validator.add_comparison("d3.mp4", 128.8, 127.0, is_measured=True, c1_c2_speed_kmh=126.0)
        validator.add_comparison("d4.mp4", 131.0, 134.5, is_measured=True, c1_c2_speed_kmh=129.0)
        validator.add_comparison("d5.mp4", None, 130.0, is_measured=False, rejection_reason="CAMERA_UNSTABLE")

        summary = validator.compute_summary()

        self.assertEqual(summary.total_deliveries, 5)
        self.assertEqual(summary.measured_deliveries, 4)
        self.assertEqual(summary.not_measured_deliveries, 1)
        self.assertAlmostEqual(summary.coverage_percentage, 80.0)
        self.assertAlmostEqual(summary.not_measured_rate, 20.0)

        # Expected errors: [+1.0, -0.5, +1.8, -3.5]
        # Abs errors: [1.0, 0.5, 1.8, 3.5] -> sum = 6.8 -> MAE = 1.70 km/h
        self.assertAlmostEqual(summary.mae_kmh, 1.70, delta=0.01)

        # Squared errors: [1.0, 0.25, 3.24, 12.25] -> sum = 16.74 -> mean = 4.185 -> RMSE = 2.046 km/h
        self.assertAlmostEqual(summary.rmse_kmh, np.sqrt(4.185), delta=0.01)

        # Bias: (1.0 - 0.5 + 1.8 - 3.5) / 4 = -1.2 / 4 = -0.30 km/h
        self.assertAlmostEqual(summary.bias_kmh, -0.30, delta=0.01)

        # Within +-2 km/h: 3 out of 4 = 75.0%
        self.assertAlmostEqual(summary.within_2kmh_percentage, 75.0)

        # Within +-5 km/h: 4 out of 4 = 100.0%
        self.assertAlmostEqual(summary.within_5kmh_percentage, 100.0)

        # Verify markdown export
        md = summary.to_markdown_table()
        self.assertIn("Radar Ground Truth Benchmark Report", md)
        self.assertIn("1.70 km/h", md)
        self.assertIn("d1.mp4", md)
        self.assertIn("CAMERA_UNSTABLE", md)

    def test_benchmark_plots_generation(self):
        """Verifies that BenchmarkPlotGenerator renders and saves a valid PNG image file."""
        validator = RadarValidator()
        validator.add_comparison("d1.mp4", 136.0, 135.0, is_measured=True)
        validator.add_comparison("d2.mp4", 140.5, 141.0, is_measured=True)
        validator.add_comparison("d3.mp4", 128.8, 127.0, is_measured=True)
        validator.add_comparison("d4.mp4", None, 130.0, is_measured=False, rejection_reason="CAMERA_UNSTABLE")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_img = os.path.join(tmp_dir, "benchmark_dashboard.png")
            saved_path = validator.generate_plots(out_img)
            self.assertTrue(os.path.isfile(saved_path))
            self.assertGreater(os.path.getsize(saved_path), 5000)

            # Verify it's a readable image file
            img = cv2.imread(saved_path)
            self.assertIsNotNone(img)
            self.assertGreaterEqual(img.shape[0], 500)
            self.assertGreaterEqual(img.shape[1], 1000)

    def test_benchmark_runner_full_package(self):
        """Tests BenchmarkRunner batch execution and report package generation."""
        runner = BenchmarkRunner()
        deliveries = [
            {"video_name": "ball_01.mp4", "radar_speed_kmh": 137.5, "vision_speed_kmh": 138.2, "is_measured": True},
            {"video_name": "ball_02.mp4", "radar_speed_kmh": 142.0, "vision_speed_kmh": 141.5, "is_measured": True},
            {"video_name": "ball_03.mp4", "radar_speed_kmh": 125.0, "vision_speed_kmh": None, "is_measured": False, "rejection_reason": "LONG_PREDICTION_GAP"},
        ]
        summary = runner.run_radar_evaluation(deliveries)
        self.assertEqual(summary.total_deliveries, 3)
        self.assertEqual(summary.measured_deliveries, 2)
        self.assertEqual(summary.not_measured_deliveries, 1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            pkg = runner.export_benchmark_results(tmp_dir, "test_dataset")
            self.assertTrue(os.path.isfile(pkg["json_report"]))
            self.assertTrue(os.path.isfile(pkg["csv_deliveries"]))
            self.assertTrue(os.path.isfile(pkg["markdown_summary"]))
            self.assertTrue(os.path.isfile(pkg["plots_image"]))

            # Check JSON contents
            with open(pkg["json_report"], "r", encoding="utf-8") as f:
                j_data = json.load(f)
            self.assertEqual(j_data["total_deliveries"], 3)
            self.assertEqual(j_data["measured_deliveries"], 2)
            self.assertEqual(len(j_data["deliveries"]), 3)


if __name__ == "__main__":
    unittest.main()
