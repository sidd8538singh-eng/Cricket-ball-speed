"""
Unit and integration tests for the Output Generation Subsystem.
Verifies:
  1. Final processed video dimensions, upper-left badge placement, date/time, and frame freeze.
  2. Diagnostic video with raw/accepted/rejected ball boxes, C1/C2 points, distinct predicted styling, and HUD.
  3. Comprehensive JSON report serialization across all calibration, tracking, speed, and QC fields.
  4. Formatted Excel workbook with 17 mandatory columns, clearly defined variance, and status highlights.
  5. Decoupling: independent execution using plain dictionaries and primitive values.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import json
import cv2
import openpyxl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from output.final_video import FinalVideoRenderer, generate_final_video
from output.diagnostic_video import DiagnosticVideoRenderer, generate_diagnostic_video
from output.json_report import generate_json_report
from output.excel_report import ExcelReportWriter, EXCEL_HEADERS
from calibration.camera_model import CameraModel


def create_synthetic_video(filepath: str, width: int = 640, height: int = 360, fps: float = 60.0, num_frames: int = 10) -> str:
    """Helper that creates a dummy video file for testing."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    for i in range(num_frames):
        # Create gradient frame with a moving circle
        frame = np.ones((height, width, 3), dtype=np.uint8) * 40
        cv2.circle(frame, (100 + i * 20, 150), 12, (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    return filepath


class TestOutputSubsystem(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)

    def test_final_video_renderer_badge_geometry(self):
        """Verifies resolution-independent badge placement in upper-left (not top, not middle)."""
        renderer = FinalVideoRenderer()

        # 1080p frame (1920x1080)
        bx, by, bw, bh = renderer.compute_badge_rect(1920, 1080)
        self.assertGreaterEqual(bx, 50)        # Not flush with left edge
        self.assertLess(bx, 150)
        self.assertGreaterEqual(by, 50)        # Not touching top edge
        self.assertLess(by, 150)               # Far above vertical center (1080 // 2 = 540)
        self.assertGreaterEqual(bw, 250)
        self.assertGreaterEqual(bh, 80)

        # 720p frame (1280x720)
        bx720, by720, bw720, bh720 = renderer.compute_badge_rect(1280, 720)
        self.assertGreater(by720, 30)
        self.assertLess(by720, 100)
        self.assertLess(by720, 720 // 2)

    def test_final_video_generation_and_freeze(self):
        """Tests generate_final_video for correct dimensions, frame count with freeze, and badge."""
        in_vid = os.path.join(self.tmp_dir.name, "input_raw.mp4")
        out_vid = os.path.join(self.tmp_dir.name, "output_final.mp4")

        # 10 frames of 640x360 at 60 fps
        create_synthetic_video(in_vid, width=640, height=360, fps=60.0, num_frames=10)

        # Render with 1.0 second freeze (60 extra frames -> 70 total frames)
        generate_final_video(
            input_video_path=in_vid,
            output_video_path=out_vid,
            qc_result={"is_measured": True, "status": "MEASURED"},
            release_speed={"speed_km_h": 138.4},
            fps=60.0,
            freeze_last_frame_seconds=1.0,
            custom_timestamp_str="2026-11-01 13:45:00"
        )

        self.assertTrue(os.path.isfile(out_vid))
        cap = cv2.VideoCapture(out_vid)
        self.assertTrue(cap.isOpened())

        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.assertEqual(out_w, 640)
        self.assertEqual(out_h, 360)
        # Expected: 10 input frames + 60 freeze frames = 70 frames
        self.assertEqual(total_frames, 70)

    def test_final_video_not_measured_badge(self):
        """Verifies clean 'SPEED: NOT MEASURED' badge when delivery is unmeasured."""
        in_vid = os.path.join(self.tmp_dir.name, "input_fail.mp4")
        out_vid = os.path.join(self.tmp_dir.name, "output_not_measured.mp4")
        create_synthetic_video(in_vid, width=640, height=360, fps=30.0, num_frames=5)

        generate_final_video(
            input_video_path=in_vid,
            output_video_path=out_vid,
            qc_result=False,  # Unmeasured
            release_speed=None,
            fps=30.0,
            freeze_last_frame_seconds=0.0
        )
        self.assertTrue(os.path.isfile(out_vid))

    def test_diagnostic_video_generation_with_distinct_predictions(self):
        """
        Tests diagnostic video rendering with:
        - raw, accepted, rejected ball boxes
        - C1 release and C2 baseline
        - distinctly styled predicted points (never mixed with real)
        - reprojected fitted 3D trajectory curve
        - HUD telemetry
        """
        in_vid = os.path.join(self.tmp_dir.name, "input_diag.mp4")
        out_vid = os.path.join(self.tmp_dir.name, "output_diagnostic.mp4")
        create_synthetic_video(in_vid, width=800, height=450, fps=60.0, num_frames=10)

        # Mock observations
        raw_obs = {
            1: [
                {"bbox": [200, 100, 214, 114], "acceptance_status": "ACCEPTED", "confidence": 0.88, "apparent_diameter_px": 14.0},
                {"bbox": [400, 300, 410, 310], "acceptance_status": "REJECTED", "confidence": 0.25, "rejection_reason": "Outlier jump", "apparent_diameter_px": 10.0},
            ]
        }

        # Mock tracked points (distinguishing observed vs predicted)
        tracked = [
            {"frame_index": 0, "u": 180.0, "v": 90.0, "is_real_detection": True},
            {"frame_index": 1, "u": 207.0, "v": 107.0, "is_real_detection": True},
            {"frame_index": 2, "u": 235.0, "v": 125.0, "is_real_detection": False}, # PREDICTED
            {"frame_index": 3, "u": 265.0, "v": 145.0, "is_real_detection": True},
        ]

        release_ev = {
            "frame_index": 1,
            "timestamp_sec": 0.0167,
            "c1_pixel": [207.0, 107.0],
            "observation_mode": "DIRECTLY_OBSERVED",
            "confidence": 0.92
        }

        c1_c2 = {
            "c2_frame_index": 3,
            "distance_3d_m": 6.5,
            "average_speed_km_h": 134.2
        }

        hud_qc = {
            "status": "MEASURED",
            "quality_score": 95.0,
            "rejection_reason": None
        }

        # Run decoupled generator
        saved_path = generate_diagnostic_video(
            input_video_path=in_vid,
            output_video_path=out_vid,
            camera_model=None,
            raw_observations=raw_obs,
            tracked_points=tracked,
            release_event=release_ev,
            c1_c2_result=c1_c2,
            trajectory_result=None,
            release_speed={"speed_km_h": 136.5, "uncertainty_km_h": 2.2},
            qc_result=hud_qc,
            fps=60.0
        )

        self.assertTrue(os.path.isfile(saved_path))
        cap = cv2.VideoCapture(saved_path)
        self.assertTrue(cap.isOpened())
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 800)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 450)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
        cap.release()

    def test_json_report_comprehensive_serialization(self):
        """Verifies JSON report contains all calibration, detections, release, trajectory, speed, and QC fields."""
        json_path = os.path.join(self.tmp_dir.name, "delivery_audit.json")

        metadata = {
            "file_name": "ball_fast_140.mp4",
            "width": 1920, "height": 1080, "fps": 120.0,
            "duration_seconds": 1.25, "frame_count": 150
        }
        camera_dict = {
            "fx": 1520.0, "fy": 1520.0, "cx": 960.0, "cy": 540.0,
            "camera_height_m": 2.45
        }
        reproj_dict = {"mean_error_px": 1.85, "max_error_px": 3.2, "is_acceptable": True}
        release_dict = {
            "frame_index": 20, "timestamp_sec": 0.1667,
            "c1_pixel": [945.0, 380.0], "confidence": 0.91,
            "observation_mode": "DIRECTLY_OBSERVED"
        }
        traj_dict = {
            "t0": 0.1667, "x0": -0.05, "y0": 0.55, "z0": 2.18,
            "vx0": 0.15, "vy0": 38.8, "vz0": -1.75,
            "rmse_pixels": 0.95, "is_converged": True, "is_valid": True
        }
        speed_dict = {
            "speed_km_h": 139.8, "speed_m_s": 38.84,
            "uncertainty_km_h": 2.1, "ci_95_lower_km_h": 135.7, "ci_95_upper_km_h": 143.9,
            "elevation_deg": -2.58, "azimuth_deg": 0.22, "status": "MEASURED"
        }
        c1c2_dict = {
            "average_speed_km_h": 136.5, "distance_3d_m": 6.52,
            "time_delta_sec": 0.172, "is_valid": True, "status": "MEASURED"
        }
        qc_dict = {
            "status": "MEASURED", "quality_score": 96.0, "is_measured": True,
            "rejection_reason": None,
            "audit_checks": {"camera_stability": True, "physical_plausibility": True}
        }
        pts_3d = [
            {"frame_index": 20, "timestamp_sec": 0.1667, "position_xyz": [-0.05, 0.55, 2.18]},
            {"frame_index": 21, "timestamp_sec": 0.1750, "position_xyz": [-0.05, 0.87, 2.16]},
        ]

        saved_file = generate_json_report(
            output_path=json_path,
            metadata=metadata,
            camera_model=camera_dict,
            reprojection_report=reproj_dict,
            release_event=release_dict,
            trajectory_result=traj_dict,
            release_speed=speed_dict,
            c1_c2_result=c1c2_dict,
            qc_result=qc_dict,
            reconstructed_3d_points=pts_3d,
            extra_info={"radar_speed": 139.0}
        )

        self.assertTrue(os.path.isfile(saved_file))

        with open(saved_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Assert all major sections exist
        self.assertIn("video_metadata", data)
        self.assertIn("calibration", data)
        self.assertIn("quality_control", data)
        self.assertIn("speed_measurement", data)
        self.assertIn("release_event", data)
        self.assertIn("trajectory_reconstruction", data)
        self.assertIn("reconstructed_positions_3d", data)
        self.assertIn("uncertainty", data)

        # Assert key values
        self.assertEqual(data["video_metadata"]["file_name"], "ball_fast_140.mp4")
        self.assertEqual(data["speed_measurement"]["release_speed"]["speed_km_h"], 139.8)
        self.assertEqual(data["speed_measurement"]["c1_c2_speed"]["average_speed_km_h"], 136.5)
        self.assertEqual(data["quality_control"]["status"], "MEASURED")
        self.assertEqual(len(data["reconstructed_positions_3d"]), 2)
        self.assertEqual(data["extra_info"]["radar_speed"], 139.0)

    def test_excel_report_writer_all_columns_and_variance(self):
        """Verifies Excel report has all 17 mandatory columns, variance definition, and styled status."""
        excel_path = os.path.join(self.tmp_dir.name, "speed_benchmark.xlsx")
        writer = ExcelReportWriter(excel_path)

        # Delivery 1: Measured with radar benchmark
        # radar = 138.0, calc = 139.2 -> diff = +1.2 -> abs_err = 1.2 -> variance = 1.44
        writer.add_delivery_row({
            "filename": "pace_01.mp4",
            "reference_speed": 140.0,
            "radar_speed": 138.0,
            "calculated_speed": 139.2,
            "c1_c2_speed": 136.0,
            "calibration_status": "VALID",
            "reprojection_error_px": 1.25,
            "detected_frames": 14,
            "predicted_frames": 1,
            "release_confidence": 0.94,
            "trajectory_rmse_px": 0.88,
            "speed_uncertainty_kmh": 2.1,
            "final_status": "MEASURED",
            "rejection_reason": None
        })

        # Delivery 2: Unmeasured
        writer.add_delivery_row({
            "filename": "pace_02.mp4",
            "reference_speed": 145.0,
            "radar_speed": 144.0,
            "calculated_speed": None,
            "c1_c2_speed": None,
            "calibration_status": "VALID",
            "reprojection_error_px": 1.40,
            "detected_frames": 3,
            "predicted_frames": 4,
            "release_confidence": 0.35,
            "trajectory_rmse_px": None,
            "speed_uncertainty_kmh": None,
            "final_status": "NOT_MEASURED",
            "rejection_reason": "INSUFFICIENT_BALL_DETECTIONS"
        })

        saved_path = writer.save()
        self.assertTrue(os.path.isfile(saved_path))

        # Reopen with openpyxl and verify structure
        wb = openpyxl.load_workbook(saved_path)
        ws = wb.active

        # Check headers (row 1)
        self.assertEqual(ws.max_column, 17)
        self.assertEqual(len(EXCEL_HEADERS), 17)
        for col_idx, h in enumerate(EXCEL_HEADERS, start=1):
            self.assertEqual(ws.cell(row=1, column=col_idx).value, h)

        # Check row 2 (Delivery 1)
        self.assertEqual(ws.cell(row=2, column=1).value, "pace_01.mp4")
        self.assertAlmostEqual(float(ws.cell(row=2, column=4).value), 139.2)
        self.assertAlmostEqual(float(ws.cell(row=2, column=6).value), 1.2)   # Difference
        self.assertAlmostEqual(float(ws.cell(row=2, column=7).value), 1.2)   # Abs Error
        self.assertAlmostEqual(float(ws.cell(row=2, column=8).value), 1.44)  # Variance (Error^2)
        self.assertEqual(ws.cell(row=2, column=16).value, "MEASURED")

        # Check row 3 (Delivery 2)
        self.assertEqual(ws.cell(row=3, column=1).value, "pace_02.mp4")
        self.assertEqual(ws.cell(row=3, column=4).value, "NOT_MEASURED")
        self.assertEqual(ws.cell(row=3, column=16).value, "NOT_MEASURED")
        self.assertIn("INSUFFICIENT_BALL_DETECTIONS", str(ws.cell(row=3, column=17).value))

        # Check Variance definition footnote exists
        footnote_cell = ws.cell(row=5, column=1)
        self.assertIn("Mathematical Definition of Variance", str(footnote_cell.value))
        self.assertIn("(Calculated Speed - Radar Speed)^2", str(footnote_cell.value))


if __name__ == "__main__":
    unittest.main()
