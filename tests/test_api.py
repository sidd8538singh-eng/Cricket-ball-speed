"""
Integration tests for FastAPI endpoints.
Tests status, health, detection, calibration, video endpoints, and sample video listing.
"""

import unittest
import numpy as np
import cv2
import json
import os
import sys
import tempfile

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi.testclient import TestClient
from backend.app import app


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Create a small valid test video for video endpoint testing
        cls.test_video_dir = tempfile.mkdtemp()
        cls.test_video_path = os.path.join(cls.test_video_dir, "test_api_delivery.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(cls.test_video_path, fourcc, 30.0, (320, 240))
        for i in range(10):
            frame = np.full((240, 320, 3), fill_value=i * 10, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    @classmethod
    def tearDownClass(cls):
        if os.path.isfile(cls.test_video_path):
            try:
                os.remove(cls.test_video_path)
            except Exception:
                pass
        if os.path.isdir(cls.test_video_dir):
            try:
                os.rmdir(cls.test_video_dir)
            except Exception:
                pass

    def test_health_endpoint(self):
        """Tests /health and /api/health endpoints for container readiness."""
        for endpoint in ["/health", "/api/health"]:
            res = self.client.get(endpoint)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["status"], "healthy")
            self.assertEqual(data["service"], "cricket-ball-speed")

    def test_status_endpoint(self):
        """Tests /api/status returns online, model details, and physical dimensions."""
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("models", data)
        self.assertIn("stumps", data["models"])
        self.assertEqual(data["physical_dimensions"]["wicket_distance_m"], 20.12)
        self.assertEqual(data["physical_dimensions"]["pitch_width_m"], 3.048)
        self.assertEqual(data["physical_dimensions"]["stump_height_m"], 0.711)

    def test_detect_frame_endpoint(self):
        """Tests /api/detect/frame with synthetic image."""
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        _, img_buf = cv2.imencode(".jpg", img)

        files = {"file": ("test.jpg", img_buf.tobytes(), "image/jpeg")}
        data = {"confidence": "0.15", "frame_index": "0"}
        res = self.client.post("/api/detect/frame", files=files, data=data)
        self.assertEqual(res.status_code, 200)
        json_data = res.json()
        self.assertIn("wickets", json_data)
        self.assertIn("status", json_data)
        self.assertEqual(json_data["image_width"], 1280)
        self.assertEqual(json_data["image_height"], 720)

    def test_calibrate_endpoint_4points(self):
        """Tests /api/calibrate with 4 stump landmarks."""
        payload = {
            "image_points": {
                "near_base": [640.0, 680.0],
                "near_top": [640.0, 520.0],
                "far_base": [640.0, 420.0],
                "far_top": [640.0, 400.0]
            },
            "image_width": 1280,
            "image_height": 720,
            "initial_focal_length_px": 1200.0
        }
        res = self.client.post("/api/calibrate", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["method"], "Stump + DRS Calibration")
        self.assertIn("camera_parameters", data)
        self.assertIn("homography_w2i", data["camera_parameters"])

    def test_sample_videos_listing(self):
        """Tests discovery of available sample videos."""
        res = self.client.get("/api/sample_videos")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("sample_videos", data)
        self.assertIsInstance(data["sample_videos"], list)

    def test_camera_list_and_status(self):
        """Tests /api/camera/list and /api/camera/status."""
        res = self.client.get("/api/camera/list")
        self.assertEqual(res.status_code, 200)
        self.assertIn("cameras", res.json())

        res_status = self.client.get("/api/camera/status")
        self.assertEqual(res_status.status_code, 200)
        cdata = res_status.json()
        self.assertIn("is_running", cdata)
        self.assertIn("is_stable", cdata)

    def test_calibration_lifecycle(self):
        """Tests calibration retrieval, confirmation, and reset."""
        # 1. Reset
        res = self.client.post("/api/calibration/reset")
        self.assertEqual(res.status_code, 200)

        # 2. Check current is uncalibrated
        res = self.client.get("/api/calibration/current")
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["calibrated"])

        # 3. Perform calibration
        payload = {
            "image_points": {
                "near_base": [640.0, 680.0],
                "near_top": [640.0, 520.0],
                "far_base": [640.0, 420.0],
                "far_top": [640.0, 400.0]
            },
            "image_width": 1280,
            "image_height": 720
        }
        res = self.client.post("/api/calibrate", json=payload)
        self.assertEqual(res.status_code, 200)

        # 4. Check current is calibrated
        res = self.client.get("/api/calibration/current")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["calibrated"])

        # 5. Confirm calibration
        res = self.client.post("/api/calibration/confirm", json={"test": "ok"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "confirmed")

    def test_video_metadata_endpoint(self):
        """Tests retrieving metadata for a video file."""
        res = self.client.get(f"/api/video/metadata?video_path={self.test_video_path}")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["width"], 320)
        self.assertEqual(data["height"], 240)
        self.assertEqual(data["total_frames"], 10)
        self.assertIn("file_hash", data)

    def test_video_upload_endpoint(self):
        """Tests uploading video file via multipart stream."""
        with open(self.test_video_path, "rb") as f:
            files = {"file": ("uploaded_test.mp4", f.read(), "video/mp4")}
            res = self.client.post("/api/video/upload", files=files)
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertIn("file_name", data)
            self.assertIn("file_path", data)
            self.assertIn("metadata", data)

    def test_frontend_static_serving(self):
        """Tests that root HTML and static CSS/JS assets are served successfully."""
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertIn("PRO CORPORATE LEAGUE", res_root.text)
        self.assertIn("CRICKET BOWLING", res_root.text)

        res_css = self.client.get("/static/style.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertIn(".measurement-pill", res_css.text)
        self.assertIn(".progress-card", res_css.text)

        res_js = self.client.get("/static/app.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn("state", res_js.text)

    def test_video_best_calibration_frame(self):
        """Tests selecting best calibration frame on video."""
        res = self.client.post("/api/video/best_calibration_frame", json={"video_path": self.test_video_path})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("best_frame_index", data)


if __name__ == "__main__":
    unittest.main()
