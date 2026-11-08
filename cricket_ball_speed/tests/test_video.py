"""
Unit tests for video loading, metadata extraction, timestamp management, and camera stability.
"""

import unittest
import numpy as np
import tempfile
import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from video.metadata import VideoMetadata, compute_file_hash, extract_video_metadata
from video.timestamp_manager import TimestampManager
from video.camera_stability import CameraStabilityAnalyzer
from video.frame_decoder import FrameDecoder
from video.video_loader import VideoLoader


def create_synthetic_video(
    file_path: str,
    num_frames: int = 30,
    fps: float = 60.0,
    width: int = 640,
    height: int = 480,
    add_jitter: bool = False
) -> str:
    """Creates a temporary synthetic test video file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background static grid
        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width, y), (50, 50, 50), 1)
        for x in range(0, width, 40):
            cv2.line(frame, (x, 0), (x, height), (50, 50, 50), 1)

        # Static corner features
        cv2.circle(frame, (50, 50), 6, (200, 200, 200), -1)
        cv2.circle(frame, (width - 50, 50), 6, (200, 200, 200), -1)
        cv2.circle(frame, (50, height - 50), 6, (200, 200, 200), -1)
        cv2.circle(frame, (width - 50, height - 50), 6, (200, 200, 200), -1)

        if add_jitter and i > 15:
            # Shake camera by shifting pixels
            M = np.float32([[1, 0, 15], [0, 1, 15]])
            frame = cv2.warpAffine(frame, M, (width, height))

        writer.write(frame)

    writer.release()
    return file_path


class TestVideo(unittest.TestCase):
    def test_timestamp_manager_monotonic_and_delta(self):
        """Verifies TimestampManager provides accurate delta time and monotonic progression."""
        tm = TimestampManager(nominal_fps=120.0)
        self.assertAlmostEqual(tm.nominal_dt, 1.0 / 120.0, places=5)

        # Add sequential PTS
        t0 = tm.add_timestamp(0, pts_msec=0.0)
        t1 = tm.add_timestamp(1, pts_msec=8.333)
        t2 = tm.add_timestamp(2, pts_msec=16.666)

        self.assertAlmostEqual(t0, 0.0, places=3)
        self.assertAlmostEqual(t1, 0.00833, places=3)
        self.assertAlmostEqual(t2, 0.01666, places=3)

        dt = tm.get_delta_time(0, 2)
        self.assertAlmostEqual(dt, 0.01666, places=3)

    def test_camera_stability_analyzer_stationary_vs_jitter(self):
        """Tests optical flow detection of stationary background vs camera displacement."""
        analyzer_stable = CameraStabilityAnalyzer(max_displacement_px=3.5, sample_stride=1)
        analyzer_shaky = CameraStabilityAnalyzer(max_displacement_px=3.5, sample_stride=1)

        # Create stationary frame
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add rich corners
        for x in range(50, 600, 60):
            for y in range(50, 440, 60):
                cv2.circle(frame1, (x, y), 5, (255, 255, 255), -1)

        # Process frame 0
        analyzer_stable.process_frame(0, frame1)
        analyzer_shaky.process_frame(0, frame1)

        # Frame 1: Identical stationary frame
        stable, disp_stable = analyzer_stable.process_frame(1, frame1)
        self.assertTrue(stable)
        self.assertLess(disp_stable, 1.0)

        # Frame 2: Shaken frame (displaced by 10 pixels)
        M_shake = np.float32([[1, 0, 10], [0, 1, 8]])
        frame_shaken = cv2.warpAffine(frame1, M_shake, (640, 480))
        shaky, disp_shaky = analyzer_shaky.process_frame(1, frame_shaken)

        self.assertFalse(shaky)
        self.assertGreater(disp_shaky, 3.5)

    def test_video_loader_and_metadata(self):
        """Tests complete VideoLoader, metadata extraction, and frame decoder on synthetic video."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            vid_path = os.path.join(tmp_dir, "synth_test.mp4")
            create_synthetic_video(vid_path, num_frames=20, fps=60.0, width=640, height=480)

            loader = VideoLoader(vid_path, min_fps=59.0)
            self.assertEqual(loader.metadata.width, 640)
            self.assertEqual(loader.metadata.height, 480)
            self.assertGreaterEqual(loader.metadata.fps, 59.0)
            self.assertIsNone(loader.fps_warning)

            # Iteration test
            frames = list(loader.iter_frames())
            self.assertEqual(len(frames), 20)
            self.assertEqual(frames[0][0], 0)
            self.assertEqual(frames[-1][0], 19)

            # Stability report
            rep = loader.get_stability_report()
            self.assertTrue(rep["is_stable"])

            loader.close()


if __name__ == "__main__":
    unittest.main()
