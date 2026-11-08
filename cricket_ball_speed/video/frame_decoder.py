"""
Frame Decoder: Robust sequential and random-access video frame decoding.
Preserves original resolution, extracts timestamps, and maintains frame integrity.
"""

from typing import Generator, Tuple, Optional
import os
import cv2
import numpy as np
from .timestamp_manager import TimestampManager


class FrameDecoder:
    """
    Decodes frames from a video file sequentially or by seeking.
    Ensures accurate frame indices and timestamp tracking.
    """

    def __init__(self, video_path: str, timestamp_manager: Optional[TimestampManager] = None):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
        self.video_path = video_path
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or np.isnan(fps):
            fps = 60.0
        self.fps = fps
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ts_manager = timestamp_manager or TimestampManager(self.fps)

    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Reads a specific frame by index (0-indexed)."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        return frame

    def iter_frames(self, start_frame: int = 0, end_frame: Optional[int] = None) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Yields (frame_idx, timestamp_seconds, frame_bgr) sequentially.
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_idx = start_frame
        max_idx = end_frame if end_frame is not None else self.total_frames

        while current_idx < max_idx:
            pts_msec = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            ret, frame = self._cap.read()
            if not ret or frame is None:
                break
            
            ts_sec = self.ts_manager.add_timestamp(current_idx, pts_msec)
            yield current_idx, ts_sec, frame
            current_idx += 1

    def release(self):
        """Releases the video capture resource."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
