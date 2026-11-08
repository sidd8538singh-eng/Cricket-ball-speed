"""
Video Loader: High-level orchestrator for loading, validating, and streaming video.
Integrates metadata extraction, presentation timestamps, and camera stability.
"""

import os
from typing import Optional, Dict, Any, Generator, Tuple
import cv2
import numpy as np

from .metadata import VideoMetadata, extract_video_metadata
from .timestamp_manager import TimestampManager
from .frame_decoder import FrameDecoder
from .camera_stability import CameraStabilityAnalyzer


class VideoLoader:
    """
    Manages complete lifecycle of a video input for cricket speed measurement.
    Validates minimum frame rate and stability standards.
    """

    def __init__(
        self,
        video_path: str,
        min_fps: float = 59.0,
        target_fps: float = 120.0,
        check_stability: bool = True,
        max_displacement_px: float = 3.5,
    ):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        self.video_path = video_path
        self.metadata = extract_video_metadata(video_path)
        self.min_fps = min_fps
        self.target_fps = target_fps
        self.check_stability = check_stability

        self.ts_manager = TimestampManager(self.metadata.fps)
        self.decoder = FrameDecoder(video_path, self.ts_manager)
        self.stability_analyzer = (
            CameraStabilityAnalyzer(max_displacement_px=max_displacement_px)
            if check_stability
            else None
        )

        self._fps_warning: Optional[str] = None
        self._validate_framerate()

    def _validate_framerate(self):
        """Enforces frame rate requirements."""
        if self.metadata.fps < self.min_fps:
            self._fps_warning = (
                f"Video FPS ({self.metadata.fps:.1f}) is below minimum target ({self.min_fps:.1f} FPS). "
                f"Ball tracking and speed precision may be degraded."
            )

    @property
    def fps_warning(self) -> Optional[str]:
        return self._fps_warning

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Fetches a specific video frame by index."""
        return self.decoder.read_frame(frame_idx)

    def iter_frames(
        self, start_frame: int = 0, end_frame: Optional[int] = None
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Yields (frame_idx, timestamp_seconds, frame_bgr) while updating stability metrics.
        """
        for frame_idx, ts, frame in self.decoder.iter_frames(start_frame, end_frame):
            if self.stability_analyzer is not None:
                self.stability_analyzer.process_frame(frame_idx, frame)
            yield frame_idx, ts, frame

    def get_stability_report(self) -> Dict[str, Any]:
        """Returns the stability report from the analyzer."""
        if self.stability_analyzer is None:
            return {"stability_checked": False, "is_stable": True}
        rep = self.stability_analyzer.get_report()
        rep["stability_checked"] = True
        return rep

    def close(self):
        """Releases all open video resources."""
        self.decoder.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
