"""
Video metadata extraction and encapsulation.
Handles resolution, frame rate, duration, codec, timestamps, and integrity hashing.
"""

from dataclasses import dataclass, field
import hashlib
import os
from typing import Dict, Any, Optional
import cv2


@dataclass
class VideoMetadata:
    """Represents all critical metadata extracted from a video source."""
    file_path: str
    file_name: str
    file_hash: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str
    is_vfr: bool = False
    rotation: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)

    def is_valid_framerate(self, min_fps: float = 59.0) -> bool:
        """Verifies if the video meets the minimum frame rate requirement."""
        return self.fps >= min_fps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_hash": self.file_hash,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "total_frames": self.total_frames,
            "duration_seconds": round(self.duration_seconds, 4),
            "codec": self.codec,
            "is_vfr": self.is_vfr,
            "rotation": self.rotation,
            "extra": self.extra,
        }


def compute_file_hash(path: str, chunk_size: int = 65536) -> str:
    """Computes a SHA-256 hash of the video file for locking calibrations."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_video_metadata(video_path: str) -> VideoMetadata:
    """
    Extracts video metadata using OpenCV with robust fallbacks.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        # Decode FOURCC
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
        if not codec or codec == "\x00\x00\x00\x00":
            codec = "UNKNOWN"

        # Check for 0 or corrupt fps
        if fps <= 0.0 or fps != fps: # NaN check
            fps = 30.0

        duration = total_frames / fps if fps > 0 and total_frames > 0 else 0.0
        file_hash = compute_file_hash(video_path)

        metadata = VideoMetadata(
            file_path=os.path.abspath(video_path),
            file_name=os.path.basename(video_path),
            file_hash=file_hash,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            codec=codec,
            is_vfr=False,
            rotation=0
        )
        return metadata
    finally:
        cap.release()
