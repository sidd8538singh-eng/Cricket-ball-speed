"""
Timestamp Manager: Tracks and validates actual frame presentation timestamps (PTS).
Detects variable frame rate (VFR), dropped frames, and timestamp jitter.
"""

from typing import List, Optional, Dict, Any
import numpy as np


class TimestampManager:
    """
    Manages presentation timestamps for all decoded video frames.
    Speed calculations require accurate, verified delta-time measurements.
    """

    def __init__(self, nominal_fps: float):
        self.nominal_fps = nominal_fps if nominal_fps > 0 else 60.0
        self.nominal_dt = 1.0 / self.nominal_fps
        self._timestamps: List[float] = [] # timestamps in seconds
        self._frame_indices: List[int] = []
        self._is_vfr: bool = False
        self._jitter_std: float = 0.0

    def add_timestamp(self, frame_idx: int, pts_msec: Optional[float] = None) -> float:
        """
        Records the timestamp for a given frame index.
        If PTS is missing, corrupt, or non-monotonic, falls back to nominal frame spacing.
        """
        if pts_msec is not None and pts_msec > 0:
            ts_sec = pts_msec / 1000.0
            # Guard against corrupt or non-monotonic hardware timestamps
            if self._timestamps and ts_sec <= self._timestamps[-1]:
                ts_sec = self._timestamps[-1] + self.nominal_dt
        else:
            ts_sec = frame_idx * self.nominal_dt

        self._frame_indices.append(frame_idx)
        self._timestamps.append(ts_sec)
        return ts_sec

    def set_all_timestamps(self, timestamps: List[float]):
        """Sets timestamps directly (e.g. from container demuxer or pre-scanned PTS)."""
        self._timestamps = list(timestamps)
        self._frame_indices = list(range(len(timestamps)))
        self._analyze_stability()

    def get_timestamp(self, frame_idx: int) -> float:
        """Retrieves the timestamp in seconds for a specific frame index."""
        if 0 <= frame_idx < len(self._timestamps):
            return self._timestamps[frame_idx]
        return frame_idx * self.nominal_dt

    def get_delta_time(self, frame_start: int, frame_end: int) -> float:
        """
        Computes accurate elapsed time between two frames.
        Returns positive duration in seconds.
        """
        t_start = self.get_timestamp(frame_start)
        t_end = self.get_timestamp(frame_end)
        dt = abs(t_end - t_start)
        if dt <= 1e-6:
            # Fallback to nominal delta if timestamps coincide
            dt = abs(frame_end - frame_start) * self.nominal_dt
        return dt

    def _analyze_stability(self):
        """Analyzes timestamp intervals for variable frame rate or dropped frames."""
        if len(self._timestamps) < 3:
            return
        diffs = np.diff(self._timestamps)
        mean_dt = float(np.mean(diffs))
        std_dt = float(np.std(diffs))
        self._jitter_std = std_dt
        # If standard deviation of inter-frame spacing is > 10% of mean, flag as VFR
        self._is_vfr = (std_dt / max(mean_dt, 1e-6)) > 0.10

    @property
    def is_vfr(self) -> bool:
        return self._is_vfr

    @property
    def jitter_std(self) -> float:
        return self._jitter_std

    def summary(self) -> Dict[str, Any]:
        return {
            "total_timestamps": len(self._timestamps),
            "nominal_fps": round(self.nominal_fps, 2),
            "nominal_dt_sec": round(self.nominal_dt, 5),
            "is_vfr": self._is_vfr,
            "jitter_std_sec": round(self._jitter_std, 6),
        }
