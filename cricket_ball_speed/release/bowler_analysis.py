"""
Bowler Analysis: Analyzes the bowler's run-up, delivery stride, and release window.
Determines the temporal search window where release physically occurs.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from detection.bowler_detector import BowlerObservation


class BowlerAnalyzer:
    """
    Tracks bowler progression down the pitch.
    Identifies the delivery stride (where the bowler plants front foot and reaches release apex).
    """

    def __init__(self):
        self.history: List[BowlerObservation] = []

    def add_observation(self, obs: BowlerObservation):
        """Adds a bowler observation to history."""
        self.history.append(obs)

    def find_delivery_window(self) -> Tuple[int, int]:
        """
        Returns (start_frame_idx, end_frame_idx) where bowling release occurs.
        The bowler approaches closest to the camera/crease right before and during release,
        where bounding box area and bottom y-coordinate reach their maximums.
        """
        if not self.history:
            return 0, 999999

        # Filter sorted by frame
        sorted_obs = sorted(self.history, key=lambda o: o.frame_index)

        # Find frame where bowler bounding box area reaches maximum (closest approach to crease)
        areas = [(obs.width * obs.height, obs.frame_index) for obs in sorted_obs]
        max_area_obs = max(areas, key=lambda a: a[0])
        max_area_frame = max_area_obs[1]

        # Delivery release typically happens within [-15, +25] frames of maximum size
        start_frame = max(0, max_area_frame - 15)
        end_frame = max_area_frame + 25
        return start_frame, end_frame

    def get_runup_velocity_px_s(self) -> float:
        """Estimates forward run-up velocity of the bowler in pixels per second."""
        if len(self.history) < 3:
            return 0.0

        sorted_obs = sorted(self.history, key=lambda o: o.frame_index)
        first_obs = sorted_obs[0]
        last_obs = sorted_obs[-1]
        dt = last_obs.timestamp_sec - first_obs.timestamp_sec
        if dt < 1e-4:
            return 0.0

        dy = last_obs.center[1] - first_obs.center[1]
        return float(abs(dy) / dt)
