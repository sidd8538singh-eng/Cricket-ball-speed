"""
Point Selector: Manages user-selected 2D calibration points.
Ensures points are recorded in original video pixel coordinates,
supports undo, reset, per-point replacement, duplicate rejection, and bounds validation.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from .world_coordinates import CALIBRATION_POINT_KEYS


POINT_DESCRIPTIONS = {
    "near_top": "1/8: Near Wicket Top Centre",
    "near_base": "2/8: Near Wicket Base Centre (Ground)",
    "far_top": "3/8: Far Wicket Top Centre",
    "far_base": "4/8: Far Wicket Base Centre (Ground)",
    "pitch_left_near": "5/8: Pitch Near-Left Corner",
    "pitch_right_near": "6/8: Pitch Near-Right Corner",
    "pitch_left_far": "7/8: Pitch Far-Left Corner",
    "pitch_right_far": "8/8: Pitch Far-Right Corner",
}

POINT_SHORT_NAMES = {
    "near_top": "Near Top",
    "near_base": "Near Base",
    "far_top": "Far Top",
    "far_base": "Far Base",
    "pitch_left_near": "Pitch L-Near",
    "pitch_right_near": "Pitch R-Near",
    "pitch_left_far": "Pitch L-Far",
    "pitch_right_far": "Pitch R-Far",
}

POINT_COLORS = {
    "near_top": (0, 200, 255),       # Orange/Yellow
    "near_base": (0, 200, 255),      # Orange/Yellow
    "far_top": (255, 200, 0),        # Cyan/Blue
    "far_base": (255, 200, 0),       # Cyan/Blue
    "pitch_left_near": (0, 255, 0),   # Green
    "pitch_right_near": (0, 255, 0),  # Green
    "pitch_left_far": (0, 255, 128),  # Light Green
    "pitch_right_far": (0, 255, 128), # Light Green
}


class CalibrationPointManager:
    """
    Tracks the 8 calibration landmarks selected by the user.
    Maintains strict ordering, bounds checking, duplicate prevention, and undo/reset.
    """

    def __init__(self, image_width: Optional[int] = None, image_height: Optional[int] = None):
        self.orig_w = image_width
        self.orig_h = image_height
        self.points: Dict[str, Optional[Tuple[float, float]]] = {
            k: None for k in CALIBRATION_POINT_KEYS
        }
        self.history: List[str] = [] # stores key of point added/modified

    @property
    def next_required_point_key(self) -> Optional[str]:
        """Returns the key of the next unassigned calibration point, or None if all are set."""
        for key in CALIBRATION_POINT_KEYS:
            if self.points[key] is None:
                return key
        return None

    def validate_bounds(self, x: float, y: float) -> Tuple[bool, Optional[str]]:
        """Verifies that (x, y) lies inside image boundaries."""
        if self.orig_w is not None and (x < 0.0 or x >= self.orig_w):
            return False, f"X coordinate {x:.1f} is outside image width [0, {self.orig_w}]."
        if self.orig_h is not None and (y < 0.0 or y >= self.orig_h):
            return False, f"Y coordinate {y:.1f} is outside image height [0, {self.orig_h}]."
        return True, None

    def validate_not_duplicate(
        self, x: float, y: float, ignore_key: Optional[str] = None, min_distance_px: float = 6.0
    ) -> Tuple[bool, Optional[str]]:
        """Checks that the point is not accidentally clicked right on top of an existing point."""
        for k, pt in self.points.items():
            if pt is not None and k != ignore_key:
                dist = np.hypot(x - pt[0], y - pt[1])
                if dist < min_distance_px:
                    return False, f"Point too close ({dist:.1f}px) to existing landmark '{k}'."
        return True, None

    def validate_geometric_ordering(self, key: str, x: float, y: float) -> Tuple[bool, Optional[str]]:
        """
        Validates basic perspective orientation constraints:
          - near_top must be above near_base
          - far_top must be above far_base
          - far_base must appear higher in image than near_base
          - pitch right corners must be to the right of pitch left corners
        """
        # 1. near_base vs near_top
        if key == "near_base" and self.points["near_top"] is not None:
            if y <= self.points["near_top"][1]:
                return False, "near_base must be positioned below near_top in the frame."
        elif key == "near_top" and self.points["near_base"] is not None:
            if y >= self.points["near_base"][1]:
                return False, "near_top must be positioned above near_base in the frame."

        # 2. far_base vs far_top
        if key == "far_base" and self.points["far_top"] is not None:
            if y <= self.points["far_top"][1]:
                return False, "far_base must be positioned below far_top in the frame."
        elif key == "far_top" and self.points["far_base"] is not None:
            if y >= self.points["far_base"][1]:
                return False, "far_top must be positioned above far_base in the frame."

        # 3. near_base vs far_base (perspective foreshortening)
        if key == "far_base" and self.points["near_base"] is not None:
            if y >= self.points["near_base"][1]:
                return False, "far_base appears lower than near_base. In perspective, far stumps appear higher."
        elif key == "near_base" and self.points["far_base"] is not None:
            if y <= self.points["far_base"][1]:
                return False, "near_base appears higher than far_base."

        # 4. pitch right vs left
        if key == "pitch_right_near" and self.points["pitch_left_near"] is not None:
            if x <= self.points["pitch_left_near"][0]:
                return False, "pitch_right_near must be to the right of pitch_left_near."
        elif key == "pitch_right_far" and self.points["pitch_left_far"] is not None:
            if x <= self.points["pitch_left_far"][0]:
                return False, "pitch_right_far must be to the right of pitch_left_far."

        return True, None

    def add_point(self, x: float, y: float, key: Optional[str] = None) -> str:
        """
        Assigns coordinate (x, y) to the next point or a specified key.
        Coordinates must be in original full-resolution video pixels.
        Validates bounds, duplicates, and geometric sanity.
        """
        target_key = key or self.next_required_point_key
        if target_key is None:
            raise ValueError("All 8 calibration points have already been set. Confirm or reset.")
        if target_key not in CALIBRATION_POINT_KEYS:
            raise KeyError(f"Invalid point key: {target_key}")

        # Bounds check
        b_ok, b_err = self.validate_bounds(x, y)
        if not b_ok:
            raise ValueError(b_err)

        # Duplicate check
        d_ok, d_err = self.validate_not_duplicate(x, y, ignore_key=target_key)
        if not d_ok:
            raise ValueError(d_err)

        # Geometric check
        g_ok, g_err = self.validate_geometric_ordering(target_key, x, y)
        if not g_ok:
            raise ValueError(g_err)

        self.points[target_key] = (float(x), float(y))
        self.history.append(target_key)
        return target_key

    def set_point_explicit(self, key: str, x: float, y: float):
        """Re-assigns a specific point with full validation."""
        self.add_point(x, y, key=key)

    def undo(self) -> Optional[str]:
        """Undoes the most recent point selection."""
        if not self.history:
            return None
        last_key = self.history.pop()
        self.points[last_key] = None
        return last_key

    def reset(self):
        """Resets all 8 calibration points."""
        for key in CALIBRATION_POINT_KEYS:
            self.points[key] = None
        self.history.clear()

    def is_complete(self) -> bool:
        """Checks if all 8 points have been selected."""
        return all(self.points[k] is not None for k in CALIBRATION_POINT_KEYS)

    def get_points_dict(self) -> Dict[str, Tuple[float, float]]:
        """Returns completed points dictionary."""
        if not self.is_complete():
            missing = [k for k in CALIBRATION_POINT_KEYS if self.points[k] is None]
            raise ValueError(f"Calibration points incomplete. Missing: {missing}")
        return {k: self.points[k] for k in CALIBRATION_POINT_KEYS}

    def get_image_points_array(self) -> np.ndarray:
        """
        Returns the 8 2D image coordinates as an (8, 2) numpy array in canonical order.
        """
        pts_dict = self.get_points_dict()
        return np.array([pts_dict[k] for k in CALIBRATION_POINT_KEYS], dtype=np.float64)

    def get_ground_image_points_array(self) -> np.ndarray:
        """
        Returns the 4 pitch corner image coordinates in order:
        [pitch_left_near, pitch_right_near, pitch_right_far, pitch_left_far]
        """
        pts_dict = self.get_points_dict()
        return np.array([
            pts_dict["pitch_left_near"],
            pts_dict["pitch_right_near"],
            pts_dict["pitch_right_far"],
            pts_dict["pitch_left_far"],
        ], dtype=np.float64)

    def to_serializable(self) -> Dict[str, Any]:
        """Converts points to JSON/YAML serializable dictionary with exact float values."""
        return {
            k: [float(v[0]), float(v[1])] if v is not None else None
            for k, v in self.points.items()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], width: Optional[int] = None, height: Optional[int] = None) -> "CalibrationPointManager":
        """Instantiates and loads points from a saved dictionary."""
        mgr = cls(image_width=width, image_height=height)
        for k in CALIBRATION_POINT_KEYS:
            if k in data and data[k] is not None:
                mgr.set_point_explicit(k, float(data[k][0]), float(data[k][1]))
        return mgr
