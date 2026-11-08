"""
World Coordinate System Definition.
Defines the canonical 3D coordinates for all 8 cricket calibration landmarks.

Reference Frame:
  - Origin A (0,0,0) = centre of near wicket base at ground level.
  - B (0,L,0) = centre of far wicket base at ground level.
  - Line A-B defines the longitudinal +Y axis.
  - +X is lateral axis (to the right when looking from near to far wicket).
  - +Z is vertical axis (upwards from ground plane).

Physical Dimensions:
  - Bowling-crease-to-bowling-crease length: 20.12 m (22 yards).
  - Accounts for offsets between bowling creases and wicket base centres.
  - Pitch width: 3.048 m (10 ft).
  - Stump height: 0.711 m (28 inches).
  - Cricket ball diameter: 0.0715 m.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


# The standard 8 calibration point keys in strict sequential order
CALIBRATION_POINT_KEYS = [
    "near_top",
    "near_base",
    "far_top",
    "far_base",
    "pitch_left_near",
    "pitch_right_near",
    "pitch_left_far",
    "pitch_right_far"
]


@dataclass
class WorldPitchGeometry:
    """
    Holds physical cricket dimensions and computes canonical 3D coordinates.
    Strictly accounts for the offset between the bowling crease line and wicket-base centres.
    """
    # ICC Law: Distance between bowling crease lines is exactly 20.12 m (22 yards)
    pitch_length_crease_to_crease_m: float = 20.12

    # Offset from near bowling crease to near wicket base centre A (0,0,0).
    # Positive if stumps are behind the crease; negative if in front.
    near_wicket_offset_from_crease_m: float = 0.0

    # Offset from far bowling crease to far wicket base centre B (0,L,0).
    # Positive if stumps are behind the far crease (further away); negative if in front.
    far_wicket_offset_from_crease_m: float = 0.0

    # Standard pitch width between marked outer pitch borders: 3.048 m (10 ft)
    pitch_width_m: float = 3.048

    # Standard stump height above ground: 0.711 m (28 in)
    stump_height_m: float = 0.711

    # Standard cricket ball diameter: 0.0715 m (7.15 cm)
    ball_diameter_m: float = 0.0715

    # Popping crease distance in front of bowling crease: 1.2192 m (4 ft)
    popping_crease_offset_m: float = 1.2192

    # Explicit pitch corner placement:
    # "crease" -> pitch corners are marked at the bowling creases
    # "popping" -> pitch corners are marked at the popping creases
    # "custom" -> uses explicit offsets below
    pitch_corner_reference: str = "crease"
    near_pitch_y_offset_m: float = 0.0
    far_pitch_y_offset_m: float = 0.0

    @property
    def wicket_distance_m(self) -> float:
        """
        Physical distance L between near wicket base centre A and far wicket base centre B.
        L = crease_to_crease - near_offset + far_offset.
        When wickets are pitched on the crease lines (offsets = 0), L = 20.12 m.
        """
        return (
            self.pitch_length_crease_to_crease_m
            + self.far_wicket_offset_from_crease_m
            - self.near_wicket_offset_from_crease_m
        )

    @property
    def L(self) -> float:
        """Alias for wicket_distance_m."""
        return self.wicket_distance_m

    def get_world_coordinates(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary mapping each of the 8 canonical calibration keys
        to its 3D world coordinate [X, Y, Z] in metres.

        Frame Definition:
          A = near_base = (0, 0, 0)
          B = far_base  = (0, L, 0)
          Line A-B is the permanent +Y axis.
        """
        half_width = self.pitch_width_m / 2.0
        L = self.wicket_distance_m
        h = self.stump_height_m

        # Compute Y coordinates of pitch corners relative to near wicket base A
        if self.pitch_corner_reference == "crease":
            # Near bowling crease is at Y = -near_wicket_offset_from_crease_m
            near_pitch_y = -self.near_wicket_offset_from_crease_m + self.near_pitch_y_offset_m
            # Far bowling crease is at Y = 20.12 - near_wicket_offset_from_crease_m
            far_pitch_y = (
                self.pitch_length_crease_to_crease_m
                - self.near_wicket_offset_from_crease_m
                + self.far_pitch_y_offset_m
            )
        elif self.pitch_corner_reference == "popping":
            near_pitch_y = (
                -self.near_wicket_offset_from_crease_m
                + self.popping_crease_offset_m
                + self.near_pitch_y_offset_m
            )
            far_pitch_y = (
                self.pitch_length_crease_to_crease_m
                - self.near_wicket_offset_from_crease_m
                - self.popping_crease_offset_m
                + self.far_pitch_y_offset_m
            )
        else: # "custom" or relative to wickets
            near_pitch_y = 0.0 + self.near_pitch_y_offset_m
            far_pitch_y = L + self.far_pitch_y_offset_m

        coords = {
            # Stumps 1 & 2: Near wicket
            "near_top": np.array([0.0, 0.0, h], dtype=np.float64),
            "near_base": np.array([0.0, 0.0, 0.0], dtype=np.float64),

            # Stumps 3 & 4: Far wicket
            "far_top": np.array([0.0, L, h], dtype=np.float64),
            "far_base": np.array([0.0, L, 0.0], dtype=np.float64),

            # Pitch Corners 5 & 6: Near corners on ground plane (Z=0)
            "pitch_left_near": np.array([-half_width, near_pitch_y, 0.0], dtype=np.float64),
            "pitch_right_near": np.array([+half_width, near_pitch_y, 0.0], dtype=np.float64),

            # Pitch Corners 7 & 8: Far corners on ground plane (Z=0)
            "pitch_left_far": np.array([-half_width, far_pitch_y, 0.0], dtype=np.float64),
            "pitch_right_far": np.array([+half_width, far_pitch_y, 0.0], dtype=np.float64),
        }
        return coords

    def get_world_points_array(self) -> np.ndarray:
        """Returns the 8 3D points as an (8, 3) numpy array ordered by CALIBRATION_POINT_KEYS."""
        coords_map = self.get_world_coordinates()
        return np.array([coords_map[k] for k in CALIBRATION_POINT_KEYS], dtype=np.float64)

    def get_ground_corners_3d(self) -> np.ndarray:
        """Returns the 4 pitch corners on the ground plane as a (4, 3) numpy array."""
        coords_map = self.get_world_coordinates()
        return np.array([
            coords_map["pitch_left_near"],
            coords_map["pitch_right_near"],
            coords_map["pitch_right_far"],
            coords_map["pitch_left_far"]
        ], dtype=np.float64)
