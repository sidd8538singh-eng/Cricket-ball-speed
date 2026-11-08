"""
Coordinate Manager: Handles transformations between display window coordinates,
tiled crops, normalized bounding boxes, and original high-resolution video pixels.
"""

from typing import Tuple, List, Union
import numpy as np


class CoordinateManager:
    """
    Ensures that regardless of UI zoom, pan, downsampling for display, or model tiling,
    all calibration and detection coordinates map accurately to original video pixels.
    """

    def __init__(self, original_width: int, original_height: int):
        self.orig_w = original_width
        self.orig_h = original_height

    def display_to_original(
        self,
        disp_x: float,
        disp_y: float,
        scale: float,
        pan_x: float,
        pan_y: float
    ) -> Tuple[float, float]:
        """
        Converts coordinates from a zoomed/panned display canvas to original video pixel space.
        disp_coord = (orig_coord * scale) + pan
        => orig_coord = (disp_coord - pan) / scale
        """
        orig_x = (disp_x - pan_x) / scale
        orig_y = (disp_y - pan_y) / scale
        # Clamp to image bounds
        orig_x = float(np.clip(orig_x, 0.0, self.orig_w - 1.0))
        orig_y = float(np.clip(orig_y, 0.0, self.orig_h - 1.0))
        return orig_x, orig_y

    def original_to_display(
        self,
        orig_x: float,
        orig_y: float,
        scale: float,
        pan_x: float,
        pan_y: float
    ) -> Tuple[int, int]:
        """Converts original pixel coordinates to the current display view."""
        disp_x = int(round((orig_x * scale) + pan_x))
        disp_y = int(round((orig_y * scale) + pan_y))
        return disp_x, disp_y

    def tile_to_original(
        self,
        tile_box: Union[Tuple[float, float, float, float], List[float]],
        tile_offset_x: int,
        tile_offset_y: int
    ) -> Tuple[float, float, float, float]:
        """
        Converts bounding box [x1, y1, x2, y2] from local tile coordinates
        back to original full-frame pixel coordinates.
        """
        x1, y1, x2, y2 = tile_box
        return (
            float(x1 + tile_offset_x),
            float(y1 + tile_offset_y),
            float(x2 + tile_offset_x),
            float(y2 + tile_offset_y)
        )

    def normalized_to_original(
        self, norm_box: Union[Tuple[float, float, float, float], List[float]]
    ) -> Tuple[float, float, float, float]:
        """Converts normalized [0, 1] bounding box [nx1, ny1, nx2, ny2] to pixel coordinates."""
        nx1, ny1, nx2, ny2 = norm_box
        return (
            float(nx1 * self.orig_w),
            float(ny1 * self.orig_h),
            float(nx2 * self.orig_w),
            float(ny2 * self.orig_h)
        )
