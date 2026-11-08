"""
Apparent Diameter: Depth estimation cue from optical ball sizing.
Section 13: Cricket ball physical diameter D = 0.0715 m.
Apparent diameter in pixels d_px ~ f * (D / Z_cam).
Used as a noisy prior and geometric depth cue, not an exact measurement.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


CRICKET_BALL_DIAMETER_M = 0.0715 # 7.15 cm


class ApparentDiameterEstimator:
    """
    Analyzes pixel size of the ball candidate to compute approximate camera-frame depth.
    Accounts for motion blur elongation.
    """

    def __init__(self, physical_diameter_m: float = CRICKET_BALL_DIAMETER_M):
        self.ball_diameter_m = physical_diameter_m

    def compute_apparent_diameter(
        self, width_px: float, height_px: float
    ) -> Tuple[float, float]:
        """
        Computes apparent spherical diameter (px) and aspect ratio elongation.
        For a motion-blurred ball, the width perpendicular to motion is closer to true diameter.
        Returns (effective_diameter_px, aspect_ratio).
        """
        w = max(0.5, float(width_px))
        h = max(0.5, float(height_px))
        aspect_ratio = max(w, h) / min(w, h)

        # In fast bowling (120-150 km/h), motion blur elongates the ball along the trajectory.
        # The smaller dimension is a less blurred estimate of diameter:
        effective_diameter = float(np.sqrt(w * h))
        return effective_diameter, aspect_ratio

    def estimate_depth_from_diameter(
        self,
        apparent_diameter_px: float,
        focal_length_px: float
    ) -> Tuple[float, float]:
        """
        Estimates depth Z_cam along camera optical axis:
        Z_cam ~ (f * D) / d_px
        Returns (estimated_depth_m, uncertainty_std_m).
        """
        if apparent_diameter_px <= 0.5:
            return float("nan"), float("inf")

        z_est = (focal_length_px * self.ball_diameter_m) / apparent_diameter_px
        
        # Uncertainty grows quadratically with distance (d_px decreases):
        # sigma_Z ~ (f * D / d_px^2) * sigma_d
        # Assuming diameter detection noise sigma_d ~ 1.5 pixels:
        sigma_d = 1.5
        sigma_z = (focal_length_px * self.ball_diameter_m / (apparent_diameter_px ** 2)) * sigma_d

        return float(z_est), float(sigma_z)
