"""
Depth Estimation: Heteroskedastic depth prior estimation from apparent ball diameter.
Solves for distance along camera viewing rays using pinhole optics and physical ball diameter (0.0715 m).

Mathematical Principle & Error Propagation:
------------------------------------------
For a sphere of physical diameter D_phys = 0.0715 m viewed by a camera of focal length f:
    d_px(Z) = (f * D_phys) / Z
    Z = (f * D_phys) / d_px

Error propagation:
    dZ / d(d_px) = - (f * D_phys) / (d_px^2)
    sigma_Z = |f * D_phys / (d_px^2)| * sigma_d

CRITICAL MONOCULAR INSIGHT:
When the ball is near the far stumps (Z ~ 20-25m), d_px drops to 3-5 pixels.
A tiny segmentation error of +/-0.75 pixel causes depth uncertainty sigma_Z > 8-12 metres!
Therefore, apparent diameter is treated strictly as a noisy, heteroskedastic prior,
weighted appropriately by inverse variance rather than as an absolute depth measurement.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from calibration.camera_model import CameraModel
from ball_processing.apparent_diameter import ApparentDiameterEstimator, CRICKET_BALL_DIAMETER_M
from .camera_rays import Ray3D


class RayDepthEstimator:
    """
    Computes depth along the camera ray using the apparent diameter of the cricket ball.
    Implements heteroskedastic variance propagation reflecting high noise at small pixel sizes.
    """

    def __init__(
        self,
        camera_model: CameraModel,
        ball_diameter_m: float = CRICKET_BALL_DIAMETER_M,
        pixel_diameter_noise_std: float = 0.75
    ):
        """
        Args:
            camera_model: Calibrated pinhole camera model.
            ball_diameter_m: Official cricket ball diameter (0.0715 m).
            pixel_diameter_noise_std: Expected pixel bounding box diameter uncertainty (pixels).
        """
        self.camera = camera_model
        self.ball_diameter_m = float(ball_diameter_m)
        self.sigma_d_px = float(pixel_diameter_noise_std)
        self.focal_length = float((camera_model.fx + camera_model.fy) / 2.0)

    def estimate_depth_and_variance(
        self,
        apparent_diameter_px: float
    ) -> Tuple[float, float]:
        """
        Calculates depth Z along camera optical axis (metres) and analytic variance.
        Returns:
            (depth_m, variance_m2)
        """
        d_px = max(0.5, float(apparent_diameter_px))
        
        # Nominal depth
        depth_m = (self.focal_length * self.ball_diameter_m) / d_px

        # Analytic standard deviation via first-order Taylor expansion
        # sigma_Z = (f * D / d_px^2) * sigma_d
        sigma_z_m = ((self.focal_length * self.ball_diameter_m) / (d_px ** 2)) * self.sigma_d_px

        # Floor uncertainty for numerical stability
        sigma_z_m = max(0.05, sigma_z_m)
        variance_m2 = float(sigma_z_m ** 2)

        return float(depth_m), variance_m2

    def estimate_point_along_ray(
        self,
        ray: Ray3D,
        apparent_diameter_px: float
    ) -> Tuple[np.ndarray, float]:
        """
        Projects along the camera ray to the estimated apparent size depth.
        Returns:
            (point_3d, depth_uncertainty_m)
        """
        depth_m, variance_m2 = self.estimate_depth_and_variance(apparent_diameter_px)
        pt_3d = ray.point_at_distance(depth_m)
        return pt_3d, float(np.sqrt(variance_m2))
