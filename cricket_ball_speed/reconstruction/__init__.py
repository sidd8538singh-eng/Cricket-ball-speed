"""
3D Ball Reconstruction and Trajectory Fitting Subsystem.
Includes camera viewing rays, depth estimation, 3D point representations,
and gravity-aware non-linear trajectory optimization.
"""

from .camera_rays import Ray3D, CameraRayBuilder, triangulate_two_rays
from .depth_estimation import RayDepthEstimator
from .ball_3d import Ball3DPoint
from .uncertainty import UncertaintyEstimator
from .trajectory_fitting import TrajectoryFitter, TrajectoryFitResult

__all__ = [
    "Ray3D",
    "CameraRayBuilder",
    "triangulate_two_rays",
    "RayDepthEstimator",
    "Ball3DPoint",
    "UncertaintyEstimator",
    "TrajectoryFitter",
    "TrajectoryFitResult",
]
