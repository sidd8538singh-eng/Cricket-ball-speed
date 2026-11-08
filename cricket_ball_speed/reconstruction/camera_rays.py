"""
Camera Rays: Computes 3D viewing rays from 2D pixel coordinates using calibrated camera models.
Encapsulates single-ray back-projection, point-to-ray orthogonal projections,
and multi-camera skew ray triangulation.
"""

from typing import Tuple, List, Optional
import numpy as np
from calibration.camera_model import CameraModel


class Ray3D:
    """Represents an infinite 3D line ray defined by camera center C and unit direction vector d."""

    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = np.asarray(origin, dtype=np.float64).flatten()
        d = np.asarray(direction, dtype=np.float64).flatten()
        norm = float(np.linalg.norm(d))
        self.direction = (d / norm) if norm > 1e-12 else d

    def point_at_distance(self, distance: float) -> np.ndarray:
        """Evaluates 3D point along ray at distance lambda from camera center."""
        return self.origin + float(distance) * self.direction

    def distance_to_point(self, pt: np.ndarray) -> float:
        """Computes shortest perpendicular distance from point to the ray."""
        pt_arr = np.asarray(pt, dtype=np.float64).flatten()
        v = pt_arr - self.origin
        proj = np.dot(v, self.direction)
        perp = v - proj * self.direction
        return float(np.linalg.norm(perp))

    def closest_point_to(self, pt: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Projects point orthogonally onto the ray.
        Returns: (closest_point_3d, distance_along_ray_lambda)
        """
        pt_arr = np.asarray(pt, dtype=np.float64).flatten()
        v = pt_arr - self.origin
        dist_along = float(np.dot(v, self.direction))
        closest = self.origin + dist_along * self.direction
        return closest, dist_along


class CameraRayBuilder:
    """Constructs 3D back-projection rays using the calibrated CameraModel."""

    def __init__(self, camera_model: CameraModel):
        self.camera = camera_model

    def build_ray(self, u: float, v: float) -> Ray3D:
        """
        Computes 3D ray starting at camera optical center C passing through pixel (u, v).
        """
        origin, direction = self.camera.back_project_ray(float(u), float(v))
        return Ray3D(origin, direction)


def triangulate_two_rays(ray1: Ray3D, ray2: Ray3D) -> Tuple[np.ndarray, float]:
    """
    Computes the midpoint of the shortest segment connecting two skew 3D rays.
    Multi-camera triangulation baseline.
    Returns: (midpoint_3d, shortest_distance)
    """
    p1 = ray1.origin
    d1 = ray1.direction
    p2 = ray2.origin
    d2 = ray2.direction

    w0 = p1 - p2
    a = float(np.dot(d1, d1))
    b = float(np.dot(d1, d2))
    c = float(np.dot(d2, d2))
    d = float(np.dot(d1, w0))
    e = float(np.dot(d2, w0))

    denom = a * c - b * b
    if abs(denom) < 1e-9:
        # Parallel rays
        midpoint = (p1 + p2) / 2.0
        return midpoint, float(np.linalg.norm(w0))

    sc = (b * e - c * d) / denom
    tc = (a * e - b * d) / denom

    closest1 = p1 + sc * d1
    closest2 = p2 + tc * d2

    midpoint = (closest1 + closest2) / 2.0
    dist = float(np.linalg.norm(closest1 - closest2))
    return midpoint, dist
