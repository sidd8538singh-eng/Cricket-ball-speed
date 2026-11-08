"""
Ground-Plane Homography: Maps between world ground plane (Z = 0) and image plane (u, v).
CRITICAL MANDATE:
  Ground-plane homography applies STRICTLY to points resting on the pitch surface (Z = 0).
  NEVER apply ground-plane homography to an airborne cricket ball.
  Applying a 2D planar homography to an airborne object produces geometrically false 3D positions.
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


class GroundPlaneHomography:
    """
    Computes and manages the planar projective transformation (homography)
    between the cricket pitch ground plane (X, Y, Z=0) and the 2D camera image.
    """

    def __init__(self, ground_points_3d: np.ndarray, ground_points_2d: np.ndarray):
        """
        ground_points_3d: (N, 3) array where Z must be 0
        ground_points_2d: (N, 2) array of pixel coordinates
        """
        if len(ground_points_3d) < 4 or len(ground_points_2d) < 4:
            raise ValueError("At least 4 ground-plane correspondences required for homography.")

        # Strict check: all 3D points must have Z = 0
        z_vals = ground_points_3d[:, 2]
        if np.max(np.abs(z_vals)) > 1e-4:
            raise ValueError(
                "All 3D points passed to GroundPlaneHomography must have Z = 0 (ground plane). "
                "Ground homography is strictly invalid for 3D points above the ground."
            )

        # 2D world points (X, Y)
        self.world_pts_2d = ground_points_3d[:, :2].astype(np.float64)
        self.img_pts_2d = ground_points_2d.astype(np.float64)

        # H_world_to_image: [u, v, 1]^T ~ H * [X, Y, 1]^T
        self.H_w2i, mask = cv2.findHomography(self.world_pts_2d, self.img_pts_2d, method=0)
        if self.H_w2i is None:
            raise RuntimeError("Homography estimation failed (degenerate points).")

        det = np.linalg.det(self.H_w2i)
        if abs(det) < 1e-12:
            raise RuntimeError("Singular homography matrix. Cannot invert.")
        self.H_i2w = np.linalg.inv(self.H_w2i)

        self._validate_conditioning()

    def _validate_conditioning(self):
        """Evaluates numerical condition number of the homography."""
        cond = np.linalg.cond(self.H_w2i)
        self.condition_number = float(cond)

    def world_to_image(self, x: float, y: float, z: float = 0.0) -> Tuple[float, float]:
        """
        Maps world ground coordinate (X, Y, Z=0) to image coordinates (u, v).
        Rejects non-zero Z!
        """
        if abs(z) > 1e-3:
            raise ValueError(
                f"Ground homography cannot project 3D point with height Z={z:.3f}m! "
                "Use the full 3D CameraModel for non-ground points."
            )

        pt_w = np.array([x, y, 1.0], dtype=np.float64).reshape((3, 1))
        pt_i = self.H_w2i @ pt_w
        if abs(pt_i[2, 0]) < 1e-9:
            raise ZeroDivisionError("Point mapped to projective infinity.")
        u = float(pt_i[0, 0] / pt_i[2, 0])
        v = float(pt_i[1, 0] / pt_i[2, 0])
        return u, v

    def image_to_ground(self, u: float, v: float) -> Tuple[float, float, float]:
        """
        Maps image coordinate (u, v) to world ground coordinate (X, Y, Z=0).
        WARNING: Only valid if the physical feature is strictly confirmed to be resting on the ground!
        Returns (X, Y, 0.0).
        """
        pt_i = np.array([u, v, 1.0], dtype=np.float64).reshape((3, 1))
        pt_w = self.H_i2w @ pt_i
        if abs(pt_w[2, 0]) < 1e-9:
            raise ZeroDivisionError("Point mapped to projective infinity.")
        x = float(pt_w[0, 0] / pt_w[2, 0])
        y = float(pt_w[1, 0] / pt_w[2, 0])
        return x, y, 0.0

    def compute_reprojection_error(self) -> float:
        """Computes mean reprojection error across ground reference points."""
        errors = []
        for (xw, yw), (ui, vi) in zip(self.world_pts_2d, self.img_pts_2d):
            u_proj, v_proj = self.world_to_image(xw, yw, z=0.0)
            err = np.hypot(u_proj - ui, v_proj - vi)
            errors.append(err)
        return float(np.mean(errors))

    def summary(self) -> Dict[str, Any]:
        return {
            "condition_number": round(self.condition_number, 2),
            "mean_reprojection_error_px": round(self.compute_reprojection_error(), 3),
            "H_matrix": self.H_w2i.tolist(),
            "is_well_conditioned": self.condition_number < 1e5,
        }
