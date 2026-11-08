"""
Pinhole and Distorted Camera Model.
Manages intrinsics (K), extrinsics (R, t), camera optical centre, projection,
lens distortion, ray back-projection, vanishing points, and ground horizon.
"""

from typing import Tuple, Optional, Dict, Any, List
import cv2
import numpy as np


class CameraModel:
    """
    Encapsulates complete camera geometry for the cricket speed measurement system.
    Supports forward projection: 3D World (X, Y, Z) -> 2D Image (u, v)
    and backward projection: 2D Image (u, v) -> 3D Ray (origin, direction).
    Calculates vanishing points and physical mounting parameters.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray],
        rvec: np.ndarray,
        tvec: np.ndarray,
        image_width: int,
        image_height: int
    ):
        self.K = np.array(camera_matrix, dtype=np.float64)
        self.dist_coeffs = (
            np.array(dist_coeffs, dtype=np.float64).reshape((-1, 1))
            if dist_coeffs is not None
            else np.zeros((5, 1), dtype=np.float64)
        )
        self.rvec = np.array(rvec, dtype=np.float64).reshape((3, 1))
        self.tvec = np.array(tvec, dtype=np.float64).reshape((3, 1))
        self.width = image_width
        self.height = image_height

        # Compute rotation matrix R in SO(3)
        self.R, _ = cv2.Rodrigues(self.rvec)

        # Camera optical centre in world coordinates: C = -R^T * t
        self.optical_centre = (-self.R.T @ self.tvec).flatten()

        # Projection matrix P = K * [R | t]
        Rt = np.hstack((self.R, self.tvec))
        self.P = self.K @ Rt

        # Inverted intrinsic matrix for ray back-projection
        self.K_inv = np.linalg.inv(self.K)

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def fov_x_deg(self) -> float:
        """Horizontal field of view in degrees."""
        return float(np.degrees(2.0 * np.arctan(self.width / (2.0 * self.fx))))

    @property
    def fov_y_deg(self) -> float:
        """Vertical field of view in degrees."""
        return float(np.degrees(2.0 * np.arctan(self.height / (2.0 * self.fy))))

    @property
    def camera_height_m(self) -> float:
        """Height of camera lens centre above ground level (Z coordinate)."""
        return float(self.optical_centre[2])

    @property
    def camera_distance_behind_near_stumps_m(self) -> float:
        """Distance behind the near wicket base A(0,0,0) (negative Y in world coordinates)."""
        return float(-self.optical_centre[1])

    @property
    def camera_lateral_offset_m(self) -> float:
        """Lateral offset from the longitudinal pitch center line (X coordinate)."""
        return float(self.optical_centre[0])

    def project_point(self, world_pt_3d: np.ndarray) -> Tuple[float, float]:
        """
        Projects a 3D world coordinate [X, Y, Z] to 2D image coordinates (u, v)
        accounting for non-linear lens distortion.
        """
        pt_3d = np.array(world_pt_3d, dtype=np.float64).reshape((1, 1, 3))
        proj_2d, _ = cv2.projectPoints(pt_3d, self.rvec, self.tvec, self.K, self.dist_coeffs)
        u = float(proj_2d[0, 0, 0])
        v = float(proj_2d[0, 0, 1])
        return u, v

    def project_points(self, world_pts_3d: np.ndarray) -> np.ndarray:
        """Projects an (N, 3) array of world points to an (N, 2) array of pixel coordinates."""
        pts_3d = np.array(world_pts_3d, dtype=np.float64).reshape((-1, 1, 3))
        proj_2d, _ = cv2.projectPoints(pts_3d, self.rvec, self.tvec, self.K, self.dist_coeffs)
        return proj_2d.reshape((-1, 2))

    def back_project_ray(self, u: float, v: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs a 3D ray in world coordinates from a 2D image coordinate (u, v).
        Returns:
            ray_origin: np.ndarray (3,) - Camera optical centre in world coordinates
            ray_direction: np.ndarray (3,) - Unit direction vector in world coordinates
        """
        pts = np.array([[[u, v]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(pts, self.K, self.dist_coeffs, P=self.K)
        u_u = undistorted[0, 0, 0]
        v_u = undistorted[0, 0, 1]

        # Normalized camera ray: r_c = K^-1 * [u, v, 1]^T
        p_hom = np.array([u_u, v_u, 1.0], dtype=np.float64).reshape((3, 1))
        r_c = self.K_inv @ p_hom

        # Transform direction to world frame: d_w = R^T * r_c
        d_w = (self.R.T @ r_c).flatten()
        norm = np.linalg.norm(d_w)
        if norm > 1e-12:
            d_w = d_w / norm

        return self.optical_centre.copy(), d_w

    def get_vanishing_points(self) -> Dict[str, Tuple[float, float]]:
        """
        Computes the three vanishing points corresponding to world directions:
          - VP_X: Lateral pitch direction [1, 0, 0]^T
          - VP_Y: Longitudinal pitch direction [0, 1, 0]^T (pitch lines merge here)
          - VP_Z: Vertical upright direction [0, 0, 1]^T (stump rods merge here)
        """
        dirs = {
            "VP_X": np.array([1.0, 0.0, 0.0]),
            "VP_Y": np.array([0.0, 1.0, 0.0]),
            "VP_Z": np.array([0.0, 0.0, 1.0]),
        }
        vps = {}
        for name, d in dirs.items():
            cam_d = self.R @ d.reshape((3, 1))
            if abs(cam_d[2, 0]) > 1e-6:
                p_hom = self.K @ cam_d
                u = float(p_hom[0, 0] / p_hom[2, 0])
                v = float(p_hom[1, 0] / p_hom[2, 0])
                vps[name] = (u, v)
            else:
                vps[name] = (float("inf"), float("inf"))
        return vps

    def get_horizon_line(self) -> Tuple[float, float, float]:
        """
        Computes the horizon line equation a*u + b*v + c = 0 on the ground plane (Z = 0).
        The horizon line passes through VP_X and VP_Y.
        """
        vps = self.get_vanishing_points()
        vx = vps["VP_X"]
        vy = vps["VP_Y"]
        if not np.isinf(vx[0]) and not np.isinf(vy[0]):
            dx = vy[0] - vx[0]
            dy = vy[1] - vx[1]
            # Normal vector (a, b) = (-dy, dx)
            a = -dy
            b = dx
            c = -(a * vx[0] + b * vx[1])
            norm = np.hypot(a, b)
            if norm > 1e-9:
                return a / norm, b / norm, c / norm
        return 0.0, 1.0, -self.cy

    def to_dict(self) -> Dict[str, Any]:
        vps = self.get_vanishing_points()
        return {
            "K": self.K.tolist(),
            "dist_coeffs": self.dist_coeffs.flatten().tolist(),
            "rvec": self.rvec.flatten().tolist(),
            "tvec": self.tvec.flatten().tolist(),
            "R": self.R.tolist(),
            "optical_centre_world": self.optical_centre.tolist(),
            "camera_height_m": round(self.camera_height_m, 3),
            "camera_distance_behind_stumps_m": round(self.camera_distance_behind_near_stumps_m, 3),
            "camera_lateral_offset_m": round(self.camera_lateral_offset_m, 3),
            "fov_x_deg": round(self.fov_x_deg, 2),
            "fov_y_deg": round(self.fov_y_deg, 2),
            "vanishing_points": {
                k: [round(v[0], 1), round(v[1], 1)] if not np.isinf(v[0]) else "inf"
                for k, v in vps.items()
            },
            "image_width": self.width,
            "image_height": self.height,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraModel":
        K = np.array(data["K"], dtype=np.float64)
        dist = np.array(data["dist_coeffs"], dtype=np.float64)
        rvec = np.array(data["rvec"], dtype=np.float64)
        tvec = np.array(data["tvec"], dtype=np.float64)
        w = int(data["image_width"])
        h = int(data["image_height"])
        return cls(K, dist, rvec, tvec, w, h)
