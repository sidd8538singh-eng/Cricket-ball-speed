"""
Calibration Validator: Rigorous perspective, orthogonality, and mounting quality auditor.
Validates that camera parameters strictly conform to cricket physics and geometric laws.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from .camera_model import CameraModel
from .reprojection import ReprojectionReport
from .point_selector import CalibrationPointManager


class CalibrationValidator:
    """
    Ensures solved camera calibration complies with:
      - Perspective foreshortening laws
      - Vertical wicket orthogonality
      - Permitted camera height and mounting bounds
      - Reprojection error tolerances
      - Numerical conditioning
    """

    def __init__(
        self,
        min_camera_height_m: float = 1.35,
        max_camera_height_m: float = 2.85,
        min_distance_behind_m: float = 0.80,
        max_distance_behind_m: float = 4.20,
        max_lateral_offset_m: float = 2.50,
        max_mean_reprojection_px: float = 5.0,
        max_single_point_reprojection_px: float = 9.0,
        max_vertical_tilt_deg: float = 15.0,
    ):
        self.min_height = min_camera_height_m
        self.max_height = max_camera_height_m
        self.min_dist = min_distance_behind_m
        self.max_dist = max_distance_behind_m
        self.max_lat = max_lateral_offset_m
        self.max_mean_reproj = max_mean_reprojection_px
        self.max_single_reproj = max_single_point_reprojection_px
        self.max_vert_tilt = max_vertical_tilt_deg

    def validate_points_perspective(self, points: Dict[str, Tuple[float, float]]) -> List[str]:
        """Validates the geometric arrangement of clicked points in image plane."""
        failures = []

        # 1. Near wicket height vs far wicket height in image pixels
        near_h = abs(points["near_base"][1] - points["near_top"][1])
        far_h = abs(points["far_base"][1] - points["far_top"][1])
        if near_h <= far_h:
            failures.append(
                f"Perspective error: Near wicket image height ({near_h:.1f}px) "
                f"must be taller than far wicket image height ({far_h:.1f}px)."
            )

        # 2. Near pitch width vs far pitch width in image pixels
        near_w = abs(points["pitch_right_near"][0] - points["pitch_left_near"][0])
        far_w = abs(points["pitch_right_far"][0] - points["pitch_left_far"][0])
        if near_w <= far_w:
            failures.append(
                f"Perspective error: Near pitch image width ({near_w:.1f}px) "
                f"must be wider than far pitch image width ({far_w:.1f}px)."
            )

        # 3. Far stumps should be higher up in image (smaller Y) than near stumps
        if points["far_base"][1] >= points["near_base"][1]:
            failures.append("Perspective error: Far wicket base appears lower in image than near wicket base.")

        # 4. Left vs Right consistency
        if points["pitch_left_near"][0] >= points["pitch_right_near"][0]:
            failures.append("Pitch left-near corner is to the right of pitch right-near corner.")
        if points["pitch_left_far"][0] >= points["pitch_right_far"][0]:
            failures.append("Pitch left-far corner is to the right of pitch right-far corner.")

        return failures

    def validate_wicket_orthogonality(self, camera: CameraModel) -> Tuple[bool, float, List[str]]:
        """
        Validates that the solved vertical axis is aligned with physical gravity,
        meaning stump rods project vertically with minimal camera roll/yaw distortion.
        """
        failures = []
        # Ray direction for vertical axis in camera coordinates: R * [0, 0, 1]^T
        cam_z = camera.R @ np.array([0.0, 0.0, 1.0]).reshape((3, 1))
        # Angle with camera vertical (-Y axis points up in image coordinates)
        # Projected vertical vector in image:
        vps = camera.get_vanishing_points()
        vp_z = vps.get("VP_Z")
        # In a typical behind-the-bowler camera with minimal roll, VP_Z is high or low near center X
        roll_angle_deg = float(np.degrees(np.arctan2(camera.R[0, 2], camera.R[1, 2])))
        if abs(roll_angle_deg) > self.max_vert_tilt:
            failures.append(
                f"Excessive camera roll ({roll_angle_deg:.1f}° > {self.max_vert_tilt}°). "
                f"Camera must be mounted level without heavy roll tilt."
            )

        return len(failures) == 0, roll_angle_deg, failures

    def validate_calibration(
        self,
        camera: CameraModel,
        reproj_report: ReprojectionReport,
        points_manager: CalibrationPointManager,
        uncertainty_report: Optional[Any] = None
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Runs comprehensive validation on the solved camera calibration.
        Returns (is_valid, failure_reasons, diagnostic_dict).
        """
        failures: List[str] = []
        pts_dict = points_manager.get_points_dict()

        # Perspective checks on clicks
        persp_failures = self.validate_points_perspective(pts_dict)
        failures.extend(persp_failures)

        # Reprojection checks
        if reproj_report.mean_error_px > self.max_mean_reproj:
            failures.append(
                f"Mean reprojection error {reproj_report.mean_error_px:.2f}px exceeds limit {self.max_mean_reproj:.2f}px."
            )
        if reproj_report.max_error_px > self.max_single_reproj:
            failures.append(
                f"Max reprojection error {reproj_report.max_error_px:.2f}px on '{reproj_report.worst_point}' "
                f"exceeds limit {self.max_single_reproj:.2f}px."
            )

        # Camera mounting checks
        cam_h = camera.camera_height_m
        cam_dist = camera.camera_distance_behind_near_stumps_m
        cam_lat = camera.camera_lateral_offset_m

        if not (self.min_height <= cam_h <= self.max_height):
            failures.append(
                f"Solved camera height {cam_h:.2f}m is outside plausible setup range [{self.min_height}, {self.max_height}]m."
            )

        if not (self.min_dist <= cam_dist <= self.max_dist):
            failures.append(
                f"Solved camera distance behind stumps {cam_dist:.2f}m is outside plausible range [{self.min_dist}, {self.max_dist}]m."
            )

        if abs(cam_lat) > self.max_lat:
            failures.append(
                f"Solved camera lateral offset {cam_lat:.2f}m exceeds max off-axis limit (+-{self.max_lat}m)."
            )

        # Optical axis check: camera must be pointing forward down-pitch (+Y in world)
        # Looking vector in world coordinates: R^T * [0, 0, 1]^T
        look_dir = camera.R.T @ np.array([0.0, 0.0, 1.0])
        if look_dir[1] <= 0.3:
            failures.append(
                f"Camera optical axis is not directed down the pitch (look vector Y = {look_dir[1]:.2f} <= 0.3)."
            )

        # Orthogonality check
        ortho_ok, roll_deg, ortho_fails = self.validate_wicket_orthogonality(camera)
        failures.extend(ortho_fails)

        # Conditioning check
        if uncertainty_report is not None and not uncertainty_report.is_well_constrained:
            failures.append(f"Uncertainty warning: {uncertainty_report.deficiency_report}")

        is_valid = len(failures) == 0
        diagnostics = {
            "is_valid": is_valid,
            "camera_height_m": round(cam_h, 3),
            "camera_distance_behind_m": round(cam_dist, 3),
            "camera_lateral_offset_m": round(cam_lat, 3),
            "camera_roll_deg": round(roll_deg, 2),
            "fov_x_deg": round(camera.fov_x_deg, 2),
            "fov_y_deg": round(camera.fov_y_deg, 2),
            "mean_reproj_err_px": round(reproj_report.mean_error_px, 3),
            "max_reproj_err_px": round(reproj_report.max_error_px, 3),
            "rmse_reproj_err_px": round(reproj_report.rmse_px, 3),
            "reprojection_acceptable": reproj_report.is_acceptable,
            "perspective_valid": len(persp_failures) == 0,
            "failures": failures,
        }

        return is_valid, failures, diagnostics
