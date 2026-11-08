"""
Coherent Camera Calibration Solver.
Jointly solves camera intrinsics (focal length f), extrinsics (R, t), and radial distortion
from the 8 non-coplanar cricket calibration landmarks.
Evaluates constraint sufficiency, Jacobian conditioning, parameter covariance, and uncertainty.
STRICT MANDATE: Never averages contradictory independently calculated camera models.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import cv2
from scipy.optimize import least_squares

from .camera_model import CameraModel
from .world_coordinates import WorldPitchGeometry, CALIBRATION_POINT_KEYS
from .reprojection import compute_reprojection_report, ReprojectionReport


@dataclass
class CalibrationUncertaintyReport:
    sigma_focal_length_px: float
    sigma_rotation_deg: Tuple[float, float, float]
    sigma_translation_m: Tuple[float, float, float]
    sigma_camera_height_m: float
    sigma_camera_distance_m: float
    jacobian_condition_number: float
    smallest_singular_value: float
    is_well_constrained: bool
    deficiency_report: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sigma_focal_length_px": round(self.sigma_focal_length_px, 2),
            "sigma_rotation_deg": [round(r, 3) for r in self.sigma_rotation_deg],
            "sigma_translation_m": [round(t, 4) for t in self.sigma_translation_m],
            "sigma_camera_height_m": round(self.sigma_camera_height_m, 4),
            "sigma_camera_distance_m": round(self.sigma_camera_distance_m, 4),
            "jacobian_condition_number": round(self.jacobian_condition_number, 2),
            "smallest_singular_value": round(self.smallest_singular_value, 6),
            "is_well_constrained": self.is_well_constrained,
            "deficiency_report": self.deficiency_report,
        }


class CameraCalibrator:
    """
    Estimates focal length, lens distortion, and 6-DOF camera pose from the 8 reference points:
      - 4 Pitch corners on ground plane (Z = 0)
      - Near wicket base (Z = 0) and top (Z = 0.711m)
      - Far wicket base (Z = 0) and top (Z = 0.711m)

    Analyzes system observability and Jacobian conditioning before accepting calibration.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        pitch_geometry: Optional[WorldPitchGeometry] = None,
        max_reprojection_error_px: float = 6.0
    ):
        self.width = image_width
        self.height = image_height
        self.geometry = pitch_geometry or WorldPitchGeometry()
        self.max_reprojection_error_px = max_reprojection_error_px
        self.cx = image_width / 2.0
        self.cy = image_height / 2.0

    def check_constraint_sufficiency(
        self, image_points: Dict[str, Tuple[float, float]]
    ) -> Tuple[bool, Optional[str]]:
        """
        Verifies whether the 8 image points provide sufficient 3D geometric disparity
        to uniquely constrain focal length and 6-DOF camera pose.
        If underconstrained, explicitly reports what additional information is required.
        """
        # 1. Missing keys
        for key in CALIBRATION_POINT_KEYS:
            if key not in image_points or image_points[key] is None:
                return False, f"Missing calibration landmark '{key}'. All 8 landmarks are mandatory."

        # 2. Vertical disparity check (stumps height in image)
        near_top = image_points["near_top"]
        near_base = image_points["near_base"]
        far_top = image_points["far_top"]
        far_base = image_points["far_base"]

        near_stump_px = float(np.hypot(near_top[0] - near_base[0], near_top[1] - near_base[1]))
        far_stump_px = float(np.hypot(far_top[0] - far_base[0], far_top[1] - far_base[1]))

        if near_stump_px < 10.0:
            return False, (
                f"Underconstrained vertical geometry: Near stump image height is only {near_stump_px:.1f}px (< 10px). "
                f"Cannot resolve vertical vanishing point and camera height. High-resolution zoom or closer camera needed."
            )

        if far_stump_px < 3.0:
            return False, (
                f"Underconstrained vertical geometry: Far stump image height is only {far_stump_px:.1f}px (< 3px). "
                f"Far wicket out-of-plane disparity is too weak to separate focal length from distance."
            )

        # 3. Longitudinal depth disparity check (pitch span in image)
        pitch_depth_px = float(abs(near_base[1] - far_base[1]))
        if pitch_depth_px < 40.0:
            return False, (
                f"Underconstrained longitudinal geometry: Pitch length disparity is only {pitch_depth_px:.1f}px (< 40px). "
                f"Pitch foreshortening is unresolvable. Both near and far wickets must be clearly framed."
            )

        # 4. Lateral width disparity check
        pl_near = image_points["pitch_left_near"]
        pr_near = image_points["pitch_right_near"]
        near_width_px = float(abs(pr_near[0] - pl_near[0]))
        if near_width_px < 30.0:
            return False, (
                f"Underconstrained lateral geometry: Pitch width is only {near_width_px:.1f}px (< 30px). "
                f"Cannot reliably constrain lateral scale."
            )

        return True, None

    def calibrate(
        self,
        image_points: Dict[str, Tuple[float, float]],
        initial_focal_length_px: Optional[float] = None,
        return_uncertainty: bool = False
    ) -> Any:
        """
        Executes coherent joint non-linear optimization across all 8 landmarks.
        Returns:
            (camera_model, reprojection_report) if return_uncertainty is False,
            (camera_model, reprojection_report, uncertainty_report) if return_uncertainty is True.
        """
        # Step 1: Constraint Sufficiency Check
        is_sufficient, deficiency_msg = self.check_constraint_sufficiency(image_points)
        if not is_sufficient:
            raise RuntimeError(f"CALIBRATION CONSTRAINT FAILURE: {deficiency_msg}")

        world_pts_dict = self.geometry.get_world_coordinates()

        obj_pts_list = [world_pts_dict[k] for k in CALIBRATION_POINT_KEYS]
        img_pts_list = [image_points[k] for k in CALIBRATION_POINT_KEYS]

        obj_pts = np.array(obj_pts_list, dtype=np.float64) # (8, 3)
        img_pts = np.array(img_pts_list, dtype=np.float64) # (8, 2)

        # Step 2: Nominal FOV Prior & PnP Initialization Search
        if initial_focal_length_px is None or initial_focal_length_px <= 0:
            f_prior = (self.width / 2.0) / np.tan(np.radians(60.0 / 2.0))
        else:
            f_prior = initial_focal_length_px

        f_candidates = np.linspace(f_prior * 0.5, f_prior * 2.0, 30)
        best_f = f_prior
        best_rvec = None
        best_tvec = None
        best_err = float("inf")

        for f_cand in f_candidates:
            K_cand = np.array([
                [f_cand, 0.0, self.cx],
                [0.0, f_cand, self.cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K_cand, None, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                continue

            R, _ = cv2.Rodrigues(rvec)
            c_world = (-R.T @ tvec).flatten()

            # Physical check: camera must be behind near stumps (Y < 0) and above ground (Z > 0)
            if c_world[1] > 0.0 or c_world[2] < 0.5:
                continue

            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K_cand, None)
            err = float(np.mean(np.linalg.norm(proj.reshape((-1, 2)) - img_pts, axis=1)))
            if err < best_err:
                best_err = err
                best_f = f_cand
                best_rvec = rvec
                best_tvec = tvec

        if best_rvec is None or best_tvec is None:
            # Fallback to standard iterative solvePnP
            K_init = np.array([[f_prior, 0, self.cx], [0, f_prior, self.cy], [0, 0, 1]], dtype=np.float64)
            success, best_rvec, best_tvec = cv2.solvePnP(obj_pts, img_pts, K_init, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not success:
                raise RuntimeError("solvePnP failed to converge on the 8 calibration landmarks.")
            best_f = f_prior

        # Step 3: Joint Non-Linear Refinement over [log(f), rx, ry, rz, tx, ty, tz, k1]
        x0 = np.array([
            np.log(best_f),
            best_rvec[0, 0], best_rvec[1, 0], best_rvec[2, 0],
            best_tvec[0, 0], best_tvec[1, 0], best_tvec[2, 0],
            0.0
        ], dtype=np.float64)

        def residuals(params: np.ndarray) -> np.ndarray:
            f_val = np.exp(params[0])
            r_val = params[1:4].reshape((3, 1))
            t_val = params[4:7].reshape((3, 1))
            k1_val = params[7]

            K = np.array([
                [f_val, 0.0, self.cx],
                [0.0, f_val, self.cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            dist = np.array([k1_val, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

            proj, _ = cv2.projectPoints(obj_pts, r_val, t_val, K, dist)
            diff = (proj.reshape((-1, 2)) - img_pts).flatten()
            return diff

        opt_res = least_squares(
            residuals,
            x0,
            method="lm",
            max_nfev=400,
            ftol=1e-9,
            xtol=1e-9
        )

        opt_f = float(np.exp(opt_res.x[0]))
        opt_rvec = opt_res.x[1:4].reshape((3, 1))
        opt_tvec = opt_res.x[4:7].reshape((3, 1))
        opt_k1 = float(opt_res.x[7])

        K_final = np.array([
            [opt_f, 0.0, self.cx],
            [0.0, opt_f, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        dist_final = np.array([opt_k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        camera_model = CameraModel(
            camera_matrix=K_final,
            dist_coeffs=dist_final,
            rvec=opt_rvec,
            tvec=opt_tvec,
            image_width=self.width,
            image_height=self.height
        )

        reproj_report = compute_reprojection_report(
            camera_model,
            world_pts_dict,
            image_points,
            max_mean_error_px=self.max_reprojection_error_px
        )

        # Step 4: Evaluate Jacobian Conditioning & Parameter Covariance
        J = opt_res.jac # (16, 8)
        # SVD of Jacobian to check observability
        _, s, _ = np.linalg.svd(J, full_matrices=False)
        cond_num = float(s[0] / max(s[-1], 1e-12))
        min_singular = float(s[-1])

        is_well_constrained = (cond_num < 5e4) and (min_singular > 1e-3)
        deficiency_msg = None
        if not is_well_constrained:
            deficiency_msg = (
                f"Jacobian condition number ({cond_num:.1e}) is elevated. "
                f"Weakest observable direction has singular value {min_singular:.6f}. "
                f"Consider verifying pitch corners and stump upright clicks."
            )

        # Residual variance: s2 = sum(r^2) / DOF
        n_obs = len(opt_res.fun)
        n_params = len(opt_res.x)
        dof = max(1, n_obs - n_params)
        res_var = float(np.sum(opt_res.fun ** 2) / dof)

        try:
            cov = np.linalg.inv(J.T @ J) * res_var
            param_sigmas = np.sqrt(np.maximum(0.0, np.diag(cov)))
        except np.linalg.LinAlgError:
            param_sigmas = np.ones(n_params) * 99.0

        sigma_log_f = param_sigmas[0]
        sigma_f_px = float(opt_f * sigma_log_f)
        sigma_r_deg = (
            float(np.degrees(param_sigmas[1])),
            float(np.degrees(param_sigmas[2])),
            float(np.degrees(param_sigmas[3]))
        )
        sigma_t_m = (
            float(param_sigmas[4]),
            float(param_sigmas[5]),
            float(param_sigmas[6])
        )

        # Optical center uncertainty: C = -R^T * t => cov(C) ~ R^T cov(t) R
        R_opt = camera_model.R
        cov_t = cov[4:7, 4:7]
        cov_c = R_opt.T @ cov_t @ R_opt
        sigmas_c = np.sqrt(np.maximum(0.0, np.diag(cov_c)))
        sigma_c_height = float(sigmas_c[2])
        sigma_c_dist = float(sigmas_c[1])

        uncertainty_report = CalibrationUncertaintyReport(
            sigma_focal_length_px=sigma_f_px,
            sigma_rotation_deg=sigma_r_deg,
            sigma_translation_m=sigma_t_m,
            sigma_camera_height_m=sigma_c_height,
            sigma_camera_distance_m=sigma_c_dist,
            jacobian_condition_number=cond_num,
            smallest_singular_value=min_singular,
            is_well_constrained=is_well_constrained,
            deficiency_report=deficiency_msg
        )

        reproj_report.uncertainty = uncertainty_report
        self.last_uncertainty_report = uncertainty_report

        if return_uncertainty:
            return camera_model, reproj_report, uncertainty_report
        return camera_model, reproj_report
