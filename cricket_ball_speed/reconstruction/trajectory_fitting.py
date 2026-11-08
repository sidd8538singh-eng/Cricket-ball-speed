r"""
Trajectory Fitting: Robust Weighted 3D Ballistic Trajectory Optimization for Single-Camera Setups.
Solves for [X0, Y0, Z0, VX0, VY0, VZ0] at release time t0 by minimizing ray reprojection
errors and heteroskedastic apparent-diameter depth cues under physical gravity.

====================================================================================================
THE SCIENTIFIC LIMITATIONS OF SINGLE-CAMERA 3D RECONSTRUCTION
====================================================================================================
1. Monocular Ray Ambiguity:
   A single camera pixel (u, v) back-projects to an infinite 1D ray in 3D space:
       r(lambda) = C + lambda * d(u, v),  lambda > 0
   A single observation possesses ZERO depth information on its own.

2. Heteroskedastic Apparent Diameter:
   The apparent ball diameter in pixels d_px(Z) scales inversely with distance Z:
       d_px(Z) = (f * D_phys) / Z  ==>  Z = (f * D_phys) / d_px
   The derivative dZ / d(d_px) = - (f * D_phys) / (d_px^2).
   At release (Z ~ 2.5m), d_px ~ 25-35 px; a 1px error changes Z by only ~0.10 m.
   At the batsman (Z ~ 22m), d_px drops to 3-5 px; a 1px error shifts Z by > 5.5 metres!
   Therefore, apparent diameter is treated strictly as a noisy, heteroskedastic depth prior,
   weighted inversely by variance (sigma_Z^2) rather than as a rigid depth coordinate.

3. Motion Foreshortening:
   When a camera is placed directly behind the bowler, bowling flight occurs largely along
   the optical axis (down-pitch +Y). Kinematic constraints (known gravity g = 9.80665 m/s^2,
   smooth continuous acceleration, and temporal baseline) allow resolving depth, but if the
   number of observed frames is insufficient (< 4), depth observability collapses.

4. Non-Forcing Principle:
   If depth is not sufficiently observable or if the condition number of the Hessian indicates
   an unconstrained solution, the system MUST return NOT_MEASURED rather than forcing a fake fit.
====================================================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import logging
import numpy as np
from scipy.optimize import least_squares

from calibration.camera_model import CameraModel
from ball_processing.ball_observations import BallObservation
from ball_processing.apparent_diameter import CRICKET_BALL_DIAMETER_M
from .ball_3d import Ball3DPoint
from .uncertainty import UncertaintyEstimator
from .depth_estimation import RayDepthEstimator

logger = logging.getLogger("CricketBallSpeed.TrajectoryFitting")


@dataclass
class TrajectoryFitResult:
    """Encapsulates the 6-DOF ballistic trajectory fit with provenance and audit metrics."""
    t0: float
    x0: float
    y0: float
    z0: float
    vx0: float
    vy0: float
    vz0: float
    rmse_pixels: float
    uncertainty_speed_kmh: float
    covariance_matrix: np.ndarray
    fitted_3d_points: List[Ball3DPoint]
    is_converged: bool
    is_valid: bool = True
    status: str = "MEASURED"                    # "MEASURED" or "NOT_MEASURED"
    rejection_reason: Optional[str] = None
    depth_observability_score: float = 1.0       # Higher is better, < 0.2 indicates ill-conditioned depth
    outliers_rejected_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def initial_velocity_vector(self) -> np.ndarray:
        return np.array([self.vx0, self.vy0, self.vz0], dtype=np.float64)

    @property
    def release_speed_m_s(self) -> float:
        return float(np.linalg.norm(self.initial_velocity_vector))

    @property
    def release_speed_km_h(self) -> float:
        return float(self.release_speed_m_s * 3.6)

    def position_at(self, t: float, g: float = 9.80665) -> np.ndarray:
        """Evaluates 3D position [X, Y, Z] at time t."""
        dt = t - self.t0
        x = self.x0 + self.vx0 * dt
        y = self.y0 + self.vy0 * dt
        z = self.z0 + self.vz0 * dt - 0.5 * g * (dt ** 2)
        return np.array([x, y, z], dtype=np.float64)

    def velocity_at(self, t: float, g: float = 9.80665) -> np.ndarray:
        """Evaluates 3D velocity [VX, VY, VZ] at time t."""
        dt = t - self.t0
        vx = self.vx0
        vy = self.vy0
        vz = self.vz0 - g * dt
        return np.array([vx, vy, vz], dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t0_sec": round(self.t0, 5),
            "c1_3d_m": [round(float(self.x0), 3), round(float(self.y0), 3), round(float(self.z0), 3)],
            "v0_3d_m_s": [round(float(self.vx0), 3), round(float(self.vy0), 3), round(float(self.vz0), 3)],
            "release_speed_kmh": round(self.release_speed_km_h, 2),
            "uncertainty_speed_kmh": round(self.uncertainty_speed_kmh, 2),
            "rmse_pixels": round(self.rmse_pixels, 3),
            "is_converged": self.is_converged,
            "is_valid": self.is_valid,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "depth_observability_score": round(self.depth_observability_score, 3),
            "outliers_rejected_count": self.outliers_rejected_count,
            "fitted_points_count": len(self.fitted_3d_points),
        }


class TrajectoryFitter:
    """
    Fits 3D ballistic trajectory to a sequence of image observations.
    Physical ballistic model:
      X(t) = X0 + VX0 * (t - t0)
      Y(t) = Y0 + VY0 * (t - t0)
      Z(t) = Z0 + VZ0 * (t - t0) - 0.5 * g * (t - t0)^2
    """

    def __init__(
        self,
        camera_model: CameraModel,
        gravity_m_s2: float = 9.80665,
        ball_diameter_m: float = CRICKET_BALL_DIAMETER_M,
        pixel_sigma: float = 1.2,
        base_depth_weight: float = 0.15,
        max_rmse_pixels_threshold: float = 3.5,
        max_speed_uncertainty_threshold: float = 8.5
    ):
        self.camera = camera_model
        self.g = float(gravity_m_s2)
        self.ball_diameter_m = float(ball_diameter_m)
        self.pixel_sigma = float(pixel_sigma)
        self.base_depth_weight = float(base_depth_weight)
        self.max_rmse_thresh = float(max_rmse_pixels_threshold)
        self.max_unc_thresh = float(max_speed_uncertainty_threshold)

        self.focal_length = float((camera_model.fx + camera_model.fy) / 2.0)
        self.depth_estimator = RayDepthEstimator(
            camera_model, ball_diameter_m=self.ball_diameter_m, pixel_diameter_noise_std=0.75
        )

    def fit_trajectory(
        self,
        observations: List[BallObservation],
        t0_release: float,
        initial_guess: Optional[np.ndarray] = None
    ) -> TrajectoryFitResult:
        """
        Fits 6-DOF ballistic state [X0, Y0, Z0, VX0, VY0, VZ0] at t0_release.
        Enforces outlier rejection, heteroskedastic depth weighting, and depth observability checks.
        """
        # Minimum observation check
        if len(observations) < 4:
            logger.warning(f"Insufficient observations for 3D trajectory fit: {len(observations)} < 4.")
            return self._build_unmeasured_result(
                t0_release,
                f"INSUFFICIENT_OBSERVATIONS: Found only {len(observations)} frames (minimum 4 required for 3D fit)."
            )

        # Filter observations starting at or slightly after release
        obs_valid = [o for o in observations if o.timestamp_sec >= (t0_release - 0.02)]
        if len(obs_valid) < 4:
            obs_valid = list(observations)

        # Extract data vectors
        times = np.array([o.timestamp_sec - t0_release for o in obs_valid], dtype=np.float64)
        targets_u = np.array([o.center[0] for o in obs_valid], dtype=np.float64)
        targets_v = np.array([o.center[1] for o in obs_valid], dtype=np.float64)
        confs = np.array([max(0.15, min(1.0, o.confidence)) for o in obs_valid], dtype=np.float64)

        # Compute heteroskedastic depth priors and variances
        depth_priors = []
        depth_weights = []
        for o in obs_valid:
            d_m, var_m2 = self.depth_estimator.estimate_depth_and_variance(o.apparent_diameter_px)
            std_m = max(0.1, np.sqrt(var_m2))
            depth_priors.append(d_m)
            # Inverse-variance weighting: w_depth ~ 1 / sigma_Z
            depth_weights.append(self.base_depth_weight / std_m)

        depth_priors = np.array(depth_priors, dtype=np.float64)
        depth_weights = np.array(depth_weights, dtype=np.float64)

        # Compute initial guess
        if initial_guess is None:
            # Estimate Y0 from earliest reliable depth cue
            cam_dist_behind = getattr(self.camera, "camera_distance_behind_near_stumps_m", 2.0)
            y0_est = max(0.0, float(depth_priors[0] - cam_dist_behind))
            total_dt = max(1e-3, times[-1] - times[0])
            vy_est = max(18.0, min(48.0, (depth_priors[-1] - depth_priors[0]) / total_dt))
            p0 = np.array([0.0, y0_est, 2.15, 0.0, vy_est, -1.8], dtype=np.float64)
        else:
            p0 = np.array(initial_guess, dtype=np.float64)

        cam_center = self.camera.optical_centre

        def objective(p: np.ndarray, active_indices: Optional[List[int]] = None) -> np.ndarray:
            x0, y0, z0, vx0, vy0, vz0 = p
            residuals = []

            eval_indices = active_indices if active_indices is not None else range(len(obs_valid))

            for idx in eval_indices:
                dt = times[idx]
                u_true = targets_u[idx]
                v_true = targets_v[idx]
                conf = confs[idx]
                depth_prior = depth_priors[idx]
                w_d = depth_weights[idx]

                # Ballistic 3D state
                x = x0 + vx0 * dt
                y = y0 + vy0 * dt
                z = z0 + vz0 * dt - 0.5 * self.g * (dt ** 2)

                # Reprojection to image plane
                u_proj, v_proj = self.camera.project_point(np.array([x, y, z]))

                # Weighted pixel residuals
                w_px = (conf / self.pixel_sigma)
                residuals.append((u_proj - u_true) * w_px)
                residuals.append((v_proj - v_true) * w_px)

                # Heteroskedastic depth residual
                dist_cam = float(np.linalg.norm(np.array([x, y, z]) - cam_center))
                residuals.append((dist_cam - depth_prior) * w_d)

            return np.array(residuals, dtype=np.float64)

        # Parameter bounds: [X0, Y0, Z0, VX0, VY0, VZ0]
        # Vy between 10.0 m/s (~36 km/h slow spin) and 52.0 m/s (~187 km/h extreme pace)
        bounds = (
            [-2.5, -2.5, 0.5, -15.0, 10.0, -15.0],
            [+2.5, +7.5, 3.2, +15.0, 52.0, +10.0]
        )

        # Ensure initial guess is strictly within bounds
        p0 = np.clip(p0, np.array(bounds[0]) + 1e-2, np.array(bounds[1]) - 1e-2)

        # Pass 1: Initial robust fit
        res1 = least_squares(
            lambda p: objective(p),
            p0,
            method="trf",
            loss="huber",
            f_scale=2.0,
            bounds=bounds,
            max_nfev=400
        )

        p_opt = res1.x
        active_indices = list(range(len(obs_valid)))
        outliers_rejected = 0

        # Pass 2: Outlier rejection if residual is excessive
        per_point_errors = []
        for idx in range(len(obs_valid)):
            dt = times[idx]
            pos = np.array([
                p_opt[0] + p_opt[3] * dt,
                p_opt[1] + p_opt[4] * dt,
                p_opt[2] + p_opt[5] * dt - 0.5 * self.g * (dt ** 2)
            ])
            up, vp = self.camera.project_point(pos)
            err = float(np.hypot(up - targets_u[idx], vp - targets_v[idx]))
            per_point_errors.append(err)

        mean_err = np.mean(per_point_errors)
        std_err = np.std(per_point_errors)
        outlier_thresh = max(4.0, mean_err + 2.5 * std_err)

        inlier_indices = [idx for idx, err in enumerate(per_point_errors) if err <= outlier_thresh]
        if len(inlier_indices) < len(obs_valid) and len(inlier_indices) >= 4:
            outliers_rejected = len(obs_valid) - len(inlier_indices)
            logger.info(f"Trajectory fit: rejected {outliers_rejected} observation outliers.")
            res2 = least_squares(
                lambda p: objective(p, active_indices=inlier_indices),
                p_opt,
                method="trf",
                loss="huber",
                f_scale=2.0,
                bounds=bounds,
                max_nfev=300
            )
            p_opt = res2.x
            res_final = res2
            active_indices = inlier_indices
        else:
            res_final = res1

        x0, y0, z0, vx0, vy0, vz0 = p_opt

        # Compute final reprojection residuals and 3D point provenance
        final_errors = []
        fitted_points: List[Ball3DPoint] = []

        for idx, o in enumerate(obs_valid):
            dt = o.timestamp_sec - t0_release
            pos_3d = np.array([
                x0 + vx0 * dt,
                y0 + vy0 * dt,
                z0 + vz0 * dt - 0.5 * self.g * (dt ** 2)
            ], dtype=np.float64)
            vel_3d = np.array([vx0, vy0, vz0 - self.g * dt], dtype=np.float64)

            u_p, v_p = self.camera.project_point(pos_3d)
            err = float(np.hypot(u_p - o.center[0], v_p - o.center[1]))
            final_errors.append(err)

            is_inlier = (idx in active_indices)
            dist_cam = float(np.linalg.norm(pos_3d - cam_center))

            fitted_points.append(Ball3DPoint(
                frame_index=o.frame_index,
                timestamp_sec=o.timestamp_sec,
                position_xyz=pos_3d,
                uncertainty_xyz=np.array([0.03, 0.12, 0.03]),
                source="ray_fitted",
                point_type="OBSERVED" if is_inlier else "PREDICTED",
                is_real_detection=is_inlier,
                reprojection_error_px=err,
                depth_from_camera_m=dist_cam,
                depth_uncertainty_m=float(np.sqrt(1.0 / max(1e-4, depth_weights[idx]))),
                observation_confidence=float(o.confidence),
                velocity_xyz=vel_3d
            ))

        inlier_errors = [final_errors[idx] for idx in active_indices]
        rmse_px = float(np.sqrt(np.mean(np.array(inlier_errors) ** 2))) if inlier_errors else 0.0

        # Parameter Covariance & Velocity Uncertainty
        cov = UncertaintyEstimator.compute_parameter_covariance(res_final.jac, res_final.fun)
        sigmas = np.sqrt(np.maximum(0.0, np.diag(cov)))
        sigma_vx, sigma_vy, sigma_vz = sigmas[3], sigmas[4], sigmas[5]
        uncertainty_speed = UncertaintyEstimator.compute_speed_uncertainty(
            vx0, vy0, vz0, sigma_vx, sigma_vy, sigma_vz
        )

        # Depth Observability Check
        # Evaluate Jacobian conditioning for longitudinal depth parameter VY0
        try:
            jt_j = res_final.jac.T @ res_final.jac
            eigenvals = np.linalg.eigvalsh(jt_j)
            min_eig, max_eig = float(np.min(eigenvals)), float(np.max(eigenvals))
            cond_num = max_eig / max(1e-12, min_eig)
            obs_score = float(1.0 / (1.0 + np.log10(max(1.0, cond_num))))
        except Exception:
            obs_score = 0.1

        # Strict Quality Assessment
        is_valid = True
        rejection_reason = None

        if not res_final.success:
            is_valid = False
            rejection_reason = "OPTIMIZATION_FAILED: Trajectory solver did not converge."
        elif rmse_px > self.max_rmse_thresh:
            is_valid = False
            rejection_reason = (
                f"POOR_TRAJECTORY_FIT: Reprojection RMSE ({rmse_px:.2f}px) exceeds "
                f"threshold ({self.max_rmse_thresh:.2f}px)."
            )
        elif uncertainty_speed > self.max_unc_thresh:
            is_valid = False
            rejection_reason = (
                f"DEPTH_UNOBSERVABLE: Speed uncertainty (+/- {uncertainty_speed:.1f} km/h) "
                f"exceeds tolerance ({self.max_unc_thresh:.1f} km/h). Monocular depth cues unconstrained."
            )
        elif obs_score < 0.05:
            is_valid = False
            rejection_reason = "DEPTH_UNOBSERVABLE: Camera ray geometry and baseline insufficient to resolve depth."

        status = "MEASURED" if is_valid else "NOT_MEASURED"
        if not is_valid:
            logger.warning(f"3D Trajectory rejected: {rejection_reason}")

        return TrajectoryFitResult(
            t0=t0_release,
            x0=x0,
            y0=y0,
            z0=z0,
            vx0=vx0,
            vy0=vy0,
            vz0=vz0,
            rmse_pixels=rmse_px,
            uncertainty_speed_kmh=uncertainty_speed,
            covariance_matrix=cov,
            fitted_3d_points=fitted_points,
            is_converged=res_final.success,
            is_valid=is_valid,
            status=status,
            rejection_reason=rejection_reason,
            depth_observability_score=obs_score,
            outliers_rejected_count=outliers_rejected
        )

    def _build_unmeasured_result(self, t0: float, reason: str) -> TrajectoryFitResult:
        """Constructs a NOT_MEASURED TrajectoryFitResult when observations are insufficient."""
        return TrajectoryFitResult(
            t0=t0,
            x0=0.0,
            y0=0.0,
            z0=0.0,
            vx0=0.0,
            vy0=0.0,
            vz0=0.0,
            rmse_pixels=0.0,
            uncertainty_speed_kmh=0.0,
            covariance_matrix=np.eye(6),
            fitted_3d_points=[],
            is_converged=False,
            is_valid=False,
            status="NOT_MEASURED",
            rejection_reason=reason,
            depth_observability_score=0.0,
            outliers_rejected_count=0
        )
