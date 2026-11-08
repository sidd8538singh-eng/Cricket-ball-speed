"""
Quality Control: Master Quality Assurance Engine for Cricket Bowling Speed.
Audits every delivery against 10 strict scientific verification gates.

THE 10 SCIENTIFIC VERIFICATION GATES:
1. Camera Stability Validation: Optical flow verification of zero camera pan/tilt/zoom jitter.
2. Calibration Validation: Perspective geometry, vanishing line consistency, valid intrinsics/extrinsics.
3. Reprojection Validation: Mean and maximum landmark reprojection error within tolerance.
4. Ball Observation Quality: Detection count, temporal continuity, and circularity/confidence quality.
5. Release Confidence: Bowler hand localization and kinematic separation confidence.
6. Depth Reconstruction Uncertainty: Condition number of ray-depth system and depth variance bounds.
7. Trajectory Fitting Residual: Convergence of 3D ballistic optimization and pixel residual RMSE.
8. Timestamp Validity: Strictly monotonic timestamps, positive delta time, and sufficient frame rate.
9. Speed Uncertainty: Formal 1-sigma covariance uncertainty and 95% confidence interval width.
10. Physical & Aerodynamic Plausibility: Athletic speed bounds (30-175 km/h) and drag consistency.

STRICT PRINCIPLE:
If ANY gate fails, the delivery is classified as NOT_MEASURED with an explicit rejection reason.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .rejection_reasons import RejectionReason
from calibration.reprojection import ReprojectionReport
from calibration.camera_model import CameraModel
from release.release_detector import ReleaseEvent
from reconstruction.trajectory_fitting import TrajectoryFitResult
from speed.release_speed import ReleaseSpeedResult
from speed.c1_c2_speed import C1C2SpeedResult


@dataclass
class QualityControlResult:
    """Encapsulates the complete 10-gate quality control audit."""
    status: str                         # "MEASURED" or "NOT_MEASURED"
    quality_score: float                # 0.0 to 100.0
    is_measured: bool
    rejection_reason: Optional[str] = None
    audit_checks: Dict[str, bool] = field(default_factory=dict)
    check_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    all_failures: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "quality_score": round(float(self.quality_score), 1),
            "is_measured": self.is_measured,
            "rejection_reason": self.rejection_reason,
            "audit_checks": self.audit_checks,
            "check_details": self.check_details,
            "all_failures": self.all_failures,
            "diagnostics": self.diagnostics,
        }


class QualityControlPipeline:
    """
    Evaluates every delivery against 10 strict scientific quality gates.
    """

    def __init__(
        self,
        min_ball_detections: int = 5,
        max_prediction_gap_frames: int = 3,
        min_observation_quality: float = 0.35,
        max_reprojection_rmse_px: float = 5.5,
        min_release_confidence: float = 0.50,
        min_release_height_m: float = 1.30,
        max_release_height_m: float = 2.90,
        min_depth_observability: float = 0.20,
        max_trajectory_rmse_px: float = 5.5,
        max_speed_uncertainty_kmh: float = 8.5,
        min_speed_kmh: float = 30.0,
        max_speed_kmh: float = 175.0,
        min_fps: float = 59.0,
        max_c1_c2_overspeed_kmh: float = 10.0
    ):
        self.min_dets = min_ball_detections
        self.max_pred_gap = max_prediction_gap_frames
        self.min_obs_quality = min_observation_quality
        self.max_reproj = max_reprojection_rmse_px
        self.min_release_conf = min_release_confidence
        self.min_release_height = min_release_height_m
        self.max_release_height = max_release_height_m
        self.min_depth_obs = min_depth_observability
        self.max_traj_rmse = max_trajectory_rmse_px
        self.max_speed_unc = max_speed_uncertainty_kmh
        self.min_speed = min_speed_kmh
        self.max_speed = max_speed_kmh
        self.min_fps = min_fps
        self.max_c1_c2_overspeed = max_c1_c2_overspeed_kmh

    def audit(
        self,
        video_fps: float,
        is_camera_stable: bool,
        reprojection_report: Optional[ReprojectionReport],
        num_ball_detections: int,
        release_event: Optional[ReleaseEvent],
        trajectory_result: Optional[TrajectoryFitResult],
        release_speed_result: Optional[ReleaseSpeedResult],
        c1_c2_result: Optional[C1C2SpeedResult] = None,
        camera_model: Optional[CameraModel] = None,
        is_timestamp_valid: bool = True,
        max_consecutive_missing_frames: int = 0,
        mean_observation_quality: Optional[float] = None
    ) -> QualityControlResult:
        """
        Executes complete quality control audit across all 10 gates.
        """
        checks: Dict[str, bool] = {}
        details: Dict[str, Dict[str, Any]] = {}
        failures: List[str] = []

        # =====================================================================
        # Gate 1: Camera Stability Validation
        # =====================================================================
        checks["camera_stability"] = bool(is_camera_stable)
        details["camera_stability"] = {
            "passed": bool(is_camera_stable),
            "measured": "STABLE" if is_camera_stable else "UNSTABLE",
            "required": "STABLE",
        }
        if not is_camera_stable:
            failures.append(RejectionReason.CAMERA_UNSTABLE.value)

        # =====================================================================
        # Gate 2: Calibration Validation
        # =====================================================================
        calib_ok = True
        calib_msg = "VALID"
        if camera_model is not None:
            model_valid = getattr(camera_model, "is_valid", True)
            if not model_valid or camera_model.fx <= 0 or camera_model.fy <= 0:
                calib_ok = False
                calib_msg = "INVALID_CAMERA_PARAMETERS"
        elif reprojection_report is None:
            calib_ok = False
            calib_msg = "CALIBRATION_MODEL_MISSING"

        checks["calibration_validity"] = calib_ok
        details["calibration_validity"] = {
            "passed": calib_ok,
            "measured": calib_msg,
            "required": "VALID",
        }
        if not calib_ok:
            failures.append(RejectionReason.CAMERA_CALIBRATION_FAILED.value)

        # =====================================================================
        # Gate 3: Reprojection Validation
        # =====================================================================
        if reprojection_report is not None:
            reproj_err = reprojection_report.mean_error_px
            reproj_ok = reprojection_report.is_acceptable and (reproj_err <= self.max_reproj)
            checks["reprojection_accuracy"] = reproj_ok
            details["reprojection_accuracy"] = {
                "passed": reproj_ok,
                "measured_mean_rmse_px": round(reproj_err, 3),
                "threshold_max_px": self.max_reproj,
            }
            if not reproj_ok:
                failures.append(
                    f"{RejectionReason.REPROJECTION_ERROR_HIGH.value} (Mean error: {reproj_err:.2f}px > {self.max_reproj:.2f}px)"
                )
        else:
            checks["reprojection_accuracy"] = False
            details["reprojection_accuracy"] = {
                "passed": False,
                "measured": "REPORT_MISSING",
                "threshold_max_px": self.max_reproj,
            }
            failures.append(RejectionReason.CAMERA_CALIBRATION_FAILED.value)

        # =====================================================================
        # Gate 4: Ball Observation Quality
        # =====================================================================
        count_ok = num_ball_detections >= self.min_dets
        gap_ok = max_consecutive_missing_frames <= self.max_pred_gap
        qual_ok = True
        if mean_observation_quality is not None:
            qual_ok = mean_observation_quality >= self.min_obs_quality

        obs_gate_passed = count_ok and gap_ok and qual_ok
        checks["observation_quality"] = obs_gate_passed
        details["observation_quality"] = {
            "passed": obs_gate_passed,
            "detections_count": num_ball_detections,
            "min_detections_required": self.min_dets,
            "max_consecutive_missing_frames": max_consecutive_missing_frames,
            "max_allowed_gap_frames": self.max_pred_gap,
            "mean_quality_score": round(mean_observation_quality, 2) if mean_observation_quality is not None else None,
        }
        if not count_ok:
            failures.append(
                f"{RejectionReason.INSUFFICIENT_BALL_DETECTIONS.value} (Found {num_ball_detections} < {self.min_dets})"
            )
        elif not gap_ok:
            failures.append(
                f"{RejectionReason.LONG_PREDICTION_GAP.value} (Gap: {max_consecutive_missing_frames} > {self.max_pred_gap})"
            )
        elif not qual_ok:
            failures.append(
                f"{RejectionReason.OBSERVATION_QUALITY_LOW.value} (Quality: {mean_observation_quality:.2f} < {self.min_obs_quality:.2f})"
            )

        # =====================================================================
        # Gate 5: Release Confidence
        # =====================================================================
        rel_ok = False
        rel_conf = 0.0
        rel_h = None
        if release_event is not None and release_event.is_valid:
            rel_conf = float(release_event.confidence)
            conf_ok = rel_conf >= self.min_release_conf
            height_ok = True
            # Check 3D release height if trajectory fit provided
            if trajectory_result is not None and trajectory_result.is_valid:
                rel_h = float(trajectory_result.z0)
                height_ok = (self.min_release_height <= rel_h <= self.max_release_height)

            rel_ok = conf_ok and height_ok
            checks["release_confidence"] = rel_ok
            details["release_confidence"] = {
                "passed": rel_ok,
                "confidence": round(rel_conf, 3),
                "min_confidence_required": self.min_release_conf,
                "release_height_m": round(rel_h, 3) if rel_h is not None else None,
                "height_bounds_m": [self.min_release_height, self.max_release_height],
            }
            if not conf_ok:
                failures.append(
                    f"{RejectionReason.RELEASE_CONFIDENCE_LOW.value} ({rel_conf:.2f} < {self.min_release_conf:.2f})"
                )
            elif not height_ok:
                failures.append(
                    f"{RejectionReason.RELEASE_HEIGHT_IMPLAUSIBLE.value} ({rel_h:.2f}m outside [{self.min_release_height:.2f}m, {self.max_release_height:.2f}m])"
                )
        else:
            checks["release_confidence"] = False
            details["release_confidence"] = {
                "passed": False,
                "confidence": 0.0,
                "min_confidence_required": self.min_release_conf,
            }
            failures.append(RejectionReason.RELEASE_NOT_FOUND.value)

        # =====================================================================
        # Gate 6: Depth Reconstruction Uncertainty
        # =====================================================================
        depth_ok = False
        depth_score = 0.0
        if trajectory_result is not None and trajectory_result.is_valid:
            depth_score = getattr(trajectory_result, "depth_observability_score", 1.0)
            depth_ok = depth_score >= self.min_depth_obs
            checks["depth_uncertainty"] = depth_ok
            details["depth_uncertainty"] = {
                "passed": depth_ok,
                "depth_observability_score": round(float(depth_score), 3),
                "min_required_score": self.min_depth_obs,
            }
            if not depth_ok:
                failures.append(
                    f"{RejectionReason.DEPTH_UNCERTAINTY_HIGH.value} (Observability score: {depth_score:.2f} < {self.min_depth_obs:.2f})"
                )
        else:
            checks["depth_uncertainty"] = False
            details["depth_uncertainty"] = {"passed": False, "measured": "NO_TRAJECTORY_FIT"}
            failures.append(RejectionReason.DEPTH_UNCERTAINTY_HIGH.value)

        # =====================================================================
        # Gate 7: Trajectory Fitting Residual
        # =====================================================================
        traj_ok = False
        if trajectory_result is not None:
            converged = getattr(trajectory_result, "is_converged", True)
            valid = getattr(trajectory_result, "is_valid", True)
            rmse = getattr(trajectory_result, "rmse_pixels", float("inf"))
            traj_ok = converged and valid and (rmse <= self.max_traj_rmse)
            checks["trajectory_residual"] = traj_ok
            details["trajectory_residual"] = {
                "passed": traj_ok,
                "is_converged": converged,
                "rmse_pixels": round(float(rmse), 3),
                "threshold_max_px": self.max_traj_rmse,
            }
            if not converged or not valid:
                failures.append(RejectionReason.TRAJECTORY_DIVERGED.value)
            elif rmse > self.max_traj_rmse:
                failures.append(
                    f"{RejectionReason.TRAJECTORY_RESIDUAL_HIGH.value} (RMSE: {rmse:.2f}px > {self.max_traj_rmse:.2f}px)"
                )
        else:
            checks["trajectory_residual"] = False
            details["trajectory_residual"] = {"passed": False, "measured": "NO_TRAJECTORY_RESULT"}
            failures.append(RejectionReason.TRAJECTORY_DIVERGED.value)

        # =====================================================================
        # Gate 8: Timestamp Validity & Frame Rate
        # =====================================================================
        fps_ok = video_fps >= self.min_fps
        ts_ok = is_timestamp_valid and fps_ok
        checks["timestamp_validity"] = ts_ok
        details["timestamp_validity"] = {
            "passed": ts_ok,
            "is_timestamp_valid": is_timestamp_valid,
            "video_fps": round(float(video_fps), 1),
            "min_fps_required": self.min_fps,
        }
        if not is_timestamp_valid:
            failures.append(RejectionReason.TIMESTAMP_IRREGULAR.value)
        if not fps_ok:
            failures.append(
                f"{RejectionReason.FRAMERATE_TOO_LOW.value} (FPS: {video_fps:.1f} < {self.min_fps:.1f})"
            )

        # =====================================================================
        # Gate 9: Speed Uncertainty
        # =====================================================================
        unc_ok = False
        if release_speed_result is not None and release_speed_result.is_valid:
            unc = float(release_speed_result.uncertainty_km_h)
            unc_ok = unc <= self.max_speed_unc
            checks["speed_uncertainty"] = unc_ok
            details["speed_uncertainty"] = {
                "passed": unc_ok,
                "speed_uncertainty_kmh": round(unc, 2),
                "max_uncertainty_threshold_kmh": self.max_speed_unc,
            }
            if not unc_ok:
                failures.append(
                    f"{RejectionReason.SPEED_UNCERTAINTY_EXCESSIVE.value} (+- {unc:.1f} km/h > +- {self.max_speed_unc:.1f} km/h)"
                )
        else:
            checks["speed_uncertainty"] = False
            details["speed_uncertainty"] = {"passed": False, "measured": "NO_VALID_SPEED_RESULT"}
            failures.append(RejectionReason.SPEED_NOT_COMPUTED.value)

        # =====================================================================
        # Gate 10: Physical & Aerodynamic Plausibility
        # =====================================================================
        phys_ok = False
        if release_speed_result is not None and release_speed_result.is_valid:
            spd = float(release_speed_result.speed_km_h)
            range_ok = self.min_speed <= spd <= self.max_speed
            # Check down-pitch forward motion (+VY)
            vy_ok = release_speed_result.velocity_vector_m_s[1] > 0.0

            # Aerodynamic consistency with C1-C2 baseline
            aero_ok = True
            c1_c2_spd = None
            if c1_c2_result is not None and c1_c2_result.is_valid and c1_c2_result.status == "MEASURED":
                c1_c2_spd = float(c1_c2_result.average_speed_km_h)
                if c1_c2_spd > (spd + self.max_c1_c2_overspeed):
                    aero_ok = False

            phys_ok = range_ok and vy_ok and aero_ok
            checks["physical_plausibility"] = phys_ok
            details["physical_plausibility"] = {
                "passed": phys_ok,
                "release_speed_kmh": round(spd, 2),
                "speed_bounds_kmh": [self.min_speed, self.max_speed],
                "vy_forward_motion": vy_ok,
                "c1_c2_speed_kmh": round(c1_c2_spd, 2) if c1_c2_spd is not None else None,
                "aerodynamic_drag_consistent": aero_ok,
            }
            if not range_ok:
                failures.append(f"{RejectionReason.SPEED_OUT_OF_RANGE.value} ({spd:.1f} km/h outside [{self.min_speed:.1f}, {self.max_speed:.1f}])")
            elif not vy_ok:
                failures.append(RejectionReason.BALL_DIRECTION_INVALID.value)
            elif not aero_ok:
                failures.append(
                    f"{RejectionReason.AERODYNAMIC_INCONSISTENCY.value} (C1-C2 speed {c1_c2_spd:.1f} km/h > release {spd:.1f} km/h + {self.max_c1_c2_overspeed:.1f} km/h)"
                )
        else:
            checks["physical_plausibility"] = False
            details["physical_plausibility"] = {"passed": False, "measured": "NO_VALID_SPEED_RESULT"}
            failures.append(RejectionReason.SPEED_NOT_COMPUTED.value)

        # =====================================================================
        # Overall Decision: MEASURED vs NOT_MEASURED
        # =====================================================================
        is_measured = (len(failures) == 0) and all(checks.values())
        status = "MEASURED" if is_measured else "NOT_MEASURED"
        primary_rejection = failures[0] if failures else None

        # Composite Quality Score (0 to 100)
        passed_count = sum(1 for v in checks.values() if v)
        total_count = len(checks)
        score = (passed_count / max(1, total_count)) * 100.0
        if not is_measured:
            score = min(score, 49.0)  # Capped below passing threshold for unmeasured deliveries

        return QualityControlResult(
            status=status,
            quality_score=score,
            is_measured=is_measured,
            rejection_reason=primary_rejection,
            audit_checks=checks,
            check_details=details,
            all_failures=failures,
            diagnostics={
                "total_gates": total_count,
                "passed_gates": passed_count,
                "failed_gates": total_count - passed_count,
            }
        )
