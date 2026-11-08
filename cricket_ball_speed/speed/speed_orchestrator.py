"""
Bowling Speed Orchestrator: Unified calculation pipeline for cricket bowling delivery speeds.
Coordinates instantaneous 3D release speed, dynamic C1-C2 average speed, frame-wise speed profiling,
and scientific aerodynamic validation.

ARCHITECTURAL PRINCIPLES:
1. Primary output is instantaneous 3D speed at estimated release time (t0):
      speed = sqrt(VX^2 + VY^2 + VZ^2) * 3.6 km/h.
2. C1-C2 average speed is dynamically chosen targeting 5 to 8 metres from release.
3. Frame-wise telemetry tracks the velocity vector and deceleration throughout flight.
4. Returns NOT_MEASURED if the required observations, trajectory fit, or physics checks are unreliable.
5. Strict: ZERO raw pixel displacement calculations, ZERO arbitrary multipliers,
   ZERO radar/filename speed inputs, ZERO release height forcing.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from reconstruction.trajectory_fitting import TrajectoryFitResult
from reconstruction.ball_3d import Ball3DPoint
from release.release_detector import ReleaseEvent
from .release_speed import compute_release_speed, ReleaseSpeedResult
from .c1_c2_speed import C1C2SpeedCalculator, C1C2SpeedResult
from .framewise_speed import FramewiseSpeedCalculator, FramewiseSpeedReport
from .speed_validator import SpeedValidator


@dataclass
class BowlingSpeedOutput:
    """Complete bowling speed calculation package with all scientific metrics and logs."""
    # Primary broadcast metric
    release_speed_km_h: float
    release_speed_m_s: float
    release_uncertainty_km_h: float
    ci_95_km_h: Tuple[float, float]
    velocity_vector_m_s: Tuple[float, float, float]
    release_position_m: Tuple[float, float, float]
    release_timestamp_sec: float

    # Flight geometry
    elevation_deg: float
    azimuth_deg: float

    # C1-C2 baseline speed
    c1_c2_speed_km_h: Optional[float]
    c1_c2_distance_m: Optional[float]
    c1_c2_dt_sec: Optional[float]
    c2_frame_index: Optional[int]

    # Status and audit
    is_measured: bool
    status: str                         # "MEASURED" or "NOT_MEASURED"
    rejection_reason: Optional[str]

    # Components
    release_speed_result: ReleaseSpeedResult
    c1_c2_result: Optional[C1C2SpeedResult]
    framewise_report: Optional[FramewiseSpeedReport]
    validation_audit: Dict[str, Any]
    master_logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "is_measured": self.is_measured,
            "rejection_reason": self.rejection_reason,
            "release_speed_km_h": round(float(self.release_speed_km_h), 2),
            "release_speed_m_s": round(float(self.release_speed_m_s), 2),
            "release_uncertainty_km_h": round(float(self.release_uncertainty_km_h), 2),
            "ci_95_km_h": [round(float(c), 2) for c in self.ci_95_km_h],
            "velocity_vector_m_s": [round(float(v), 3) for v in self.velocity_vector_m_s],
            "release_position_m": [round(float(p), 3) for p in self.release_position_m],
            "release_timestamp_sec": round(float(self.release_timestamp_sec), 5),
            "elevation_deg": round(float(self.elevation_deg), 2),
            "azimuth_deg": round(float(self.azimuth_deg), 2),
            "c1_c2_speed_km_h": round(float(self.c1_c2_speed_km_h), 2) if self.c1_c2_speed_km_h is not None else None,
            "c1_c2_distance_m": round(float(self.c1_c2_distance_m), 3) if self.c1_c2_distance_m is not None else None,
            "c1_c2_dt_sec": round(float(self.c1_c2_dt_sec), 5) if self.c1_c2_dt_sec is not None else None,
            "c2_frame_index": self.c2_frame_index,
            "c1_c2_details": self.c1_c2_result.to_dict() if self.c1_c2_result else None,
            "framewise_summary": self.framewise_report.to_dict() if self.framewise_report else None,
            "validation_audit": self.validation_audit,
            "master_logs": self.master_logs,
        }


class BowlingSpeedEngine:
    """
    High-level engine that computes the complete set of cricket bowling speeds.
    """

    def __init__(
        self,
        c1_c2_target_distance_m: float = 6.5,
        c1_c2_min_distance_m: float = 4.0,
        c1_c2_max_distance_m: float = 9.0,
        min_plausible_speed_kmh: float = 30.0,
        max_plausible_speed_kmh: float = 175.0,
        max_uncertainty_kmh: float = 8.5
    ):
        self.c1_c2_calculator = C1C2SpeedCalculator(
            target_distance_m=c1_c2_target_distance_m,
            min_distance_m=c1_c2_min_distance_m,
            max_distance_m=c1_c2_max_distance_m
        )
        self.validator = SpeedValidator(
            min_plausible_speed_kmh=min_plausible_speed_kmh,
            max_plausible_speed_kmh=max_plausible_speed_kmh,
            max_uncertainty_kmh=max_uncertainty_kmh
        )

    def compute(
        self,
        trajectory_fit: Optional[TrajectoryFitResult],
        fitted_3d_points: Optional[List[Ball3DPoint]] = None,
        release_event: Optional[ReleaseEvent] = None
    ) -> BowlingSpeedOutput:
        """
        Executes complete speed computation workflow.
        """
        master_logs: List[str] = []
        master_logs.append("=== CRICKET BOWLING SPEED CALCULATION ENGINE ===")

        # 1. Compute instantaneous 3D release speed
        release_speed_res = compute_release_speed(trajectory_fit)
        master_logs.extend(release_speed_res.calculation_logs)

        # If release speed could not be computed, abort with NOT_MEASURED
        if not release_speed_res.is_valid or release_speed_res.status == "NOT_MEASURED":
            reason = release_speed_res.rejection_reason or "TRAJECTORY_FIT_UNAVAILABLE"
            master_logs.append(f"CALCULATION STATUS: NOT_MEASURED -> {reason}")
            return BowlingSpeedOutput(
                release_speed_km_h=0.0,
                release_speed_m_s=0.0,
                release_uncertainty_km_h=0.0,
                ci_95_km_h=(0.0, 0.0),
                velocity_vector_m_s=(0.0, 0.0, 0.0),
                release_position_m=(0.0, 0.0, 0.0),
                release_timestamp_sec=0.0,
                elevation_deg=0.0,
                azimuth_deg=0.0,
                c1_c2_speed_km_h=None,
                c1_c2_distance_m=None,
                c1_c2_dt_sec=None,
                c2_frame_index=None,
                is_measured=False,
                status="NOT_MEASURED",
                rejection_reason=reason,
                release_speed_result=release_speed_res,
                c1_c2_result=None,
                framewise_report=None,
                validation_audit={"status": "ABORTED", "reason": reason},
                master_logs=master_logs
            )

        # 2. Compute dynamic C1-C2 average speed
        c1_c2_res: Optional[C1C2SpeedResult] = None
        c1_arr = np.array(release_speed_res.c1_release_pos_m, dtype=np.float64)
        t1 = release_speed_res.release_timestamp_sec

        if fitted_3d_points and len(fitted_3d_points) > 1:
            c1_c2_res = self.c1_c2_calculator.compute_c1_c2(c1_arr, t1, fitted_3d_points)
            master_logs.append("--- Dynamic C1-C2 Baseline Speed Analysis ---")
            master_logs.extend(c1_c2_res.selection_logs)
        else:
            master_logs.append("C1-C2 baseline speed bypassed: insufficient 3D trajectory points.")

        # 3. Compute Frame-wise speed profile
        framewise_report: Optional[FramewiseSpeedReport] = None
        if fitted_3d_points and len(fitted_3d_points) >= 2:
            framewise_report = FramewiseSpeedCalculator.compute_profile(fitted_3d_points)
            master_logs.append("--- Frame-wise Telemetry Profile ---")
            master_logs.append(
                f"Computed {len(framewise_report.records)} frame steps: mean = {framewise_report.mean_speed_km_h:.2f} km/h, "
                f"drag deceleration = {framewise_report.mean_deceleration_m_s2:.2f} m/s^2 "
                f"({framewise_report.deceleration_loss_kmh_per_m:.2f} km/h per metre)."
            )

        # 4. Aerodynamic & Physical Validation
        is_valid, rejection_reason, audit = self.validator.validate_with_details(
            release_speed_res, c1_c2_res
        )

        master_logs.append("--- Speed Quality & Physics Audit ---")
        if is_valid:
            master_logs.append(f"PHYSICAL VALIDATION PASSED: All checks satisfied ({', '.join(audit['checks_passed'])}).")
            status = "MEASURED"
            is_measured = True
            final_reason = None
        else:
            master_logs.append(f"PHYSICAL VALIDATION REJECTED: {rejection_reason}")
            status = "NOT_MEASURED"
            is_measured = False
            final_reason = rejection_reason

        return BowlingSpeedOutput(
            release_speed_km_h=release_speed_res.speed_km_h,
            release_speed_m_s=release_speed_res.speed_m_s,
            release_uncertainty_km_h=release_speed_res.uncertainty_km_h,
            ci_95_km_h=(release_speed_res.ci_95_lower_km_h, release_speed_res.ci_95_upper_km_h),
            velocity_vector_m_s=release_speed_res.velocity_vector_m_s,
            release_position_m=release_speed_res.c1_release_pos_m,
            release_timestamp_sec=release_speed_res.release_timestamp_sec,
            elevation_deg=release_speed_res.elevation_deg,
            azimuth_deg=release_speed_res.azimuth_deg,
            c1_c2_speed_km_h=c1_c2_res.average_speed_km_h if (c1_c2_res and c1_c2_res.is_valid) else None,
            c1_c2_distance_m=c1_c2_res.distance_3d_m if (c1_c2_res and c1_c2_res.is_valid) else None,
            c1_c2_dt_sec=c1_c2_res.time_delta_sec if (c1_c2_res and c1_c2_res.is_valid) else None,
            c2_frame_index=c1_c2_res.c2_frame_index if (c1_c2_res and c1_c2_res.is_valid) else None,
            is_measured=is_measured,
            status=status,
            rejection_reason=final_reason,
            release_speed_result=release_speed_res,
            c1_c2_result=c1_c2_res,
            framewise_report=framewise_report,
            validation_audit=audit,
            master_logs=master_logs
        )
