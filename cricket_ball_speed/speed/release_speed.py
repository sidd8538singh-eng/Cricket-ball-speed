"""
Release Speed: Computes instantaneous 3D bowling speed at release time (t1).
Primary broadcast metric: ||V(t1)|| * 3.6 km/h with formal confidence intervals.

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. Speed is computed strictly from the fitted 3D velocity vector (VX, VY, VZ) at release.
2. Final speed is NEVER calculated from raw pixel displacement.
3. No arbitrary multipliers are applied (exact 3.6 km/h per m/s conversion).
4. Radar speed and filename speed are NEVER used as calculation inputs.
5. Release height is NEVER used to force the speed.
6. Returns NOT_MEASURED if the trajectory fit or observations are unreliable.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

from reconstruction.trajectory_fitting import TrajectoryFitResult
from .velocity_estimator import VelocityEstimator


@dataclass
class ReleaseSpeedResult:
    """Encapsulates instantaneous 3D bowling speed at release with full audit logs."""
    speed_km_h: float
    speed_m_s: float
    uncertainty_km_h: float
    ci_95_lower_km_h: float
    ci_95_upper_km_h: float
    velocity_vector_m_s: Tuple[float, float, float]
    c1_release_pos_m: Tuple[float, float, float]
    release_timestamp_sec: float
    is_valid: bool = True
    status: str = "MEASURED"                    # "MEASURED" or "NOT_MEASURED"
    rejection_reason: Optional[str] = None
    elevation_deg: float = 0.0                  # Pitch angle relative to horizontal ground
    azimuth_deg: float = 0.0                    # Lateral drift/cut angle relative to pitch axis
    calculation_logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speed_km_h": round(self.speed_km_h, 2),
            "speed_m_s": round(self.speed_m_s, 2),
            "uncertainty_km_h": round(self.uncertainty_km_h, 2),
            "ci_95_km_h": [round(self.ci_95_lower_km_h, 2), round(self.ci_95_upper_km_h, 2)],
            "velocity_vector_m_s": [round(float(v), 3) for v in self.velocity_vector_m_s],
            "c1_release_pos_m": [round(float(p), 3) for p in self.c1_release_pos_m],
            "release_timestamp_sec": round(self.release_timestamp_sec, 5),
            "elevation_deg": round(self.elevation_deg, 2),
            "azimuth_deg": round(self.azimuth_deg, 2),
            "is_valid": self.is_valid,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "calculation_logs": self.calculation_logs,
        }


def compute_release_speed(fit_result: Optional[TrajectoryFitResult]) -> ReleaseSpeedResult:
    """
    Computes instantaneous release speed, velocity vector components, angles,
    and 95% confidence intervals from the fitted 3D trajectory at release t0.
    Generates step-by-step calculation logs.
    """
    logs: List[str] = []

    # Guard: Trajectory fit absent or invalid
    if fit_result is None or not fit_result.is_valid or fit_result.status == "NOT_MEASURED":
        reason = fit_result.rejection_reason if fit_result else "Trajectory fit result is None"
        logs.append(f"CALCULATION ABORTED: {reason}")
        return ReleaseSpeedResult(
            speed_km_h=0.0,
            speed_m_s=0.0,
            uncertainty_km_h=0.0,
            ci_95_lower_km_h=0.0,
            ci_95_upper_km_h=0.0,
            velocity_vector_m_s=(0.0, 0.0, 0.0),
            c1_release_pos_m=(0.0, 0.0, 0.0),
            release_timestamp_sec=0.0,
            is_valid=False,
            status="NOT_MEASURED",
            rejection_reason=f"TRAJECTORY_INVALID: {reason}",
            calculation_logs=logs
        )

    t0 = fit_result.t0
    vx = fit_result.vx0
    vy = fit_result.vy0
    vz = fit_result.vz0
    x0 = fit_result.x0
    y0 = fit_result.y0
    z0 = fit_result.z0

    logs.append(f"Step 1: Extracted release state at t0 = {t0:.5f}s:")
    logs.append(f"  Release position C1 = ({x0:.3f}, {y0:.3f}, {z0:.3f}) metres")
    logs.append(f"  Release velocity vector = (VX={vx:.3f}, VY={vy:.3f}, VZ={vz:.3f}) m/s")

    # Step 2: Instantaneous 3D velocity magnitude
    speed_m_s, speed_km_h = VelocityEstimator.vector_to_speed(vx, vy, vz)
    logs.append(f"Step 2: Computed 3D vector norm ||V|| = sqrt({vx:.3f}^2 + {vy:.3f}^2 + {vz:.3f}^2) = {speed_m_s:.3f} m/s")

    # Step 3: Exact physical conversion by 3.6
    logs.append(f"Step 3: Converted to km/h using exact constant 3.6: {speed_m_s:.3f} * 3.6 = {speed_km_h:.2f} km/h")
    logs.append("  (Zero arbitrary multipliers or fudge factors applied)")

    # Step 4: Flight angles
    elev_deg, azim_deg = VelocityEstimator.compute_trajectory_angles_deg(vx, vy, vz)
    logs.append(f"Step 4: Computed trajectory angles: Elevation = {elev_deg:.2f} deg, Azimuth = {azim_deg:.2f} deg")

    # Step 5: Uncertainty and 95% Confidence Interval
    unc_kmh = fit_result.uncertainty_speed_kmh
    ci_lower = max(0.0, speed_km_h - 1.96 * unc_kmh)
    ci_upper = speed_km_h + 1.96 * unc_kmh
    logs.append(f"Step 5: Evaluated speed uncertainty: +/- {unc_kmh:.2f} km/h")
    logs.append(f"  95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}] km/h")

    return ReleaseSpeedResult(
        speed_km_h=speed_km_h,
        speed_m_s=speed_m_s,
        uncertainty_km_h=unc_kmh,
        ci_95_lower_km_h=ci_lower,
        ci_95_upper_km_h=ci_upper,
        velocity_vector_m_s=(vx, vy, vz),
        c1_release_pos_m=(x0, y0, z0),
        release_timestamp_sec=t0,
        is_valid=True,
        status="MEASURED",
        rejection_reason=None,
        elevation_deg=elev_deg,
        azimuth_deg=azim_deg,
        calculation_logs=logs
    )
