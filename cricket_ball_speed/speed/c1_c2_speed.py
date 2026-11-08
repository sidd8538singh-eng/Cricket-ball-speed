"""
C1-C2 Average Speed Calculator.
Computes average delivery speed between release (C1) and a later verified 3D position (C2).
Dynamically selects C2 targeting approximately 5 to 8 metres down the pitch.

STRICT ARCHITECTURAL PRINCIPLES:
1. C2 is selected DYNAMICALLY based on physical 3D distance from C1, NEVER by a fixed frame number.
2. Uses actual timestamps (dt = t2 - t1), accounting for non-uniform frame intervals.
3. Does NOT calculate speed using raw pixel displacement.
4. Does NOT apply arbitrary multipliers or fudge factors.
5. Distinct from instantaneous release speed.
6. Returns NOT_MEASURED if observations do not reach a reliable distance baseline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from reconstruction.ball_3d import Ball3DPoint
from .velocity_estimator import VelocityEstimator


@dataclass
class C1C2SpeedResult:
    """Encapsulates C1-C2 average speed result with dynamic selection metadata and audit logs."""
    c1_pos_m: Tuple[float, float, float]
    c2_pos_m: Tuple[float, float, float]
    t1_sec: float
    t2_sec: float
    distance_3d_m: float
    time_delta_sec: float
    average_speed_km_h: float
    average_speed_m_s: float
    is_valid: bool = True
    status: str = "MEASURED"                    # "MEASURED" or "NOT_MEASURED"
    rejection_reason: Optional[str] = None
    c2_frame_index: int = 0
    selection_logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "c1_pos_m": [round(float(c), 3) for c in self.c1_pos_m],
            "c2_pos_m": [round(float(c), 3) for c in self.c2_pos_m],
            "t1_sec": round(float(self.t1_sec), 5),
            "t2_sec": round(float(self.t2_sec), 5),
            "distance_3d_m": round(float(self.distance_3d_m), 3),
            "time_delta_sec": round(float(self.time_delta_sec), 5),
            "average_speed_km_h": round(float(self.average_speed_km_h), 2),
            "average_speed_m_s": round(float(self.average_speed_m_s), 2),
            "c2_frame_index": self.c2_frame_index,
            "is_valid": self.is_valid,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "selection_logs": self.selection_logs,
        }


class C1C2SpeedCalculator:
    """
    Selects C2 dynamically targeting 5 to 8 metres down-pitch from release (C1).
    Calculates average speed over the baseline interval using physical 3D distance and actual timestamps.
    """

    def __init__(
        self,
        target_distance_m: float = 6.5,
        min_distance_m: float = 4.0,
        max_distance_m: float = 9.0
    ):
        """
        Parameters:
            target_distance_m: Optimal distance from C1 to target for C2 (default 6.5m, mid-range of 5-8m).
            min_distance_m: Minimum allowable baseline distance (default 4.0m).
            max_distance_m: Maximum allowable baseline distance before bounce (default 9.0m).
        """
        self.target_dist = target_distance_m
        self.min_dist = min_distance_m
        self.max_dist = max_distance_m

    def compute_c1_c2(
        self,
        c1_pos_3d: np.ndarray,
        t1_sec: float,
        fitted_points: List[Ball3DPoint]
    ) -> C1C2SpeedResult:
        """
        Dynamically finds best candidate point C2 from verified 3D trajectory points.
        Evaluates distance ||P_i - C1|| and selects point minimizing |dist - target_dist|.
        
        Returns C1C2SpeedResult with status='MEASURED' if successful,
        or status='NOT_MEASURED' if baseline criteria cannot be met.
        """
        logs: List[str] = []
        c1_arr = np.asarray(c1_pos_3d, dtype=np.float64).flatten()
        logs.append(f"C1 Release Position: ({c1_arr[0]:.3f}, {c1_arr[1]:.3f}, {c1_arr[2]:.3f}) m at t1 = {t1_sec:.5f}s")
        logs.append(f"Targeting baseline distance: {self.target_dist:.1f}m (acceptable: [{self.min_dist:.1f}m, {self.max_dist:.1f}m])")

        if not fitted_points:
            msg = "INSUFFICIENT_OBSERVATIONS: No 3D points available for C2 selection."
            logs.append(msg)
            return C1C2SpeedResult(
                c1_pos_m=(float(c1_arr[0]), float(c1_arr[1]), float(c1_arr[2])),
                c2_pos_m=(0.0, 0.0, 0.0),
                t1_sec=t1_sec,
                t2_sec=0.0,
                distance_3d_m=0.0,
                time_delta_sec=0.0,
                average_speed_km_h=0.0,
                average_speed_m_s=0.0,
                is_valid=False,
                status="NOT_MEASURED",
                rejection_reason=msg,
                c2_frame_index=0,
                selection_logs=logs,
            )

        # Filter points that occur after release with meaningful time delta (> 20ms)
        post_pts = [p for p in fitted_points if p.timestamp_sec > (t1_sec + 0.02)]
        logs.append(f"Found {len(post_pts)} post-release 3D points (dt > 0.02s).")

        if not post_pts:
            msg = "INSUFFICIENT_DURATION: No 3D points detected after release threshold (t1 + 0.02s)."
            logs.append(msg)
            return C1C2SpeedResult(
                c1_pos_m=(float(c1_arr[0]), float(c1_arr[1]), float(c1_arr[2])),
                c2_pos_m=(0.0, 0.0, 0.0),
                t1_sec=t1_sec,
                t2_sec=0.0,
                distance_3d_m=0.0,
                time_delta_sec=0.0,
                average_speed_km_h=0.0,
                average_speed_m_s=0.0,
                is_valid=False,
                status="NOT_MEASURED",
                rejection_reason=msg,
                c2_frame_index=0,
                selection_logs=logs,
            )

        # Find point closest to target distance within [min_dist, max_dist]
        best_pt: Optional[Ball3DPoint] = None
        best_diff = float("inf")
        best_dist = 0.0

        for pt in post_pts:
            dist = float(np.linalg.norm(pt.position_xyz - c1_arr))
            if self.min_dist <= dist <= self.max_dist:
                diff = abs(dist - self.target_dist)
                if diff < best_diff:
                    best_diff = diff
                    best_pt = pt
                    best_dist = dist

        # Fallback: if no point inside [min_dist, max_dist], pick furthest point if >= min_dist
        if best_pt is None:
            distances = [(float(np.linalg.norm(p.position_xyz - c1_arr)), p) for p in post_pts]
            max_d, furthest_pt = max(distances, key=lambda d: d[0])
            if max_d >= self.min_dist:
                best_pt = furthest_pt
                best_dist = max_d
                logs.append(f"Fallback selection: Furthest available point at distance {max_d:.2f}m selected.")
            else:
                msg = (
                    f"BASELINE_TOO_SHORT: Furthest observable 3D point reached only {max_d:.2f}m from C1, "
                    f"which is less than the required minimum baseline {self.min_dist:.2f}m."
                )
                logs.append(msg)
                return C1C2SpeedResult(
                    c1_pos_m=(float(c1_arr[0]), float(c1_arr[1]), float(c1_arr[2])),
                    c2_pos_m=(0.0, 0.0, 0.0),
                    t1_sec=t1_sec,
                    t2_sec=0.0,
                    distance_3d_m=0.0,
                    time_delta_sec=0.0,
                    average_speed_km_h=0.0,
                    average_speed_m_s=0.0,
                    is_valid=False,
                    status="NOT_MEASURED",
                    rejection_reason=msg,
                    c2_frame_index=0,
                    selection_logs=logs,
                )

        c2_pos = best_pt.position_xyz
        t2_sec = best_pt.timestamp_sec
        dt = t2_sec - t1_sec

        logs.append(
            f"Selected C2 dynamically: frame {best_pt.frame_index}, t2 = {t2_sec:.5f}s, "
            f"pos = ({c2_pos[0]:.3f}, {c2_pos[1]:.3f}, {c2_pos[2]:.3f}) m."
        )
        logs.append(f"Physical 3D baseline distance = {best_dist:.3f} m, delta time = {dt:.5f} s.")

        speed_m_s, speed_km_h, _ = VelocityEstimator.compute_displacement_speed(
            c1_arr, c2_pos, dt
        )
        logs.append(
            f"Computed C1-C2 average speed: {best_dist:.3f}m / {dt:.5f}s = {speed_m_s:.3f} m/s "
            f"-> {speed_km_h:.2f} km/h (exact 3.6 constant)."
        )

        return C1C2SpeedResult(
            c1_pos_m=(float(c1_arr[0]), float(c1_arr[1]), float(c1_arr[2])),
            c2_pos_m=(float(c2_pos[0]), float(c2_pos[1]), float(c2_pos[2])),
            t1_sec=t1_sec,
            t2_sec=t2_sec,
            distance_3d_m=best_dist,
            time_delta_sec=dt,
            average_speed_km_h=speed_km_h,
            average_speed_m_s=speed_m_s,
            c2_frame_index=best_pt.frame_index,
            is_valid=True,
            status="MEASURED",
            rejection_reason=None,
            selection_logs=logs,
        )
