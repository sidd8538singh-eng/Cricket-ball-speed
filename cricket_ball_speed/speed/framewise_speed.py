"""
Framewise Speed: Calculates frame-by-frame 3D speed profiles across the trajectory.
Provides telemetry diagnostics without using noisy raw pixel differences as final output.

CRITICAL SCIENTIFIC PRINCIPLES:
1. Frame-wise speed is computed strictly between reconstructed 3D points using actual timestamps.
2. dt is evaluated per frame pair (accounting for non-uniform frame intervals or skipped frames).
3. Acceleration/deceleration is derived directly from the physical 3D velocity progression.
4. Telemetry profiles provide diagnostic visibility into aerodynamic drag and trajectory continuity.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from reconstruction.ball_3d import Ball3DPoint
from .velocity_estimator import VelocityEstimator


@dataclass
class FramewiseSpeedReport:
    """Encapsulates full frame-wise speed profile, smoothing, and deceleration metrics."""
    records: List[Dict[str, Any]]
    mean_speed_km_h: float
    min_speed_km_h: float
    max_speed_km_h: float
    total_distance_m: float
    total_duration_sec: float
    mean_deceleration_m_s2: float       # Physical deceleration due to drag (+ is slowing down)
    deceleration_loss_kmh_per_m: float  # km/h lost per metre of forward flight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_speed_km_h": round(float(self.mean_speed_km_h), 2),
            "min_speed_km_h": round(float(self.min_speed_km_h), 2),
            "max_speed_km_h": round(float(self.max_speed_km_h), 2),
            "total_distance_m": round(float(self.total_distance_m), 3),
            "total_duration_sec": round(float(self.total_duration_sec), 5),
            "mean_deceleration_m_s2": round(float(self.mean_deceleration_m_s2), 2),
            "deceleration_loss_kmh_per_m": round(float(self.deceleration_loss_kmh_per_m), 3),
            "num_steps": len(self.records),
            "records": self.records,
        }


class FramewiseSpeedCalculator:
    """
    Computes speed profiles between consecutive reconstructed 3D ball coordinates.
    Includes moving-average smoothing and deceleration trend analysis.
    """

    @staticmethod
    def compute_framewise_speeds(
        points_3d: List[Ball3DPoint], window_size: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Legacy list output for backward compatibility:
        Computes consecutive speed between adjacent 3D points.
        Returns a list of dicts with frame, timestamp, velocity, and smoothed speed.
        """
        report = FramewiseSpeedCalculator.compute_profile(points_3d, window_size)
        return report.records

    @staticmethod
    def compute_profile(
        points_3d: List[Ball3DPoint], window_size: int = 3
    ) -> FramewiseSpeedReport:
        """
        Computes comprehensive frame-by-frame 3D speed and aerodynamic deceleration profile.
        """
        if len(points_3d) < 2:
            return FramewiseSpeedReport(
                records=[],
                mean_speed_km_h=0.0,
                min_speed_km_h=0.0,
                max_speed_km_h=0.0,
                total_distance_m=0.0,
                total_duration_sec=0.0,
                mean_deceleration_m_s2=0.0,
                deceleration_loss_kmh_per_m=0.0
            )

        records: List[Dict[str, Any]] = []
        raw_speeds_kmh: List[float] = []
        raw_speeds_ms: List[float] = []

        total_dist = 0.0
        total_time = 0.0

        for i in range(len(points_3d) - 1):
            p1 = points_3d[i]
            p2 = points_3d[i + 1]
            dt = float(p2.timestamp_sec - p1.timestamp_sec)
            if dt <= 1e-6:
                continue

            diff = p2.position_xyz - p1.position_xyz
            dist = float(np.linalg.norm(diff))
            speed_m_s = dist / dt
            speed_km_h = speed_m_s * 3.6

            raw_speeds_ms.append(speed_m_s)
            raw_speeds_kmh.append(speed_km_h)
            total_dist += dist
            total_time += dt

            records.append({
                "frame_start": p1.frame_index,
                "frame_end": p2.frame_index,
                "timestamp_mid_sec": round((p1.timestamp_sec + p2.timestamp_sec) / 2.0, 5),
                "speed_raw_kmh": round(speed_km_h, 2),
                "speed_m_s": round(speed_m_s, 2),
                "vx_m_s": round(float(diff[0] / dt), 3),
                "vy_m_s": round(float(diff[1] / dt), 3),
                "vz_m_s": round(float(diff[2] / dt), 3),
                "distance_m": round(dist, 3),
                "dt_sec": round(dt, 5),
                "acceleration_m_s2": 0.0,
            })

        if not records:
            return FramewiseSpeedReport(
                records=[],
                mean_speed_km_h=0.0,
                min_speed_km_h=0.0,
                max_speed_km_h=0.0,
                total_distance_m=0.0,
                total_duration_sec=0.0,
                mean_deceleration_m_s2=0.0,
                deceleration_loss_kmh_per_m=0.0
            )

        # Compute accelerations between consecutive speed steps
        for i in range(len(records) - 1):
            dt_step = records[i + 1]["timestamp_mid_sec"] - records[i]["timestamp_mid_sec"]
            if dt_step > 1e-6:
                acc = (records[i + 1]["speed_m_s"] - records[i]["speed_m_s"]) / dt_step
                records[i]["acceleration_m_s2"] = round(acc, 2)

        # Apply moving average filter for smoothed speed
        if len(raw_speeds_kmh) >= window_size and window_size > 1:
            smoothed = np.convolve(raw_speeds_kmh, np.ones(window_size) / window_size, mode="same")
            for i, rec in enumerate(records):
                rec["speed_smoothed_kmh"] = round(float(smoothed[i]), 2)
        else:
            for rec in records:
                rec["speed_smoothed_kmh"] = rec["speed_raw_kmh"]

        mean_speed = float(np.mean(raw_speeds_kmh))
        min_speed = float(np.min(raw_speeds_kmh))
        max_speed = float(np.max(raw_speeds_kmh))

        # Mean deceleration: speed loss over total duration
        mean_decel = 0.0
        decel_loss_per_m = 0.0
        if total_time > 1e-4 and len(raw_speeds_ms) > 1:
            speed_delta_ms = raw_speeds_ms[0] - raw_speeds_ms[-1]
            mean_decel = speed_delta_ms / total_time
            if total_dist > 0.5:
                decel_loss_per_m = (raw_speeds_kmh[0] - raw_speeds_kmh[-1]) / total_dist

        return FramewiseSpeedReport(
            records=records,
            mean_speed_km_h=mean_speed,
            min_speed_km_h=min_speed,
            max_speed_km_h=max_speed,
            total_distance_m=total_dist,
            total_duration_sec=total_time,
            mean_deceleration_m_s2=mean_decel,
            deceleration_loss_kmh_per_m=decel_loss_per_m
        )
