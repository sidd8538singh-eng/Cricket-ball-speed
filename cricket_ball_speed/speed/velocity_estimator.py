"""
Velocity Estimator: Computes 3D velocity vectors and speed magnitudes.
Converts physical m/s to broadcast standard km/h using the exact physical constant 3.6.

CRITICAL SCIENTIFIC PRINCIPLES:
1. Speed is computed strictly from 3D world coordinates in metres and actual timestamps in seconds.
2. Final speed is NEVER calculated from raw 2D pixel displacement.
3. No arbitrary multipliers, fudge factors, or empirical scaling factors are ever applied.
4. Exact conversion: 1 m/s = 3.6 km/h.
"""

from typing import Tuple, Dict, Any, List
import numpy as np


class VelocityEstimator:
    """Computes speed and directional velocity components in world pitch coordinates."""

    @staticmethod
    def vector_to_speed(vx: float, vy: float, vz: float) -> Tuple[float, float]:
        """
        Computes speed in m/s and km/h from 3D velocity components:
            speed_m_s = sqrt(vx^2 + vy^2 + vz^2)
            speed_km_h = speed_m_s * 3.6
        Returns:
            (speed_m_s, speed_km_h)
        """
        vx = float(vx)
        vy = float(vy)
        vz = float(vz)
        speed_m_s = float(np.hypot(vx, np.hypot(vy, vz)))
        speed_km_h = float(speed_m_s * 3.6)
        return speed_m_s, speed_km_h

    @staticmethod
    def compute_displacement_speed(
        pos_start: np.ndarray,
        pos_end: np.ndarray,
        dt_seconds: float
    ) -> Tuple[float, float, np.ndarray]:
        """
        Computes average 3D velocity and speed between two 3D positions over actual delta time:
            dist = ||pos_end - pos_start||
            speed = dist / dt
        Returns:
            (speed_m_s, speed_km_h, velocity_vector_m_s)
        """
        dt = float(dt_seconds)
        if dt <= 1e-6:
            raise ValueError(f"Time delta {dt:.6f}s is too small for velocity estimation.")

        p1 = np.asarray(pos_start, dtype=np.float64).flatten()
        p2 = np.asarray(pos_end, dtype=np.float64).flatten()

        diff = p2 - p1
        dist = float(np.linalg.norm(diff))
        speed_m_s = float(dist / dt)
        speed_km_h = float(speed_m_s * 3.6)
        v_vector = diff / dt
        return speed_m_s, speed_km_h, v_vector

    @staticmethod
    def compute_trajectory_angles_deg(vx: float, vy: float, vz: float) -> Tuple[float, float]:
        """
        Computes flight angles in degrees:
        - elevation_angle_deg: angle relative to horizontal ground plane (+ is upward, - is downward towards pitch)
        - azimuth_angle_deg: lateral cut/drift angle relative to down-pitch axis Y (+ is off-side, - is leg-side)
        """
        vx = float(vx)
        vy = float(vy)
        vz = float(vz)
        horizontal_speed = float(np.hypot(vx, vy))
        elevation_deg = float(np.degrees(np.arctan2(vz, max(1e-6, horizontal_speed))))
        azimuth_deg = float(np.degrees(np.arctan2(vx, max(1e-6, vy))))
        return elevation_deg, azimuth_deg
