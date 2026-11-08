"""
Ball 3D Point: Encapsulates 3D spatial position, timestamp, source provenance,
and uncertainty metrics for reconstructed cricket ball points.
Distinguishes real sensor-observed ray points from interpolated/extrapolated predicted points.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class Ball3DPoint:
    """Represents a 3D reconstructed ball position in world pitch coordinates."""
    frame_index: int
    timestamp_sec: float
    position_xyz: np.ndarray       # [X, Y, Z] in world metres (origin A = near stumps base)
    uncertainty_xyz: np.ndarray    # [sigma_X, sigma_Y, sigma_Z] in metres
    source: str                    # "ray_fitted", "apparent_diameter", "triangulated", "kalman_3d"
    point_type: str = "OBSERVED"   # "OBSERVED" (backed by sensor ray) or "PREDICTED" (interpolated)
    is_real_detection: bool = True # False for prediction-only states
    reprojection_error_px: float = 0.0
    depth_from_camera_m: float = 0.0
    depth_uncertainty_m: float = 0.0
    observation_confidence: float = 1.0
    velocity_xyz: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def x(self) -> float:
        return float(self.position_xyz[0])

    @property
    def y(self) -> float:
        return float(self.position_xyz[1])

    @property
    def z(self) -> float:
        return float(self.position_xyz[2])

    @property
    def total_uncertainty_m(self) -> float:
        return float(np.linalg.norm(self.uncertainty_xyz))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 5),
            "position_xyz_m": [round(float(c), 3) for c in self.position_xyz],
            "uncertainty_xyz_m": [round(float(u), 3) for u in self.uncertainty_xyz],
            "total_uncertainty_m": round(self.total_uncertainty_m, 3),
            "point_type": self.point_type,
            "is_real_detection": self.is_real_detection,
            "source": self.source,
            "reprojection_error_px": round(self.reprojection_error_px, 3),
            "depth_from_camera_m": round(self.depth_from_camera_m, 3),
            "depth_uncertainty_m": round(self.depth_uncertainty_m, 3),
            "observation_confidence": round(self.observation_confidence, 4),
            "velocity_xyz_m_s": (
                [round(float(v), 3) for v in self.velocity_xyz]
                if self.velocity_xyz is not None
                else None
            ),
        }
