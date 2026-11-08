"""
DRS Reference Line Generation and Geometry.
Constructs and validates the 20.12m longitudinal geometric reference axis between
the near wicket base centre and the far wicket base centre.

Mathematical Definition:
  Origin A = (0, 0, 0) : Centre of near wicket base at ground level.
  Target B = (0, L, 0) : Centre of far wicket base at ground level, L = 20.12 m.
  DRS Vector in World Space: D_world = B - A = (0, L, 0).
  DRS Line in Image Plane: Ray / segment connecting image projections a = pi(A) and b = pi(B).

CRITICAL ENGINEERING RULE:
  The DRS reference line defines the longitudinal reference axis of the pitch.
  It is NEVER forced to be the actual ball trajectory; the cricket ball may move
  laterally or vertically relative to this line.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np


@dataclass
class DRSLine:
    """
    Represents the automatically constructed 20.12m DRS reference line.
    Stored in both original pixel coordinates and metric reference space.
    """
    near_point: Tuple[float, float]     # a = (u_near, v_near) in original image pixels
    far_point: Tuple[float, float]      # b = (u_far, v_far) in original image pixels
    real_distance_m: float = 20.12      # Physical distance between wicket base centres
    pixel_length: float = 0.0           # Euclidean length in image pixels
    angle_deg: float = 0.0              # Orientation angle relative to image horizontal (degrees)
    midpoint: Tuple[float, float] = (0.0, 0.0) # (u_mid, v_mid) in image pixels
    is_valid: bool = False              # Whether geometric and perspective validation passed
    is_provisional: bool = False        # Whether line is provisional prior to calibration confirmation
    confidence: float = 0.0             # Aggregate confidence score [0.0, 1.0]
    validation_notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.compute_geometry()

    def compute_geometry(self):
        """Calculates pixel length, midpoint, and angle."""
        dx = self.far_point[0] - self.near_point[0]
        dy = self.far_point[1] - self.near_point[1]
        self.pixel_length = float(math.hypot(dx, dy))
        self.midpoint = (
            float((self.near_point[0] + self.far_point[0]) / 2.0),
            float((self.near_point[1] + self.far_point[1]) / 2.0),
        )
        # Angle in degrees (-180 to 180)
        self.angle_deg = float(math.degrees(math.atan2(dy, dx)))

    @property
    def direction_vector_2d(self) -> Tuple[float, float]:
        """Unit vector in image coordinates pointing from near wicket to far wicket."""
        if self.pixel_length < 1e-6:
            return (0.0, 0.0)
        dx = (self.far_point[0] - self.near_point[0]) / self.pixel_length
        dy = (self.far_point[1] - self.near_point[1]) / self.pixel_length
        return (float(dx), float(dy))

    def lateral_distance_px(self, point: Tuple[float, float]) -> float:
        """
        Calculates the signed perpendicular (lateral) pixel distance of a 2D image point
        from the DRS reference line. Positive if to the right of the near->far ray.
        """
        x0, y0 = point
        x1, y1 = self.near_point
        x2, y2 = self.far_point
        if self.pixel_length < 1e-6:
            return 0.0
        # 2D cross product: (x2-x1)*(y0-y1) - (y2-y1)*(x0-x1)
        cross = (x2 - x1) * (y0 - y1) - (y2 - y1) * (x0 - x1)
        return float(cross / self.pixel_length)

    def longitudinal_projection_px(self, point: Tuple[float, float]) -> float:
        """
        Calculates normalized longitudinal projection fraction along the DRS line [0 = near, 1 = far].
        """
        if self.pixel_length < 1e-6:
            return 0.0
        dx = self.far_point[0] - self.near_point[0]
        dy = self.far_point[1] - self.near_point[1]
        px = point[0] - self.near_point[0]
        py = point[1] - self.near_point[1]
        dot = px * dx + py * dy
        return float(dot / (self.pixel_length ** 2))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes DRSLine into a JSON-compatible dictionary."""
        return {
            "near_point": [round(c, 2) for c in self.near_point],
            "far_point": [round(c, 2) for c in self.far_point],
            "real_distance_m": float(self.real_distance_m),
            "pixel_length": round(float(self.pixel_length), 2),
            "angle_deg": round(float(self.angle_deg), 2),
            "midpoint": [round(c, 2) for c in self.midpoint],
            "direction_vector_2d": [round(c, 4) for c in self.direction_vector_2d],
            "is_valid": bool(self.is_valid),
            "is_provisional": bool(self.is_provisional),
            "confidence": round(float(self.confidence), 4),
            "validation_notes": list(self.validation_notes)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DRSLine":
        """Instantiates DRSLine from dictionary."""
        return cls(
            near_point=tuple(data["near_point"]),
            far_point=tuple(data["far_point"]),
            real_distance_m=float(data.get("real_distance_m", 20.12)),
            pixel_length=float(data.get("pixel_length", 0.0)),
            angle_deg=float(data.get("angle_deg", 0.0)),
            midpoint=tuple(data.get("midpoint", (0.0, 0.0))),
            is_valid=bool(data.get("is_valid", False)),
            is_provisional=bool(data.get("is_provisional", False)),
            confidence=float(data.get("confidence", 0.0)),
            validation_notes=list(data.get("validation_notes", []))
        )


class DRSLineValidator:
    """
    Validates the geometric plausibility of an automatically generated DRS line
    based on image perspective, stump alignment, and pitch direction.
    """

    @staticmethod
    def validate(
        near_base: Tuple[float, float],
        far_base: Tuple[float, float],
        image_width: int,
        image_height: int,
        near_confidence: float = 1.0,
        far_confidence: float = 1.0,
        pitch_bbox: Optional[List[float]] = None,
        min_pixel_length: float = 50.0
    ) -> DRSLine:
        """
        Constructs and thoroughly validates a candidate DRS reference line.
        """
        notes: List[str] = []
        is_valid = True

        dx = far_base[0] - near_base[0]
        dy = far_base[1] - near_base[1]
        pixel_length = math.hypot(dx, dy)

        # 1. Minimum Pixel Length Check
        if pixel_length < min_pixel_length:
            is_valid = False
            notes.append(f"DRS line pixel length ({pixel_length:.1f}px) is below minimum threshold ({min_pixel_length}px).")

        # 2. Perspective Check (In standard cricket broadcast/practice cameras behind bowler/stumps,
        # the near wicket is lower in the frame (larger y) and the far wicket is higher in the frame (smaller y)).
        # If far_base[1] >= near_base[1], the perspective orientation is inverted or camera is overhead.
        if far_base[1] > near_base[1]:
            notes.append("Inverted perspective: far wicket base is positioned below near wicket base in image space.")
            # We don't automatically invalidate for top-down/drone cameras, but flag confidence penalty
            is_valid = False

        # 3. Image Boundary Check
        for pt, name in [(near_base, "Near"), (far_base, "Far")]:
            if pt[0] < 0 or pt[0] > image_width or pt[1] < 0 or pt[1] > image_height:
                is_valid = False
                notes.append(f"{name} wicket base coordinate {pt} lies outside image bounds ({image_width}x{image_height}).")

        # 4. Angle / Alignment Check (Typically pitch is oriented roughly vertically in frame, -150 to -30 deg)
        angle = math.degrees(math.atan2(dy, dx))
        # Near to far generally has dy < 0 (pointing up towards horizon)
        if dy > 0:
            notes.append(f"DRS direction points downward in image (dy = {dy:.1f}px).")

        # 5. Pitch Bounding Box Cross-Check (if available)
        if pitch_bbox is not None and len(pitch_bbox) >= 4:
            px1, py1, px2, py2 = pitch_bbox
            # Both bases should lie reasonably near the pitch region
            margin_x = (px2 - px1) * 0.35
            if not (px1 - margin_x <= near_base[0] <= px2 + margin_x and px1 - margin_x <= far_base[0] <= px2 + margin_x):
                notes.append("DRS line endpoints diverge significantly from detected pitch bounding box.")
                is_valid = False

        # Aggregate confidence
        geom_penalty = 1.0 if is_valid else 0.4
        conf = float(math.sqrt(max(0.0, near_confidence * far_confidence)) * geom_penalty)

        drs = DRSLine(
            near_point=near_base,
            far_point=far_base,
            real_distance_m=20.12,
            is_valid=is_valid,
            is_provisional=not is_valid,
            confidence=conf,
            validation_notes=notes
        )
        return drs
