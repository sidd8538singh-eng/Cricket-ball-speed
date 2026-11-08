"""
Ball Observation Representation.
Encapsulates individual ball candidate detections with full audit trail:
raw detections, bounding boxes, apparent diameters, and acceptance/rejection states.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np


@dataclass
class BallObservation:
    """Represents a single ball detection candidate in a video frame."""
    frame_index: int
    timestamp_sec: float
    bbox: List[float]               # [x1, y1, x2, y2] in original image coordinates
    confidence: float
    center: List[float]             # [cx, cy] in original image coordinates
    width: float
    height: float
    apparent_diameter_px: float     # Estimated diameter in pixels
    detection_source: str           # "full_frame", "tiled", "refined", etc.
    acceptance_status: str = "PENDING"  # "ACCEPTED", "REJECTED", "PENDING"
    rejection_reason: Optional[str] = None
    quality_score: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_bbox(
        cls,
        frame_idx: int,
        timestamp_sec: float,
        bbox: List[float],
        confidence: float,
        detection_source: str = "raw"
    ) -> "BallObservation":
        x1, y1, x2, y2 = bbox
        w = max(0.1, x2 - x1)
        h = max(0.1, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        # For a spherical ball, apparent diameter is approximately the average of width and height
        apparent_diam = (w + h) / 2.0

        return cls(
            frame_index=frame_idx,
            timestamp_sec=timestamp_sec,
            bbox=[float(x1), float(y1), float(x2), float(y2)],
            confidence=float(confidence),
            center=[float(cx), float(cy)],
            width=float(w),
            height=float(h),
            apparent_diameter_px=float(apparent_diam),
            detection_source=detection_source
        )

    def mark_accepted(self):
        self.acceptance_status = "ACCEPTED"
        self.rejection_reason = None

    def mark_rejected(self, reason: str):
        self.acceptance_status = "REJECTED"
        self.rejection_reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 4),
            "bbox": [round(c, 2) for c in self.bbox],
            "confidence": round(self.confidence, 4),
            "center": [round(c, 2) for c in self.center],
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "apparent_diameter_px": round(self.apparent_diameter_px, 2),
            "detection_source": self.detection_source,
            "acceptance_status": self.acceptance_status,
            "rejection_reason": self.rejection_reason,
            "quality_score": round(self.quality_score, 3),
        }
