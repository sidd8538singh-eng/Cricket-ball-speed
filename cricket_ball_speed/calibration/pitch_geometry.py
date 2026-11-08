"""
Pitch Geometry and Ground-Plane Alignment.
Integrates the optional pitch YOLO detector, validates pitch boundaries,
estimates 4 ground-plane pitch corners when available, and verifies DRS line alignment.

ICC Regulation Pitch Dimensions:
  - Crease-to-crease distance: 20.12 m (22 yards)
  - Pitch width: 3.048 m (10 ft, +/-1.524m from longitudinal center)
  - Ground Plane: Z = 0.0

Decoupling Principle:
  System functions fully with or without the pitch model.
  If unavailable, Stump + DRS calibration is utilized.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import logging
import cv2
import numpy as np

from .drs_line import DRSLine

logger = logging.getLogger("CricketBallSpeed.PitchGeometry")


@dataclass
class PitchCorners:
    """Represents the 4 ground-plane corners of the cricket pitch in image pixels."""
    pitch_left_near: Tuple[float, float]
    pitch_right_near: Tuple[float, float]
    pitch_left_far: Tuple[float, float]
    pitch_right_far: Tuple[float, float]
    confidence: float = 0.0
    source: str = "estimated"   # "model_mask", "model_bbox_perspective", "manual"
    is_valid: bool = False
    alignment_error_deg: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pitch_left_near": [round(c, 2) for c in self.pitch_left_near],
            "pitch_right_near": [round(c, 2) for c in self.pitch_right_near],
            "pitch_left_far": [round(c, 2) for c in self.pitch_left_far],
            "pitch_right_far": [round(c, 2) for c in self.pitch_right_far],
            "confidence": round(float(self.confidence), 4),
            "source": self.source,
            "is_valid": self.is_valid,
            "alignment_error_deg": round(float(self.alignment_error_deg), 2)
        }

    def to_points_dict(self) -> Dict[str, Tuple[float, float]]:
        return {
            "pitch_left_near": self.pitch_left_near,
            "pitch_right_near": self.pitch_right_near,
            "pitch_left_far": self.pitch_left_far,
            "pitch_right_far": self.pitch_right_far
        }


@dataclass
class PitchDetectionResult:
    """Result of pitch boundary detection and geometric cross-check."""
    is_detected: bool = False
    bbox: Optional[List[float]] = None
    corners: Optional[PitchCorners] = None
    confidence: float = 0.0
    alignment_angle_deg: float = 0.0
    lateral_offset_px: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_detected": self.is_detected,
            "bbox": [round(c, 2) for c in self.bbox] if self.bbox else None,
            "corners": self.corners.to_dict() if self.corners else None,
            "confidence": round(float(self.confidence), 4),
            "alignment_angle_deg": round(float(self.alignment_angle_deg), 2),
            "lateral_offset_px": round(float(self.lateral_offset_px), 2),
            "notes": self.notes
        }


class PitchGeometryEngine:
    """
    Manages pitch detection, corner estimation, and DRS-to-pitch alignment.
    """

    def __init__(self, model_path: Optional[str] = "models/pitch.pt", conf_threshold: float = 0.25):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self._model = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        """Loads optional pitch YOLO model."""
        import os
        if self.model_path and os.path.isfile(self.model_path):
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
                self.is_loaded = True
                logger.info(f"Loaded optional pitch model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load pitch model: {e}")
                self._model = None
                self.is_loaded = False
        else:
            self._model = None
            self.is_loaded = False

    def detect_pitch(
        self,
        frame_bgr: np.ndarray,
        drs_line: Optional[DRSLine] = None,
        custom_conf: Optional[float] = None
    ) -> PitchDetectionResult:
        """
        Runs pitch detector if available, extracts boundaries,
        and validates against the longitudinal DRS reference line.
        """
        if not self.is_loaded or self._model is None:
            return PitchDetectionResult(
                is_detected=False,
                notes=["Pitch model not loaded; pitch geometry estimation inactive."]
            )

        conf = custom_conf if custom_conf is not None else self.conf_threshold
        try:
            results = self._model.predict(frame_bgr, conf=conf, verbose=False)
            if not results or len(results[0].boxes) == 0:
                return PitchDetectionResult(
                    is_detected=False,
                    notes=["No pitch detected above confidence threshold."]
                )

            box = results[0].boxes[0]
            c = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = xyxy

            # Estimate pitch corners from detection
            corners = self._estimate_corners(xyxy, frame_bgr.shape, drs_line)

            # Cross-check alignment with DRS line
            align_angle = 0.0
            lat_offset = 0.0
            notes = ["Pitch boundaries successfully detected."]

            if drs_line is not None and drs_line.is_valid:
                # Longitudinal pitch angle
                pitch_center_near = ((corners.pitch_left_near[0] + corners.pitch_right_near[0]) / 2.0,
                                     (corners.pitch_left_near[1] + corners.pitch_right_near[1]) / 2.0)
                pitch_center_far = ((corners.pitch_left_far[0] + corners.pitch_right_far[0]) / 2.0,
                                    (corners.pitch_left_far[1] + corners.pitch_right_far[1]) / 2.0)

                p_dx = pitch_center_far[0] - pitch_center_near[0]
                p_dy = pitch_center_far[1] - pitch_center_near[1]
                p_angle = math.degrees(math.atan2(p_dy, p_dx))
                align_angle = abs(p_angle - drs_line.angle_deg)
                corners.alignment_error_deg = align_angle

                # Check lateral offset of DRS line relative to pitch center
                lat_offset = drs_line.lateral_distance_px(pitch_center_near)
                if align_angle > 12.0:
                    notes.append(f"Pitch orientation diverges from DRS line by {align_angle:.1f}°.")
                    corners.is_valid = False
                else:
                    corners.is_valid = True

            return PitchDetectionResult(
                is_detected=True,
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                corners=corners,
                confidence=c,
                alignment_angle_deg=align_angle,
                lateral_offset_px=lat_offset,
                notes=notes
            )

        except Exception as e:
            logger.error(f"Error during pitch detection: {e}")
            return PitchDetectionResult(is_detected=False, notes=[f"Pitch inference error: {e}"])

    def _estimate_corners(
        self,
        bbox: List[float],
        img_shape: Tuple[int, ...],
        drs_line: Optional[DRSLine] = None
    ) -> PitchCorners:
        """
        Estimates the 4 pitch corners from the detected bounding box,
        taking into account perspective foreshortening and DRS orientation.
        """
        x1, y1, x2, y2 = bbox
        h, w = img_shape[:2]

        # In standard perspective, the far end of the pitch is narrower in image pixels
        # than the near end. Pitch width ratio far:near is approximately 0.4 to 0.7 depending on lens.
        # If DRS line is available, anchor corners at near and far wicket base heights.
        if drs_line is not None and drs_line.is_valid:
            y_near = drs_line.near_point[1]
            y_far = drs_line.far_point[1]
            cx_near = drs_line.near_point[0]
            cx_far = drs_line.far_point[0]

            # Approximate pitch half-width in image pixels:
            # Stumps width is ~0.228m, pitch width is ~3.048m (ratio ~ 13.3)
            # Estimate near width from bbox
            w_near = (x2 - x1) * 0.45
            w_far = w_near * 0.42

            c_pln = (max(0.0, cx_near - w_near), min(float(h), y_near))
            c_prn = (min(float(w), cx_near + w_near), min(float(h), y_near))
            c_plf = (max(0.0, cx_far - w_far), max(0.0, y_far))
            c_prf = (min(float(w), cx_far + w_far), max(0.0, y_far))
        else:
            # Fallback estimation using trapezoid perspective from bbox
            w_mid = (x2 - x1)
            cx = (x1 + x2) / 2.0
            w_near = w_mid * 0.48
            w_far = w_mid * 0.22

            c_pln = (max(0.0, cx - w_near), y2)
            c_prn = (min(float(w), cx + w_near), y2)
            c_plf = (max(0.0, cx - w_far), y1)
            c_prf = (min(float(w), cx + w_far), y1)

        return PitchCorners(
            pitch_left_near=c_pln,
            pitch_right_near=c_prn,
            pitch_left_far=c_plf,
            pitch_right_far=c_prf,
            confidence=0.80,
            source="model_bbox_perspective",
            is_valid=True
        )
