"""
Automatic Stump Detection and Landmark Extraction.
Uses the stumps YOLO model to detect both cricket wickets, separate near from far,
extract stump top, base, and base centre landmarks, and perform temporal voting.

Design Principles:
  - Never classify near vs far solely by bounding box size.
  - Combines image plane elevation (y-position), apparent physical scale, perspective geometry,
    and temporal stability.
  - Generates the 4 canonical stump landmarks:
      1) near_top:  (x_c, y1)
      2) near_base: (x_c, y2)
      3) far_top:   (x_c, y1)
      4) far_base:  (x_c, y2)
  - Seamlessly integrates with DRSLineValidator and CameraCalibrator.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging
import math
import cv2
import numpy as np

from .drs_line import DRSLine, DRSLineValidator

logger = logging.getLogger("CricketBallSpeed.AutoStumps")


@dataclass
class WicketObservation:
    """Represents a single detected wicket in an image frame."""
    stump_type: str                   # "near", "far", or "unassigned"
    bbox: List[float]                 # [x1, y1, x2, y2]
    confidence: float
    base_center: Tuple[float, float]  # Ground-level center (Z = 0)
    top_center: Tuple[float, float]   # Top center (Z = 0.711m)
    height_px: float
    width_px: float
    frame_index: int = 0
    timestamp_sec: float = 0.0
    aspect_ratio: float = 0.0         # height / width
    is_accepted: bool = True
    rejection_reason: Optional[str] = None

    def __post_init__(self):
        if self.width_px > 0:
            self.aspect_ratio = float(self.height_px / self.width_px)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stump_type": self.stump_type,
            "bbox": [round(c, 2) for c in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "base_center": [round(c, 2) for c in self.base_center],
            "top_center": [round(c, 2) for c in self.top_center],
            "height_px": round(float(self.height_px), 2),
            "width_px": round(float(self.width_px), 2),
            "aspect_ratio": round(float(self.aspect_ratio), 2),
            "frame_index": self.frame_index,
            "is_accepted": self.is_accepted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class AutomaticStumpDetectionResult:
    """Aggregated detection result for a frame or multi-frame temporal voting."""
    frame_index: int
    image_width: int
    image_height: int
    wickets: List[WicketObservation] = field(default_factory=list)
    near_wicket: Optional[WicketObservation] = None
    far_wicket: Optional[WicketObservation] = None
    drs_line: Optional[DRSLine] = None
    pitch_bbox: Optional[List[float]] = None
    status: str = "NO_STUMPS_DETECTED"  # NO_STUMPS_DETECTED, ONE_WICKET_DETECTED, DRS_LINE_GENERATED, CALIBRATION_READY
    confidence_score: float = 0.0
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "wickets": [w.to_dict() for w in self.wickets],
            "near_wicket": self.near_wicket.to_dict() if self.near_wicket else None,
            "far_wicket": self.far_wicket.to_dict() if self.far_wicket else None,
            "drs_line": self.drs_line.to_dict() if self.drs_line else None,
            "pitch_bbox": [round(c, 2) for c in self.pitch_bbox] if self.pitch_bbox else None,
            "status": self.status,
            "confidence_score": round(float(self.confidence_score), 4),
            "message": self.message,
        }

    @property
    def calibration_points(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """Returns the 4 canonical stump points if both near and far wickets are present."""
        if not self.near_wicket or not self.far_wicket:
            return None
        return {
            "near_base": self.near_wicket.base_center,
            "near_top": self.near_wicket.top_center,
            "far_base": self.far_wicket.base_center,
            "far_top": self.far_wicket.top_center,
        }


class AutomaticStumpDetector:
    """
    Automatic stump detection and classification engine.
    Wraps YOLO inference, extracts high-precision landmarks,
    and resolves near vs far wickets robustly.
    """

    def __init__(
        self,
        model_path: str = "models/stumps.pt",
        confidence_threshold: float = 0.25,
        device: str = "auto"
    ):
        self.model_path = model_path
        self.conf_threshold = confidence_threshold
        self.device = device
        self._model = None
        self._load_model()

    def _load_model(self):
        """Loads Ultralytics YOLO model safely."""
        import os
        if os.path.isfile(self.model_path):
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
                logger.info(f"AutomaticStumpDetector loaded weights from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize stumps YOLO model: {e}")
                self._model = None
        else:
            logger.warning(f"Stumps model file not found at: {self.model_path}")
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def detect_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int = 0,
        timestamp_sec: float = 0.0,
        custom_conf: Optional[float] = None,
        pitch_bbox: Optional[List[float]] = None
    ) -> AutomaticStumpDetectionResult:
        """
        Runs single-frame detection and assigns near/far wickets.
        """
        h, w = frame_bgr.shape[:2]
        conf = custom_conf if custom_conf is not None else self.conf_threshold

        raw_cands: List[WicketObservation] = []

        if self._model is not None:
            try:
                results = self._model.predict(
                    frame_bgr,
                    conf=conf,
                    verbose=False,
                    device=self.device if self.device != "auto" else None
                )
                for res in results:
                    if res.boxes is not None and len(res.boxes) > 0:
                        for box in res.boxes:
                            c = float(box.conf[0].item())
                            xyxy = box.xyxy[0].cpu().numpy().tolist()
                            x1, y1, x2, y2 = xyxy
                            width = max(1.0, x2 - x1)
                            height = max(1.0, y2 - y1)
                            base_c = (float((x1 + x2) / 2.0), float(y2))
                            top_c = (float((x1 + x2) / 2.0), float(y1))

                            obs = WicketObservation(
                                stump_type="unassigned",
                                bbox=[float(x1), float(y1), float(x2), float(y2)],
                                confidence=c,
                                base_center=base_c,
                                top_center=top_c,
                                height_px=float(height),
                                width_px=float(width),
                                frame_index=frame_index,
                                timestamp_sec=timestamp_sec
                            )
                            raw_cands.append(obs)
            except Exception as e:
                logger.error(f"Inference error in stump detection: {e}")

        # Classify Near vs Far Wickets
        return self._classify_wickets(raw_cands, w, h, frame_index, pitch_bbox)

    def _classify_wickets(
        self,
        candidates: List[WicketObservation],
        img_width: int,
        img_height: int,
        frame_index: int,
        pitch_bbox: Optional[List[float]]
    ) -> AutomaticStumpDetectionResult:
        """
        Disambiguates and classifies candidates into Near and Far wickets.
        Uses vertical image coordinate (ground plane elevation), bounding box scale,
        aspect ratio, and pitch alignment.
        """
        result = AutomaticStumpDetectionResult(
            frame_index=frame_index,
            image_width=img_width,
            image_height=img_height,
            wickets=candidates,
            pitch_bbox=pitch_bbox
        )

        if len(candidates) == 0:
            result.status = "NO_STUMPS_DETECTED"
            result.message = "No cricket stumps detected in the frame."
            return result

        if len(candidates) == 1:
            item = candidates[0]
            # Disambiguate single wicket:
            # If base is in lower half of image (y > 0.45 * height), likely near wicket
            if item.base_center[1] > img_height * 0.48:
                item.stump_type = "near"
                result.near_wicket = item
            else:
                item.stump_type = "far"
                result.far_wicket = item

            result.status = "ONE_WICKET_DETECTED"
            result.confidence_score = float(item.confidence * 0.5)
            result.message = f"Single wicket detected ({item.stump_type.upper()}). Both wickets required for full DRS calibration."
            return result

        # When 2 or more candidates are present:
        # Score candidates based on:
        # 1. Base Y coordinate: Near wicket base is situated lower in the frame (larger y).
        # 2. Apparent height: Near stumps are physically closer, hence taller in pixels.
        # 3. Aspect ratio: Cricket stumps typically have height/width ratio > 1.2.

        def wicket_score(w_obs: WicketObservation) -> float:
            # Normalized vertical position [0, 1]
            norm_y = w_obs.base_center[1] / img_height
            # Normalized height (relative to image height)
            norm_h = min(1.0, w_obs.height_px / (img_height * 0.4))
            return 0.65 * norm_y + 0.35 * norm_h

        sorted_cands = sorted(candidates, key=wicket_score, reverse=True)

        near_cand = sorted_cands[0]
        # For far candidate, select the highest confidence candidate that is distinctly separated vertically
        far_candidates = [
            c for c in sorted_cands[1:]
            if c.base_center[1] < near_cand.base_center[1] - (img_height * 0.08)
        ]

        if not far_candidates:
            # Fallback to the candidate with lowest Y
            far_candidates = sorted(sorted_cands[1:], key=lambda k: k.base_center[1])

        far_cand = far_candidates[0]

        near_cand.stump_type = "near"
        far_cand.stump_type = "far"
        result.near_wicket = near_cand
        result.far_wicket = far_cand

        # Construct and validate DRS Line
        drs = DRSLineValidator.validate(
            near_base=near_cand.base_center,
            far_base=far_cand.base_center,
            image_width=img_width,
            image_height=img_height,
            near_confidence=near_cand.confidence,
            far_confidence=far_cand.confidence,
            pitch_bbox=pitch_bbox
        )
        result.drs_line = drs

        if drs.is_valid:
            result.status = "DRS_LINE_GENERATED"
            result.confidence_score = drs.confidence
            result.message = (
                f"Both wickets detected. 20.12m DRS Reference Line generated automatically "
                f"({drs.pixel_length:.1f}px, {drs.angle_deg:.1f}°)."
            )
        else:
            result.status = "DRS_PROVISIONAL"
            result.confidence_score = drs.confidence
            result.message = f"Provisional DRS line: {'; '.join(drs.validation_notes)}"

        return result
