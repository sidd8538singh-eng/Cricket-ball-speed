"""
Zoota-Style Cricket Stump Detection, DRS Reference Line & Calibration Engine.
Integrates directly with cricket_ball_speed models, ICC geometry, and camera calibration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import os
import sys
import logging
import cv2
import numpy as np

# Ensure cricket_ball_speed is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
CBS_DIR = os.path.join(ROOT_DIR, "cricket_ball_speed")
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if CBS_DIR not in sys.path:
    sys.path.insert(0, CBS_DIR)

from calibration.world_coordinates import WorldPitchGeometry, CALIBRATION_POINT_KEYS
from calibration.camera_model import CameraModel
from calibration.camera_calibration import CameraCalibrator
from calibration.homography import GroundPlaneHomography
from calibration.calibration_validator import CalibrationValidator
from calibration.drs_line import DRSLine, DRSLineValidator
from calibration.automatic_stump_calibration import (
    AutomaticStumpDetector, AutomaticStumpDetectionResult, WicketObservation
)
from calibration.pitch_geometry import PitchGeometryEngine, PitchDetectionResult, PitchCorners
from calibration.calibration_manager import CalibrationManager, CalibrationResult

logger = logging.getLogger("ZootaDRS.Engine")


@dataclass
class DetectedWicket:
    """Represents a detected cricket wicket."""
    stump_type: str         # "near" or "far"
    bbox: List[float]       # [x1, y1, x2, y2]
    confidence: float
    base_center: Tuple[float, float]  # Ground-level center (Z = 0)
    top_center: Tuple[float, float]   # Top center (Z = 0.711m)
    height_px: float
    width_px: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stump_type": self.stump_type,
            "bbox": [round(c, 2) for c in self.bbox],
            "confidence": round(float(self.confidence), 4),
            "base_center": [round(c, 2) for c in self.base_center],
            "top_center": [round(c, 2) for c in self.top_center],
            "height_px": round(float(self.height_px), 2),
            "width_px": round(float(self.width_px), 2),
        }


@dataclass
class FrameDetectionResult:
    """Detection results for a single image/video frame."""
    frame_index: int
    image_width: int
    image_height: int
    wickets: List[DetectedWicket]
    near_wicket: Optional[DetectedWicket] = None
    far_wicket: Optional[DetectedWicket] = None
    drs_line: Optional[DRSLine] = None
    pitch_bbox: Optional[List[float]] = None
    status: str = "NO_STUMPS_DETECTED"  # NO_STUMPS_DETECTED, ONE_WICKET_DETECTED, DRS_LINE_GENERATED
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
            "message": self.message,
        }


class DRSCalibrationEngine:
    """
    Core engine for:
      - YOLO Stump detection & near/far assignment
      - Automatic DRS line construction (20.12m constraint)
      - Dual-mode calibration:
          1) Stump + DRS Calibration (without pitch model)
          2) Stump + DRS + Pitch Calibration (with pitch model / corners)
      - Calibration validation & reprojection reporting
    """

    def __init__(
        self,
        stumps_model_path: Optional[str] = None,
        pitch_model_path: Optional[str] = None,
        conf_threshold: float = 0.20,
        device: str = "auto"
    ):
        self.conf_threshold = conf_threshold
        self.geometry = WorldPitchGeometry(
            pitch_length_crease_to_crease_m=20.12,
            pitch_width_m=3.048,
            stump_height_m=0.711,
            ball_diameter_m=0.0715
        )

        # Resolve model paths
        if stumps_model_path is None:
            stumps_model_path = os.path.join(CBS_DIR, "models", "stumps.pt")
        if pitch_model_path is None:
            pitch_model_path = os.path.join(CBS_DIR, "models", "pitch.pt")

        self.stumps_model_path = stumps_model_path
        self.pitch_model_path = pitch_model_path

        # Underlying modular calibration manager
        self.manager = CalibrationManager(
            stumps_model_path=self.stumps_model_path,
            pitch_model_path=self.pitch_model_path if os.path.isfile(self.pitch_model_path) else None,
            storage_dir=os.path.join(ROOT_DIR, "storage", "calibration"),
            pitch_geometry=self.geometry,
            max_reprojection_error_px=6.0,
            device=device
        )

    @property
    def is_stumps_loaded(self) -> bool:
        return self.manager.stump_detector.is_loaded

    @property
    def is_pitch_loaded(self) -> bool:
        return self.manager.pitch_engine.is_loaded

    def detect_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int = 0,
        custom_conf: Optional[float] = None
    ) -> FrameDetectionResult:
        """
        Runs stump detection on a single frame, identifies near and far wickets,
        and automatically constructs the DRS line if both are found.
        """
        h, w = frame_bgr.shape[:2]

        # 1. Pitch detection (optional)
        pitch_res = self.manager.pitch_engine.detect_pitch(frame_bgr, custom_conf=custom_conf)
        pitch_bbox = pitch_res.bbox if pitch_res.is_detected else None

        # 2. Stump detection
        stump_res = self.manager.stump_detector.detect_frame(
            frame_bgr=frame_bgr,
            frame_index=frame_index,
            custom_conf=custom_conf or self.conf_threshold,
            pitch_bbox=pitch_bbox
        )

        # Convert to DetectedWicket format
        wickets: List[DetectedWicket] = []
        near_dw = None
        far_dw = None

        for w_obs in stump_res.wickets:
            dw = DetectedWicket(
                stump_type=w_obs.stump_type,
                bbox=w_obs.bbox,
                confidence=w_obs.confidence,
                base_center=w_obs.base_center,
                top_center=w_obs.top_center,
                height_px=w_obs.height_px,
                width_px=w_obs.width_px
            )
            wickets.append(dw)
            if w_obs.stump_type == "near":
                near_dw = dw
            elif w_obs.stump_type == "far":
                far_dw = dw

        return FrameDetectionResult(
            frame_index=frame_index,
            image_width=w,
            image_height=h,
            wickets=wickets,
            near_wicket=near_dw,
            far_wicket=far_dw,
            drs_line=stump_res.drs_line,
            pitch_bbox=pitch_bbox,
            status=stump_res.status,
            message=stump_res.message
        )

    def calibrate(
        self,
        image_points: Dict[str, Optional[Tuple[float, float]]],
        image_width: int,
        image_height: int,
        initial_focal_length_px: Optional[float] = None,
        video_hash: Optional[str] = None,
        camera_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes camera calibration using:
          - Stump + DRS Calibration (4 points)
          - Stump + DRS + Pitch Calibration (8 points)
        """
        res = self.manager.solve_calibration(
            image_points=image_points,
            image_width=image_width,
            image_height=image_height,
            initial_focal_length_px=initial_focal_length_px,
            video_hash=video_hash,
            camera_id=camera_id
        )
        return res.to_dict()
