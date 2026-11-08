"""
Stumps Detector: Detects cricket wickets (near and far stumps).
Verifies alignment with manual calibration and flags scene or camera drift.

CRITICAL ARCHITECTURAL GUARANTEE:
Stumps detection must NEVER silently change, update, or overwrite the manual eight-point calibration.
The manual points are mathematically locked to the video. Stumps detections are strictly used
for camera drift monitoring and diagnostic sanity checking.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Callable
import os
import copy
import logging
import cv2
import numpy as np

from .class_mapper import ModelClassResolver, DEFAULT_STUMPS_KEYWORDS
from .model_validator import ModelValidator

logger = logging.getLogger("CricketBallSpeed.StumpsDetector")


class CalibrationSecurityError(RuntimeError):
    """Raised when an illegal attempt is made to alter locked manual calibration with detections."""
    pass


@dataclass
class StumpsObservation:
    """Represents detected cricket stumps."""
    frame_index: int
    timestamp_sec: float
    bbox: List[float]       # [x1, y1, x2, y2] in original image coordinates
    confidence: float
    stump_type: str         # "near", "far", or "unknown"
    detection_source: str = "full_frame"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 4),
            "bbox": [round(c, 2) for c in self.bbox],
            "confidence": round(self.confidence, 4),
            "stump_type": self.stump_type,
            "detection_source": self.detection_source
        }


class StumpsDetector:
    """
    Detects wickets in video frames.
    Used exclusively for camera stability checks, pitch alignment, and scene drift checks.
    """

    def __init__(
        self,
        model_path: str = "models/stumps.pt",
        confidence_threshold: float = 0.30,
        input_size: Tuple[int, int] = (640, 640),
        target_classes: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None,
        device: str = "auto",
        mock_predictor: Optional[Callable[[np.ndarray], List[Tuple[List[float], float, int]]]] = None
    ):
        self.model_path = model_path
        self.conf_thresh = confidence_threshold
        self.input_size = input_size
        self.explicit_target_classes = target_classes
        self.explicit_class_names = class_names
        self.mock_predictor = mock_predictor

        self.device, self.cuda_available, self.gpu_name = ModelValidator.resolve_device(device)
        self._model = None
        self._framework = None
        self.target_classes: List[int] = [0]
        self.all_classes: Dict[int, str] = {}
        self.last_raw_candidates: List[StumpsObservation] = []
        self._injected_detections: Dict[int, List[StumpsObservation]] = {}

        self._load_model()

    def _load_model(self):
        if self.mock_predictor is not None:
            self._framework = "mock"
            logger.info("StumpsDetector initialized with mock predictor.")
            return

        if not os.path.isfile(self.model_path):
            logger.warning(f"Stumps model not found at {self.model_path}. Running in stub/injection mode.")
            self._framework = "stub"
            return

        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self._framework = "ultralytics"
            logger.info(f"Loaded stumps YOLO model from {self.model_path}.")
        except ImportError:
            try:
                import torch
                checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
                self._model = checkpoint
                self._framework = "torch_checkpoint"
            except Exception as e:
                logger.warning(f"Could not load stumps model weights ({e}). Initialized in fallback mode.")
                self._framework = "stub"

        resolution = ModelClassResolver.resolve_classes(
            model=self._model,
            target_keywords=DEFAULT_STUMPS_KEYWORDS,
            explicit_class_ids=self.explicit_target_classes,
            explicit_class_names=self.explicit_class_names,
            fallback_class_ids=[0],
            entity_name="stumps"
        )
        self.target_classes = resolution.target_class_ids
        self.all_classes = resolution.all_model_classes
        logger.info(f"StumpsDetector configured with target class IDs: {self.target_classes}")

    def inject_detections(self, frame_idx: int, obs_list: List[StumpsObservation]):
        """Injects synthetic or ground-truth stumps observations for automated tests."""
        self._injected_detections[frame_idx] = obs_list

    def detect(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp_sec: float
    ) -> List[StumpsObservation]:
        """Detects wickets in the current frame."""
        if frame_idx in self._injected_detections:
            injected = self._injected_detections[frame_idx]
            self.last_raw_candidates = list(injected)
            return list(injected)

        detections: List[StumpsObservation] = []
        h, w = frame_bgr.shape[:2]

        if self.mock_predictor is not None:
            raw_dets = self.mock_predictor(frame_bgr)
            for xyxy, conf, cls_id in raw_dets:
                if self.target_classes and cls_id not in self.target_classes:
                    continue
                if conf < self.conf_thresh:
                    continue
                x1, y1, x2, y2 = xyxy
                cy = (y1 + y2) / 2.0
                stump_type = "near" if cy > h * 0.50 else "far"
                detections.append(StumpsObservation(
                    frame_index=frame_idx,
                    timestamp_sec=timestamp_sec,
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    confidence=float(conf),
                    stump_type=stump_type,
                    detection_source="mock"
                ))

        elif self._framework == "ultralytics" and self._model is not None:
            results = self._model.predict(
                frame_bgr,
                conf=self.conf_thresh,
                imgsz=self.input_size,
                device=self.device if self.device != "auto" else None,
                verbose=False
            )
            for res in results:
                if res.boxes is not None and len(res.boxes) > 0:
                    for box in res.boxes:
                        cls_id = int(box.cls[0].item())
                        if self.target_classes and cls_id not in self.target_classes:
                            continue
                        conf = float(box.conf[0].item())
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                        cy = (y1 + y2) / 2.0
                        stump_type = "near" if cy > h * 0.50 else "far"
                        detections.append(StumpsObservation(
                            frame_index=frame_idx,
                            timestamp_sec=timestamp_sec,
                            bbox=[float(x1), float(y1), float(x2), float(y2)],
                            confidence=conf,
                            stump_type=stump_type,
                            detection_source="full_frame"
                        ))

        self.last_raw_candidates = list(detections)
        return detections

    def verify_against_calibration(
        self,
        stumps_detections: List[StumpsObservation],
        manual_calibration_points: Dict[str, Tuple[float, float]],
        max_drift_threshold_px: float = 35.0
    ) -> Dict[str, Any]:
        """
        Performs drift verification between detected stumps and manual calibration points.
        GUARANTEE: Returns an audit dictionary. Does NOT modify manual_calibration_points in any way.
        """
        # Deep copy snapshot to verify immutability
        original_snapshot = copy.deepcopy(manual_calibration_points)

        drift_report = {
            "is_drift_detected": False,
            "max_drift_px": 0.0,
            "near_drift_px": None,
            "far_drift_px": None,
            "calibration_preserved": True
        }

        for det in stumps_detections:
            x1, y1, x2, y2 = det.bbox
            det_cx = (x1 + x2) / 2.0
            det_bottom_y = y2

            if det.stump_type == "near" and "near_base" in manual_calibration_points:
                cal_x, cal_y = manual_calibration_points["near_base"]
                drift = float(np.hypot(det_cx - cal_x, det_bottom_y - cal_y))
                drift_report["near_drift_px"] = round(drift, 2)
                drift_report["max_drift_px"] = max(drift_report["max_drift_px"], drift)

            elif det.stump_type == "far" and "far_base" in manual_calibration_points:
                cal_x, cal_y = manual_calibration_points["far_base"]
                drift = float(np.hypot(det_cx - cal_x, det_bottom_y - cal_y))
                drift_report["far_drift_px"] = round(drift, 2)
                drift_report["max_drift_px"] = max(drift_report["max_drift_px"], drift)

        if drift_report["max_drift_px"] > max_drift_threshold_px:
            drift_report["is_drift_detected"] = True
            logger.warning(
                f"Camera drift detected! Max stumps drift is {drift_report['max_drift_px']:.1f}px "
                f"(threshold: {max_drift_threshold_px}px). Manual calibration points remain intact."
            )

        # Enforce that manual calibration points were not mutated
        if manual_calibration_points != original_snapshot:
            raise CalibrationSecurityError(
                "SECURITY VIOLATION: Manual calibration points were modified by stumps detector!"
            )

        return drift_report
