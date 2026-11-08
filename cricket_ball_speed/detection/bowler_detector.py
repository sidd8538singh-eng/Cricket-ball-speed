"""
Bowler Detector: Detects the bowler's full-body bounding box and tracks bowler position.
Provides spatial and temporal gating for bowling action and ball release.

IMPORTANT ARCHITECTURAL REQUIREMENT:
The bowler bounding box represents the athlete's body envelope. It must NEVER be confused
with or substituted for the precise bowling-hand landmark. Precise hand localization is
performed independently by the HandEstimator module.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Callable
import os
import logging
import cv2
import numpy as np

from .class_mapper import ModelClassResolver, DEFAULT_BOWLER_KEYWORDS
from .model_validator import ModelValidator

logger = logging.getLogger("CricketBallSpeed.BowlerDetector")


@dataclass
class BowlerObservation:
    """
    Represents the detected bowler's body bounding box.
    Explicitly marked as is_hand_landmark=False to prevent conflation with the release point.
    """
    frame_index: int
    timestamp_sec: float
    bbox: List[float]               # [x1, y1, x2, y2] in original image coordinates
    confidence: float
    center: List[float]             # [cx, cy]
    width: float
    height: float
    detection_source: str = "full_frame"
    is_hand_landmark: bool = False  # NEVER True for a whole-body bowler bbox!
    landmark_type: str = "body_bbox" # Explicitly distinguished from "hand_landmark"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 4),
            "bbox": [round(c, 2) for c in self.bbox],
            "confidence": round(self.confidence, 4),
            "center": [round(c, 2) for c in self.center],
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "is_hand_landmark": self.is_hand_landmark,
            "landmark_type": self.landmark_type,
            "detection_source": self.detection_source
        }


class BowlerDetector:
    """
    Detects the bowler in each frame.
    Separates the bowler from the umpire or batsman by spatial proximity to the bowling crease.
    """

    def __init__(
        self,
        model_path: str = "models/bowler.pt",
        confidence_threshold: float = 0.35,
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
        self.last_raw_candidates: List[BowlerObservation] = []
        self._injected_detections: Dict[int, Optional[BowlerObservation]] = {}

        self._load_model()

    def _load_model(self):
        if self.mock_predictor is not None:
            self._framework = "mock"
            logger.info("BowlerDetector initialized with mock predictor.")
            return

        if not os.path.isfile(self.model_path):
            logger.warning(f"Bowler model not found at {self.model_path}. Running in stub/injection mode.")
            self._framework = "stub"
            return

        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self._framework = "ultralytics"
            logger.info(f"Loaded bowler YOLO model from {self.model_path}.")
        except ImportError:
            try:
                import torch
                checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
                self._model = checkpoint
                self._framework = "torch_checkpoint"
            except Exception as e:
                logger.warning(f"Could not load bowler model weights ({e}). Initialized in fallback mode.")
                self._framework = "stub"

        resolution = ModelClassResolver.resolve_classes(
            model=self._model,
            target_keywords=DEFAULT_BOWLER_KEYWORDS,
            explicit_class_ids=self.explicit_target_classes,
            explicit_class_names=self.explicit_class_names,
            fallback_class_ids=[0],
            entity_name="bowler"
        )
        self.target_classes = resolution.target_class_ids
        self.all_classes = resolution.all_model_classes
        logger.info(f"BowlerDetector configured with target class IDs: {self.target_classes}")

    def inject_detection(self, frame_idx: int, obs: Optional[BowlerObservation]):
        """Injects synthetic or ground-truth bowler observation for automated testing."""
        self._injected_detections[frame_idx] = obs

    def detect(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp_sec: float
    ) -> Optional[BowlerObservation]:
        """
        Detects the primary bowler in the frame.
        When multiple persons appear (e.g. umpire, batsman), selects the bowler
        in the lower foreground near the delivery stride crease.
        """
        if frame_idx in self._injected_detections:
            injected = self._injected_detections[frame_idx]
            self.last_raw_candidates = [injected] if injected else []
            return injected

        candidates: List[BowlerObservation] = []
        h, w = frame_bgr.shape[:2]

        if self.mock_predictor is not None:
            dets = self.mock_predictor(frame_bgr)
            for xyxy, conf, cls_id in dets:
                if self.target_classes and cls_id not in self.target_classes:
                    continue
                if conf < self.conf_thresh:
                    continue
                x1, y1, x2, y2 = xyxy
                obs = BowlerObservation(
                    frame_index=frame_idx,
                    timestamp_sec=timestamp_sec,
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    confidence=float(conf),
                    center=[float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)],
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                    detection_source="mock",
                    is_hand_landmark=False,
                    landmark_type="body_bbox"
                )
                candidates.append(obs)

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
                        obs = BowlerObservation(
                            frame_index=frame_idx,
                            timestamp_sec=timestamp_sec,
                            bbox=[float(x1), float(y1), float(x2), float(y2)],
                            confidence=float(conf),
                            center=[float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)],
                            width=float(x2 - x1),
                            height=float(y2 - y1),
                            detection_source="full_frame",
                            is_hand_landmark=False,
                            landmark_type="body_bbox"
                        )
                        candidates.append(obs)

        self.last_raw_candidates = list(candidates)

        if not candidates:
            return None

        # Prioritize bowler: largest person in lower half of the frame (near wicket foreground)
        # Score = area * (y_bottom / frame_height)
        candidates.sort(
            key=lambda c: (c.width * c.height) * ((c.bbox[3]) / max(1.0, float(h))),
            reverse=True
        )
        return candidates[0]
