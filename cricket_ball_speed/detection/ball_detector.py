"""
Ball Detector: High-precision cricket ball detection.
Combines full-frame inference, overlapping tiled inference, optional upscaling for tiny balls,
coordinate restoration to original video pixels, preservation of low-confidence candidates,
and cross-tile duplicate merging.
"""

from typing import List, Optional, Dict, Any, Tuple, Callable
import os
import logging
import cv2
import numpy as np

from ball_processing.ball_observations import BallObservation
from .tiled_inference import (
    compute_tiles,
    restore_tile_coordinates,
    apply_nms,
    merge_candidate_observations,
    TiledInferenceEngine
)
from .class_mapper import ModelClassResolver, DEFAULT_BALL_KEYWORDS
from .model_validator import ModelValidator

logger = logging.getLogger("CricketBallSpeed.BallDetector")


class BallDetector:
    """
    Detects cricket ball candidates in video frames.
    Preserves all raw detections, bboxes, apparent diameters, and confidence scores.
    Guarantees that low-confidence candidates are preserved for temporal trajectory verification.
    """

    def __init__(
        self,
        model_path: str = "models/ball.pt",
        confidence_threshold: float = 0.15,
        min_candidate_confidence: float = 0.10,
        nms_iou_threshold: float = 0.35,
        input_size: Tuple[int, int] = (640, 640),
        use_tiling: bool = True,
        tile_size: Tuple[int, int] = (640, 640),
        tile_overlap: float = 0.20,
        upscale_tiles: bool = False,
        upscale_factor: float = 1.0,
        device: str = "auto",
        target_classes: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None,
        mock_predictor: Optional[Callable[[np.ndarray], List[Tuple[List[float], float, int]]]] = None
    ):
        self.model_path = model_path
        self.conf_thresh = confidence_threshold
        # Preserve low-confidence candidates for ballistic temporal verification
        self.min_candidate_conf = min(min_candidate_confidence, confidence_threshold)
        self.nms_iou = nms_iou_threshold
        self.input_size = input_size
        self.use_tiling = use_tiling
        self.tile_w, self.tile_h = tile_size
        self.tile_overlap = tile_overlap
        self.upscale_tiles = upscale_tiles
        self.upscale_factor = upscale_factor if upscale_tiles else 1.0
        self.explicit_target_classes = target_classes
        self.explicit_class_names = class_names
        self.mock_predictor = mock_predictor

        # Resolved device
        self.device, self.cuda_available, self.gpu_name = ModelValidator.resolve_device(device)

        # Internal model and class mappings
        self._model = None
        self._framework = None
        self.target_classes: List[int] = [0]
        self.all_classes: Dict[int, str] = {}
        self.last_raw_candidates: List[BallObservation] = []
        self._injected_detections: Dict[int, List[BallObservation]] = {}

        # Tiled engine
        self.tiled_engine = TiledInferenceEngine(
            tile_width=self.tile_w,
            tile_height=self.tile_h,
            overlap_x=self.tile_overlap,
            overlap_y=self.tile_overlap,
            upscale_factor=self.upscale_factor,
            nms_iou_threshold=self.nms_iou,
            min_confidence=self.min_candidate_conf
        )

        self._load_model()

    def _load_model(self):
        """Loads YOLO model via Ultralytics or Torch checkpoint, resolving class mappings."""
        if self.mock_predictor is not None:
            self._framework = "mock"
            logger.info("BallDetector initialized with mock predictor.")
            return

        if not os.path.isfile(self.model_path):
            logger.warning(f"Ball model file not found at: {self.model_path}. Running in stub/injection mode.")
            self._framework = "stub"
            return

        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self._framework = "ultralytics"
            logger.info(f"Loaded ball YOLO model from {self.model_path} via Ultralytics.")
        except ImportError:
            try:
                import torch
                checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
                self._model = checkpoint
                self._framework = "torch_checkpoint"
                logger.info(f"Loaded ball model checkpoint from {self.model_path} via PyTorch.")
            except Exception as e:
                logger.warning(f"Could not load ball model weights ({e}). Initialized in fallback mode.")
                self._framework = "stub"

        # Resolve class mappings dynamically from model metadata
        resolution = ModelClassResolver.resolve_classes(
            model=self._model,
            target_keywords=DEFAULT_BALL_KEYWORDS,
            explicit_class_ids=self.explicit_target_classes,
            explicit_class_names=self.explicit_class_names,
            fallback_class_ids=[0],
            entity_name="ball"
        )
        self.target_classes = resolution.target_class_ids
        self.all_classes = resolution.all_model_classes
        logger.info(f"BallDetector configured with target class IDs: {self.target_classes}")

    def inject_detections(self, frame_idx: int, candidates: List[BallObservation]):
        """Allows injecting synthetic or ground-truth detections for automated tests."""
        self._injected_detections[frame_idx] = candidates

    def _predict_raw(self, img_bgr: np.ndarray) -> List[Tuple[List[float], float, int]]:
        """
        Runs model inference on an image (full frame or tile crop).
        Returns list of (bbox_xyxy, confidence, class_id).
        """
        if self.mock_predictor is not None:
            return self.mock_predictor(img_bgr)

        results_list: List[Tuple[List[float], float, int]] = []
        if self._framework == "ultralytics" and self._model is not None:
            # Predict using low candidate threshold to preserve flight candidates
            preds = self._model.predict(
                img_bgr,
                conf=self.min_candidate_conf,
                imgsz=self.input_size,
                device=self.device if self.device != "auto" else None,
                verbose=False
            )
            for res in preds:
                if res.boxes is not None and len(res.boxes) > 0:
                    for box in res.boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        results_list.append((xyxy, conf, cls_id))

        return results_list

    def detect(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp_sec: float
    ) -> List[BallObservation]:
        """
        Runs complete ball detection on a frame:
        1. Checks for injected synthetic detections (testing).
        2. Full-frame inference.
        3. Overlapping tiled inference (with optional upscaling).
        4. Coordinate restoration to original video pixel space.
        5. Preserves all raw detections.
        6. Merges overlapping duplicate detections via global NMS.
        7. Returns merged observations while preserving low-confidence candidates.
        """
        # Test injection path
        if frame_idx in self._injected_detections:
            injected = self._injected_detections[frame_idx]
            self.last_raw_candidates = list(injected)
            return list(injected)

        h, w = frame_bgr.shape[:2]
        raw_candidates: List[BallObservation] = []

        # 1. Full-frame inference
        full_dets = self._predict_raw(frame_bgr)
        for xyxy, conf, cls_id in full_dets:
            if self.target_classes and cls_id not in self.target_classes:
                continue
            if conf < self.min_candidate_conf:
                continue

            obs = BallObservation.from_bbox(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                bbox=xyxy,
                confidence=conf,
                detection_source="full_frame"
            )
            raw_candidates.append(obs)

        # 2. Overlapping tiled inference (for tiny distant balls)
        if self.use_tiling and (w > self.tile_w or h > self.tile_h):
            _, tile_raw = self.tiled_engine.process_tiled_frame(
                frame_bgr=frame_bgr,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                predict_fn=self._predict_raw,
                target_classes=self.target_classes
            )
            raw_candidates.extend(tile_raw)

        # Store all raw candidates for audit trail before deduplication
        self.last_raw_candidates = list(raw_candidates)

        # 3. Merge duplicate detections across full-frame and overlapping tiles
        if len(raw_candidates) > 1:
            merged_candidates = merge_candidate_observations(raw_candidates, iou_threshold=self.nms_iou)
            return merged_candidates

        return raw_candidates
