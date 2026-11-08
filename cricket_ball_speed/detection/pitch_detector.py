"""
Pitch Detector: Optional model for detecting the pitch surface and boundaries.
Supplementary to manual calibration; does not replace manual pitch corner selection.
"""

from typing import List, Optional, Dict, Any
import os
import logging
import numpy as np


logger = logging.getLogger("CricketBallSpeed.PitchDetector")


class PitchDetector:
    """Optional detector for pitch segmentation or bounding box."""

    def __init__(
        self,
        model_path: str = "models/pitch.pt",
        confidence_threshold: float = 0.30,
        device: str = "auto"
    ):
        self.model_path = model_path
        self.conf_thresh = confidence_threshold
        self.device = device
        self._model = None
        self._framework = None

        if os.path.isfile(model_path):
            try:
                from ultralytics import YOLO
                self._model = YOLO(model_path)
                self._framework = "ultralytics"
            except ImportError:
                self._framework = "stub"
        else:
            self._framework = "disabled"

    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> Optional[Dict[str, Any]]:
        """Optional pitch boundary detection."""
        if self._framework != "ultralytics" or self._model is None:
            return None
        try:
            results = self._model.predict(frame_bgr, conf=self.conf_thresh, verbose=False)
            if results and len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                return {
                    "bbox": box.xyxy[0].cpu().numpy().tolist(),
                    "confidence": float(box.conf[0].item())
                }
        except Exception as e:
            logger.debug(f"Pitch detection error: {e}")
        return None
