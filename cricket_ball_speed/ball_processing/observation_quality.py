"""
Observation Quality Assessor.
Scores ball candidates based on geometric circularity, plausible pixel size,
detector confidence, and image boundary proximity.
"""

from typing import Tuple
import numpy as np
from .ball_observations import BallObservation


class ObservationQualityAssessor:
    """
    Assigns a quality score [0.0, 1.0] to each ball candidate.
    Discards extreme aspect ratios and non-ball shapes.
    """

    def __init__(
        self,
        min_diameter_px: float = 2.5,
        max_diameter_px: float = 90.0,
        max_aspect_ratio: float = 3.2
    ):
        self.min_diam = min_diameter_px
        self.max_diam = max_diameter_px
        self.max_aspect_ratio = max_aspect_ratio

    def assess(
        self,
        obs: BallObservation,
        frame_width: int,
        frame_height: int
    ) -> float:
        """
        Computes composite quality score. Returns float in [0.0, 1.0].
        """
        w = max(0.1, obs.width)
        h = max(0.1, obs.height)
        aspect = max(w, h) / min(w, h)

        # 1. Aspect ratio penalty
        if aspect > self.max_aspect_ratio:
            aspect_score = 0.0
        else:
            aspect_score = 1.0 - (aspect - 1.0) / (self.max_aspect_ratio - 1.0)
            aspect_score = float(np.clip(aspect_score, 0.1, 1.0))

        # 2. Diameter bounds penalty
        diam = obs.apparent_diameter_px
        if diam < self.min_diam or diam > self.max_diam:
            size_score = 0.0
        else:
            size_score = 1.0

        # 3. Image border proximity penalty
        x1, y1, x2, y2 = obs.bbox
        border_margin = 4.0
        if x1 <= border_margin or y1 <= border_margin or x2 >= (frame_width - border_margin) or y2 >= (frame_height - border_margin):
            border_score = 0.5
        else:
            border_score = 1.0

        # 4. Detector confidence
        conf_score = float(np.clip(obs.confidence, 0.0, 1.0))

        composite = 0.40 * conf_score + 0.30 * aspect_score + 0.20 * size_score + 0.10 * border_score
        obs.quality_score = float(np.clip(composite, 0.0, 1.0))
        return obs.quality_score
