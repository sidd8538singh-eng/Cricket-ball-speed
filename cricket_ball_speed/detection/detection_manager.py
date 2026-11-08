"""
Detection Manager: Coordinates multi-model AI inference across video frames.
Orchestrates ball detection (full-frame & tiled), bowler tracking, stumps checks,
and maintains complete audit trails for raw candidates and detection sources.

Enforces critical constraints:
1. Low-confidence ball candidates are preserved for temporal trajectory verification.
2. Bowler bounding box is strictly body-level, never confused with release hand landmark.
3. Stumps detection is read-only and never mutates manual 8-point calibration.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import cv2
import numpy as np

from ball_processing.ball_observations import BallObservation
from .ball_detector import BallDetector
from .bowler_detector import BowlerDetector, BowlerObservation
from .stumps_detector import StumpsDetector, StumpsObservation
from .pitch_detector import PitchDetector
from .model_validator import ModelValidator, ValidationReport

logger = logging.getLogger("CricketBallSpeed.DetectionManager")


class DetectionManager:
    """
    Central coordinator for model inferences.
    Maintains raw candidate histories and provides structured access to observations.
    """

    def __init__(
        self,
        ball_detector: BallDetector,
        bowler_detector: Optional[BowlerDetector] = None,
        stumps_detector: Optional[StumpsDetector] = None,
        pitch_detector: Optional[PitchDetector] = None,
        stumps_interval: int = 10
    ):
        self.ball_detector = ball_detector
        self.bowler_detector = bowler_detector
        self.stumps_detector = stumps_detector
        self.pitch_detector = pitch_detector
        self.stumps_interval = stumps_interval

        # Structured historical stores
        self.raw_ball_observations: Dict[int, List[BallObservation]] = {}
        self.merged_ball_observations: Dict[int, List[BallObservation]] = {}
        self.bowler_observations: Dict[int, BowlerObservation] = {}
        self.stumps_observations: Dict[int, List[StumpsObservation]] = {}
        self.pitch_observations: Dict[int, Dict[str, Any]] = {}

    def validate_system_models(self, raise_on_error: bool = False) -> ValidationReport:
        """Validates all configured models and reported devices."""
        paths = {
            "ball": self.ball_detector.model_path,
            "bowler": self.bowler_detector.model_path if self.bowler_detector else "",
            "stumps": self.stumps_detector.model_path if self.stumps_detector else "",
            "pitch": self.pitch_detector.model_path if self.pitch_detector else ""
        }
        return ModelValidator.validate_models(
            model_paths=paths,
            requested_device=self.ball_detector.device,
            raise_on_error=raise_on_error
        )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp_sec: float
    ) -> Dict[str, Any]:
        """
        Executes detector pipeline on a single frame.
        Preserves all raw detections, source tags, and coordinates.
        """
        # 1. Ball candidates (full frame + overlapping tiles)
        merged_balls = self.ball_detector.detect(frame_bgr, frame_idx, timestamp_sec)
        raw_balls = list(self.ball_detector.last_raw_candidates)

        self.merged_ball_observations[frame_idx] = merged_balls
        self.raw_ball_observations[frame_idx] = raw_balls

        # 2. Bowler candidate
        bowler_cand: Optional[BowlerObservation] = None
        if self.bowler_detector is not None:
            bowler_cand = self.bowler_detector.detect(frame_bgr, frame_idx, timestamp_sec)
            if bowler_cand is not None:
                # Enforce boundary: never hand landmark
                assert bowler_cand.is_hand_landmark is False, (
                    "BowlerDetector output must have is_hand_landmark=False."
                )
                self.bowler_observations[frame_idx] = bowler_cand

        # 3. Stumps candidate (periodic check for scene drift)
        stumps_cands: List[StumpsObservation] = []
        if self.stumps_detector is not None and (frame_idx % self.stumps_interval == 0 or frame_idx == 0):
            stumps_cands = self.stumps_detector.detect(frame_bgr, frame_idx, timestamp_sec)
            if stumps_cands:
                self.stumps_observations[frame_idx] = stumps_cands

        # 4. Optional pitch detector
        pitch_cand: Optional[Dict[str, Any]] = None
        if self.pitch_detector is not None and frame_idx == 0:
            pitch_cand = self.pitch_detector.detect(frame_bgr, frame_idx)
            if pitch_cand:
                self.pitch_observations[frame_idx] = pitch_cand

        return {
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp_sec,
            "ball_candidates": merged_balls,
            "raw_ball_candidates": raw_balls,
            "bowler": bowler_cand,
            "stumps": stumps_cands,
            "pitch": pitch_cand
        }

    def get_all_ball_observations(self, include_raw: bool = False) -> List[BallObservation]:
        """Returns flattened list of all ball observations (merged or raw)."""
        source_dict = self.raw_ball_observations if include_raw else self.merged_ball_observations
        all_obs: List[BallObservation] = []
        for frame_idx in sorted(source_dict.keys()):
            all_obs.extend(source_dict[frame_idx])
        return all_obs

    def get_summary(self) -> Dict[str, Any]:
        """Returns statistical overview of detections across all processed frames."""
        total_raw_balls = sum(len(v) for v in self.raw_ball_observations.values())
        total_merged_balls = sum(len(v) for v in self.merged_ball_observations.values())
        frames_with_ball = sum(1 for v in self.merged_ball_observations.values() if len(v) > 0)
        return {
            "total_frames_processed": len(self.raw_ball_observations),
            "total_raw_ball_candidates": total_raw_balls,
            "total_merged_ball_detections": total_merged_balls,
            "frames_with_ball": frames_with_ball,
            "bowler_detected_frames": len(self.bowler_observations),
            "stumps_checked_frames": len(self.stumps_observations)
        }
