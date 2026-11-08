"""
Detection Subsystem: Multi-model AI detection for cricket ball, bowler, and stumps.
"""

from .class_mapper import ModelClassResolver, ClassResolutionResult
from .model_validator import ModelValidator, ValidationReport, ModelInfo
from .tiled_inference import (
    compute_tiles,
    restore_tile_coordinates,
    apply_nms,
    merge_candidate_observations,
    TiledInferenceEngine
)
from .ball_detector import BallDetector
from .bowler_detector import BowlerDetector, BowlerObservation
from .stumps_detector import StumpsDetector, StumpsObservation, CalibrationSecurityError
from .pitch_detector import PitchDetector
from .detection_manager import DetectionManager
from .detection_visualizer import DetectionVisualizer

__all__ = [
    "ModelClassResolver",
    "ClassResolutionResult",
    "ModelValidator",
    "ValidationReport",
    "ModelInfo",
    "compute_tiles",
    "restore_tile_coordinates",
    "apply_nms",
    "merge_candidate_observations",
    "TiledInferenceEngine",
    "BallDetector",
    "BowlerDetector",
    "BowlerObservation",
    "StumpsDetector",
    "StumpsObservation",
    "CalibrationSecurityError",
    "PitchDetector",
    "DetectionManager",
    "DetectionVisualizer",
]
