"""
Calibration package.
Provides coordinate frames, camera models, calibration solvers,
automatic stump and pitch calibration, DRS reference lines, and validation.
"""

from .world_coordinates import WorldPitchGeometry, CALIBRATION_POINT_KEYS
from .camera_model import CameraModel
from .camera_calibration import CameraCalibrator
from .calibration_validator import CalibrationValidator
from .reprojection import ReprojectionReport, compute_reprojection_report
from .homography import GroundPlaneHomography
from .drs_line import DRSLine, DRSLineValidator
from .automatic_stump_calibration import (
    AutomaticStumpDetector, AutomaticStumpDetectionResult, WicketObservation
)
from .pitch_geometry import PitchGeometryEngine, PitchDetectionResult, PitchCorners
from .calibration_storage import CalibrationStorage, StoredCalibration
from .calibration_manager import CalibrationManager, CalibrationResult

__all__ = [
    "WorldPitchGeometry",
    "CALIBRATION_POINT_KEYS",
    "CameraModel",
    "CameraCalibrator",
    "CalibrationValidator",
    "ReprojectionReport",
    "compute_reprojection_report",
    "GroundPlaneHomography",
    "DRSLine",
    "DRSLineValidator",
    "AutomaticStumpDetector",
    "AutomaticStumpDetectionResult",
    "WicketObservation",
    "PitchGeometryEngine",
    "PitchDetectionResult",
    "PitchCorners",
    "CalibrationStorage",
    "StoredCalibration",
    "CalibrationManager",
    "CalibrationResult",
]
