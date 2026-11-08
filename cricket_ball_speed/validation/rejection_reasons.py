"""
Rejection Reasons: Canonical enumerations and descriptive messages for unmeasured deliveries.
Provides transparent auditability whenever quality control flags a measurement failure.
"""

from enum import Enum
from typing import Optional


class RejectionReason(str, Enum):
    # Gate 1: Camera Stability
    CAMERA_UNSTABLE = "CAMERA_UNSTABLE: Excessive camera vibration, pan, or zoom detected."

    # Gate 2: Calibration Validity
    CAMERA_CALIBRATION_FAILED = "CAMERA_CALIBRATION_FAILED: Failed to solve valid camera pose."
    PERSPECTIVE_GEOMETRY_INVALID = "PERSPECTIVE_GEOMETRY_INVALID: Calibration points violate perspective geometry."

    # Gate 3: Reprojection Error
    REPROJECTION_ERROR_HIGH = "REPROJECTION_ERROR_HIGH: Landmarks reprojection error exceeds tolerance."

    # Gate 4: Ball Observation Quality
    INSUFFICIENT_BALL_DETECTIONS = "INSUFFICIENT_BALL_DETECTIONS: Too few valid ball observations found."
    OBSERVATION_QUALITY_LOW = "OBSERVATION_QUALITY_LOW: Ball candidate observation quality score below threshold."
    LONG_PREDICTION_GAP = "LONG_PREDICTION_GAP: Excessive consecutive missing ball detection frames."

    # Gate 5: Release Confidence
    RELEASE_NOT_FOUND = "RELEASE_NOT_FOUND: Bowling release event could not be localized."
    RELEASE_CONFIDENCE_LOW = "RELEASE_CONFIDENCE_LOW: Release event confidence below threshold."
    RELEASE_HEIGHT_IMPLAUSIBLE = "RELEASE_HEIGHT_IMPLAUSIBLE: Release height outside athletic bounds."

    # Gate 6: Depth Reconstruction Uncertainty
    DEPTH_UNCERTAINTY_HIGH = "DEPTH_UNCERTAINTY_HIGH: Depth uncertainty along camera rays too large."

    # Gate 7: Trajectory Fitting Residual
    TRAJECTORY_RESIDUAL_HIGH = "TRAJECTORY_RESIDUAL_HIGH: Ballistic trajectory fit residual exceeds limit."
    TRAJECTORY_DIVERGED = "TRAJECTORY_DIVERGED: Mathematical trajectory fit failed to converge."

    # Gate 8: Timestamp Validity & Frame Rate
    TIMESTAMP_IRREGULAR = "TIMESTAMP_IRREGULAR: Severe frame dropping, zero dt, or timestamp jitter detected."
    FRAMERATE_TOO_LOW = "FRAMERATE_TOO_LOW: Video framerate below minimum required standard."

    # Gate 9: Speed Uncertainty
    SPEED_UNCERTAINTY_EXCESSIVE = "SPEED_UNCERTAINTY_EXCESSIVE: Speed uncertainty bounds exceed threshold."

    # Gate 10: Physical & Aerodynamic Plausibility
    SPEED_OUT_OF_RANGE = "SPEED_OUT_OF_RANGE: Speed outside physically plausible bounds."
    AERODYNAMIC_INCONSISTENCY = "AERODYNAMIC_INCONSISTENCY: C1-C2 speed violates aerodynamic deceleration laws."
    BALL_DIRECTION_INVALID = "BALL_DIRECTION_INVALID: Ball trajectory velocity vector is not moving down-pitch."

    # General / Video Input
    CORRUPT_VIDEO = "CORRUPT_VIDEO: Unable to decode video frames."
    SPEED_NOT_COMPUTED = "SPEED_NOT_COMPUTED: Speed estimation could not be derived."

    @classmethod
    def get_code(cls, reason_str: Optional[str]) -> str:
        """Extracts the uppercase category code from a rejection string."""
        if not reason_str:
            return "UNKNOWN"
        parts = reason_str.split(":", 1)
        return parts[0].strip()
