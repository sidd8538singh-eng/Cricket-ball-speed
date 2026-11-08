"""
Release Validator: Validates detected release against physical constraints.
Enforces that release height and position are physically sound.
Reports NOT_MEASURED if release point is ambiguous, unphysical, or unreliable.

CRITICAL ARCHITECTURAL CONSTRAINT:
Release height is checked purely as a post-hoc physical plausibility gate (human biomechanics).
It is NEVER used as a prior, multiplier, or constraint during speed calculation.
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from .release_detector import ReleaseEvent


class ReleaseValidator:
    """
    Validates detected release event against human biomechanical bounds.
    Rejects noisy detections and flags unmeasured deliveries.
    """

    def __init__(
        self,
        min_release_confidence: float = 0.45,
        min_post_release_detections: int = 4,
        min_plausible_height_m: float = 1.40,
        max_plausible_height_m: float = 2.90,
        min_crease_y_m: float = -2.0,
        max_crease_y_m: float = 3.5
    ):
        """
        Args:
            min_release_confidence: Minimum release event confidence.
            min_post_release_detections: Minimum real frames after release.
            min_plausible_height_m: Biomechanical floor for release height (metres).
            max_plausible_height_m: Biomechanical ceiling for release height (metres).
            min_crease_y_m: Minimum crease longitudinal bound (metres).
            max_crease_y_m: Maximum crease longitudinal bound (metres).
        """
        self.min_conf = float(min_release_confidence)
        self.min_post_dets = int(min_post_release_detections)
        self.min_h = float(min_plausible_height_m)
        self.max_h = float(max_plausible_height_m)
        self.min_y = float(min_crease_y_m)
        self.max_y = float(max_crease_y_m)

    def validate_event(
        self,
        event: Optional[ReleaseEvent],
        release_3d_xyz: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates the release event.
        Returns (is_valid, rejection_reason).
        """
        if event is None:
            return False, "RELEASE_NOT_DETECTED: No bowling release event identified."

        if not event.is_valid or event.status == "NOT_MEASURED":
            reason = event.rejection_reason or "RELEASE_UNIDENTIFIED: Could not reliably identify release point."
            return False, f"RELEASE_REJECTED: {reason}"

        if event.confidence < self.min_conf:
            return False, f"RELEASE_LOW_CONFIDENCE: Confidence {event.confidence:.2f} < {self.min_conf:.2f}."

        if event.supporting_detections_count < self.min_post_dets:
            return False, (
                f"INSUFFICIENT_POST_RELEASE_FRAMES: Found only {event.supporting_detections_count} "
                f"real detections after release (minimum required: {self.min_post_dets})."
            )

        # Post-hoc 3D sanity check (strictly sanity check, NOT a speed prior)
        if release_3d_xyz is not None:
            rx, ry, rz = [float(v) for v in release_3d_xyz]
            # Check vertical height above ground
            if not (self.min_h <= rz <= self.max_h):
                return False, (
                    f"RELEASE_HEIGHT_IMPLAUSIBLE: Solved release height {rz:.2f}m is outside "
                    f"human biomechanical bounds [{self.min_h:.2f}, {self.max_h:.2f}]m."
                )
            # Check near-wicket distance (should be near bowling crease: Y between -2.0m and +3.5m)
            if not (self.min_y <= ry <= self.max_y):
                return False, (
                    f"RELEASE_POSITION_IMPLAUSIBLE: Release Y={ry:.2f}m is outside crease delivery zone "
                    f"[{self.min_y:.2f}, {self.max_y:.2f}]m."
                )

        return True, None
