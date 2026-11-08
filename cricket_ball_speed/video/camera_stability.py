"""
Camera Stability Analyzer: Detects camera movement, shake, vibration, or zoom.
Ensures fixed-camera assumptions remain valid throughout the bowling delivery.
"""

from typing import Dict, Any, List, Tuple
import cv2
import numpy as np


class CameraStabilityAnalyzer:
    """
    Evaluates frame-to-frame geometric stability using sparse optical flow (Lucas-Kanade).
    Detects whether the camera panned, tilted, vibrated, or zoomed during recording.
    """

    def __init__(
        self,
        max_displacement_px: float = 3.5,
        max_zoom_scale_delta: float = 0.03,
        num_features: int = 150,
        sample_stride: int = 5
    ):
        self.max_displacement_px = max_displacement_px
        self.max_zoom_scale_delta = max_zoom_scale_delta
        self.num_features = num_features
        self.sample_stride = sample_stride
        self.displacements: List[float] = []
        self.scale_factors: List[float] = []
        self._prev_gray: np.ndarray = None
        self._prev_pts: np.ndarray = None
        self._is_stable: bool = True
        self._max_observed_drift: float = 0.0

    def process_frame(self, frame_idx: int, frame_bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Processes a video frame to evaluate camera stability relative to earlier frames.
        Returns (is_stable, current_displacement).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            # Detect strong corner features for tracking
            self._prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.num_features, qualityLevel=0.01, minDistance=20
            )
            return True, 0.0

        # Subsample checks according to sample_stride to keep processing fast
        if frame_idx % self.sample_stride != 0:
            return self._is_stable, self.displacements[-1] if self.displacements else 0.0

        if self._prev_pts is None or len(self._prev_pts) < 10:
            self._prev_pts = cv2.goodFeaturesToTrack(
                self._prev_gray, maxCorners=self.num_features, qualityLevel=0.01, minDistance=20
            )
            if self._prev_pts is None or len(self._prev_pts) < 10:
                return True, 0.0

        # Compute optical flow
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        good_prev = self._prev_pts[status == 1]
        good_next = next_pts[status == 1]

        if len(good_prev) < 8:
            # Not enough tracked points to establish camera movement
            return self._is_stable, 0.0

        # Estimate affine or rigid transform between tracked background points
        # to separate camera motion from foreground bowler motion (RANSAC)
        transform, inliers = cv2.estimateAffinePartial2D(
            good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=2.0
        )

        displacement = 0.0
        scale = 1.0

        if transform is not None:
            # Translation vector (dx, dy)
            dx = transform[0, 2]
            dy = transform[1, 2]
            displacement = float(np.hypot(dx, dy))
            # Scale factor (zoom)
            scale = float(np.hypot(transform[0, 0], transform[0, 1]))

        self.displacements.append(displacement)
        self.scale_factors.append(scale)
        if displacement > self._max_observed_drift:
            self._max_observed_drift = displacement

        # Check thresholds
        zoom_delta = abs(scale - 1.0)
        if displacement > self.max_displacement_px or zoom_delta > self.max_zoom_scale_delta:
            self._is_stable = False

        # Update previous frame and points
        self._prev_gray = gray
        self._prev_pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.num_features, qualityLevel=0.01, minDistance=20
        )

        return self._is_stable, displacement

    @property
    def is_stable(self) -> bool:
        return self._is_stable

    @property
    def max_observed_drift(self) -> float:
        return self._max_observed_drift

    def get_report(self) -> Dict[str, Any]:
        mean_disp = float(np.mean(self.displacements)) if self.displacements else 0.0
        p95_disp = float(np.percentile(self.displacements, 95)) if self.displacements else 0.0
        return {
            "is_stable": self._is_stable,
            "max_drift_px": round(self._max_observed_drift, 3),
            "mean_drift_px": round(mean_disp, 3),
            "p95_drift_px": round(p95_disp, 3),
            "threshold_px": self.max_displacement_px,
            "samples_checked": len(self.displacements)
        }
