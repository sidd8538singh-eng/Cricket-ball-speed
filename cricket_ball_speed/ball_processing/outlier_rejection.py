"""
Outlier Rejection: Filters spurious detections and impossible motion discontinuities.
Preserves raw observations while setting acceptance_status and detailed rejection_reasons.
Scales kinematic thresholds with actual timestamp deltas (nonuniform frame intervals).
"""

from typing import List, Optional
import numpy as np
from .ball_observations import BallObservation


class OutlierRejector:
    """
    Applies kinematic and spatial filters to ball candidates.
    Rejects physically impossible jumps and erratic detections while documenting explicit rejection reasons.
    """

    def __init__(
        self,
        max_interframe_displacement_px: float = 120.0,
        nominal_dt_sec: float = 0.00833,
        min_quality_score: float = 0.25,
        max_diameter_ratio_change: float = 2.5,
        max_velocity_px_s: float = 18000.0
    ):
        """
        Args:
            max_interframe_displacement_px: Displacement limit at nominal frame time (e.g. 120 FPS / 8.33ms).
            nominal_dt_sec: Reference frame interval for displacement scaling.
            min_quality_score: Minimum morphology/circularity score.
            max_diameter_ratio_change: Maximum allowed ratio of apparent diameter between frames.
            max_velocity_px_s: Absolute physical velocity limit in image coordinates (px/s).
        """
        self.max_disp = float(max_interframe_displacement_px)
        self.nominal_dt = float(nominal_dt_sec)
        self.min_quality = float(min_quality_score)
        self.max_diam_ratio = float(max_diameter_ratio_change)
        self.max_vel_px_s = float(max_velocity_px_s)

    def filter_candidates(
        self,
        candidates: List[BallObservation],
        last_accepted: Optional[BallObservation] = None,
        dt_sec: float = 0.00833
    ) -> List[BallObservation]:
        """
        Evaluates a list of candidates in a frame against kinematic constraints.
        Marks rejected candidates with clear reasons.
        Returns accepted candidates.
        """
        accepted = []
        dt = max(1e-4, float(dt_sec))

        for cand in candidates:
            # 1. Quality filter
            if cand.quality_score < self.min_quality:
                cand.mark_rejected(
                    f"Low morphology quality score ({cand.quality_score:.2f} < {self.min_quality:.2f})"
                )
                continue

            # 2. Kinematic continuity check against previous accepted ball
            if last_accepted is not None:
                dx = cand.center[0] - last_accepted.center[0]
                dy = cand.center[1] - last_accepted.center[1]
                disp = float(np.hypot(dx, dy))

                # Compute instantaneous pixel velocity
                v_px_s = disp / dt

                # Scale allowable displacement with actual dt
                allowed_disp = max(self.max_disp, self.max_disp * (dt / self.nominal_dt))

                if disp > allowed_disp or v_px_s > self.max_vel_px_s:
                    cand.mark_rejected(
                        f"Kinematic jump ({disp:.1f}px > {allowed_disp:.1f}px in {dt*1000:.1f}ms; "
                        f"v={v_px_s:.0f}px/s > {self.max_vel_px_s:.0f}px/s)"
                    )
                    continue

                # 3. Apparent diameter ratio check
                d_cand = max(0.5, cand.apparent_diameter_px)
                d_last = max(0.5, last_accepted.apparent_diameter_px)
                ratio = max(d_cand, d_last) / min(d_cand, d_last)

                if ratio > self.max_diam_ratio:
                    cand.mark_rejected(
                        f"Sudden diameter change ratio ({ratio:.2f}x > max {self.max_diam_ratio:.2f}x; "
                        f"{d_last:.1f}px -> {d_cand:.1f}px)"
                    )
                    continue

            cand.mark_accepted()
            accepted.append(cand)

        return accepted
