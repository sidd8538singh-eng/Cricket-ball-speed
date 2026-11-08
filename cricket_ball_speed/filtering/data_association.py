"""
Data Association: Matches predicted tracks with detected ball candidates.
Employs Euclidean pre-filtering, adaptive Mahalanobis distance gating, and candidate ranking.
Records explicit rejection reasons on every non-associated candidate.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import logging

from ball_processing.ball_observations import BallObservation
from .kalman_2d import Kalman2D

logger = logging.getLogger("CricketBallSpeed.DataAssociation")


class DataAssociator:
    """
    Associates 2D detections with Kalman filter predictions using kinematic and statistical gating.
    Guarantees every candidate is assigned a clear acceptance or rejection reason.
    """

    def __init__(
        self,
        mahalanobis_gate_threshold: float = 3.5,
        max_pixel_distance: float = 90.0
    ):
        """
        Args:
            mahalanobis_gate_threshold: Statistical gate threshold (d_M <= gate).
                For 2 DoF chi-square, 3.03 corresponds to 99% confidence.
            max_pixel_distance: Maximum plausible displacement from predicted position (px).
        """
        self.gate_threshold = float(mahalanobis_gate_threshold)
        self.max_dist_px = float(max_pixel_distance)

    def associate(
        self,
        kalman_filter: Kalman2D,
        candidates: List[BallObservation]
    ) -> Tuple[Optional[BallObservation], List[BallObservation]]:
        """
        Finds the candidate that best matches the Kalman prediction.
        Sets rejection_reason on all non-associated candidates.

        Returns:
            matched_candidate: BallObservation or None
            unmatched_candidates: List[BallObservation] (with rejection reasons set)
        """
        if not candidates:
            return None, []

        pred_u, pred_v = kalman_filter.position
        unmatched: List[BallObservation] = []
        valid_gated_candidates: List[Tuple[float, BallObservation]] = []

        for cand in candidates:
            # 1. Euclidean pre-gating
            euclid_dist = float(np.hypot(cand.center[0] - pred_u, cand.center[1] - pred_v))
            if euclid_dist > self.max_dist_px:
                cand.mark_rejected(
                    f"Euclidean gate rejection: dist {euclid_dist:.1f}px > max {self.max_dist_px:.1f}px"
                )
                unmatched.append(cand)
                continue

            # 2. Mahalanobis distance gating
            d_m, y, S = kalman_filter.compute_mahalanobis(
                cand.center[0], cand.center[1], confidence=cand.confidence
            )

            if d_m > self.gate_threshold:
                cand.mark_rejected(
                    f"Mahalanobis gate rejection: d_M={d_m:.2f} > threshold={self.gate_threshold:.2f} (p < 0.01)"
                )
                unmatched.append(cand)
                continue

            # Candidate is within gate: score based on Mahalanobis distance discounted by confidence
            score = d_m / max(0.2, cand.confidence)
            valid_gated_candidates.append((score, cand))

        if not valid_gated_candidates:
            return None, unmatched

        # Sort valid candidates: lowest score first
        valid_gated_candidates.sort(key=lambda item: item[0])

        # Best candidate is selected
        best_cand = valid_gated_candidates[0][1]

        # All other in-gate candidates are marked as rejected
        for _, other_cand in valid_gated_candidates[1:]:
            other_cand.mark_rejected(
                f"Subsumed by higher-ranked candidate (score {valid_gated_candidates[0][0]:.2f} vs {other_cand.confidence:.2f})"
            )
            unmatched.append(other_cand)

        return best_cand, unmatched
