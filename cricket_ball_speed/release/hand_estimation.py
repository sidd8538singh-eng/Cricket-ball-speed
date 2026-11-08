"""
Hand Estimation: Estimates the bowler's release hand / arm landmark region.
Combines upper-body biomechanical quadrant analysis, bowler bounding box geometry,
optional pose keypoints, and ball proximity snapping.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from detection.bowler_detector import BowlerObservation


class HandEstimator:
    """
    Estimates 2D coordinates of the bowler's release hand.
    In cricket bowling, release occurs at the apex of the arm circle,
    in the upper portion (top 15-25%) of the bowler's bounding box.
    """

    def __init__(self, arm_side: str = "auto"):
        """
        Args:
            arm_side: "right", "left", or "auto" (infers from bowler lateral position or arm arc).
        """
        self.arm_side = arm_side.lower().strip()

    def estimate_hand_position(
        self,
        bowler_obs: BowlerObservation,
        ball_candidates_nearby: Optional[List[Tuple[float, float]]] = None
    ) -> Tuple[float, float, float]:
        """
        Estimates hand (x, y) and estimation confidence.

        Priority order:
        1. Explicit pose keypoints (if available in bowler_obs.extra['keypoints'])
        2. Proximity snap to confirmed ball in upper body quadrant
        3. Biomechanical upper-quadrant arm extension heuristic

        Returns:
            (hand_x, hand_y, hand_confidence)
        """
        x1, y1, x2, y2 = bowler_obs.bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        # 1. Explicit pose keypoints if provided by a pose estimation backend
        if "keypoints" in bowler_obs.extra and isinstance(bowler_obs.extra["keypoints"], dict):
            kpts = bowler_obs.extra["keypoints"]
            # Check for right/left wrist
            wrist_key = "right_wrist" if self.arm_side == "right" else "left_wrist"
            if wrist_key in kpts:
                wx, wy, w_conf = kpts[wrist_key]
                if w_conf > 0.4:
                    return float(wx), float(wy), float(w_conf)
            elif "wrist" in kpts:
                wx, wy, w_conf = kpts["wrist"]
                if w_conf > 0.4:
                    return float(wx), float(wy), float(w_conf)

        # 2. Determine bowling arm side
        side = self.arm_side
        if side == "auto":
            # Right-arm bowler typically releases on right side of their body (screen right: x > center)
            # Default to right-arm bowler (predominant in cricket)
            side = "right"

        # Biomechanical arm apex position:
        # Release apex occurs near top 12-18% of bowler bounding box height
        nominal_y = y1 + 0.15 * bh
        if side == "right":
            # Right arm extends laterally outward (rightward in rear camera view)
            nominal_x = x1 + 0.68 * bw
        else:
            # Left arm extends leftward
            nominal_x = x1 + 0.32 * bw

        # 3. Ball candidate snapping:
        # If ball candidates are nearby in the upper quadrant (top 40% of bowler), snap toward ball
        if ball_candidates_nearby:
            best_snap = None
            min_dist = float("inf")
            for bx, by in ball_candidates_nearby:
                # Must be in upper half of bowler box
                if y1 <= by <= (y1 + 0.45 * bh) and (x1 - 0.1 * bw) <= bx <= (x2 + 0.1 * bw):
                    dist = float(np.hypot(bx - nominal_x, by - nominal_y))
                    if dist < min_dist:
                        min_dist = dist
                        best_snap = (bx, by)

            if best_snap is not None and min_dist < max(30.0, 0.06 * bh):
                # High confidence snap: ball is in or directly touching the bowler's delivery hand
                return float(best_snap[0]), float(best_snap[1]), 0.88

        # 4. Biomechanical heuristic fallback
        return float(nominal_x), float(nominal_y), 0.65
