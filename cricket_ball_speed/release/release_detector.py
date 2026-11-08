"""
Release Detector: Pinpoints the exact bowling release frame (t1) and 2D ball position (C1).
Determines when the ball departs the bowler's hand and commences ballistic flight.

CRITICAL ARCHITECTURAL RULES:
1. The first detected ball frame is NEVER automatically assumed to be release.
2. Uses bowler detection, hand landmarks, ball detections, ball-to-hand distance,
   separation velocity, and forward temporal consistency.
3. Distinguishes whether release was DIRECTLY_OBSERVED or TEMPORALLY_ESTIMATED.
4. Quantifies uncertainty in release timing (+/- seconds and +/- frames).
5. Does NOT invent a release point when observations are insufficient.
6. Returns NOT_MEASURED with an explicit rejection reason if release cannot be reliably identified.
7. Does NOT use release height as a speed calculation prior.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np

from ball_processing.ball_observations import BallObservation
from detection.bowler_detector import BowlerObservation
from filtering.prediction_manager import TrackStatePoint
from .hand_estimation import HandEstimator
from .bowler_analysis import BowlerAnalyzer

logger = logging.getLogger("CricketBallSpeed.ReleaseDetector")


@dataclass
class ReleaseEvent:
    """
    Encapsulates the bowling release event with full physical and temporal auditability.
    """
    frame_index: int
    timestamp_sec: float
    c1_pixel: Tuple[float, float]               # (u1, v1) image coordinates of ball at release (C1)
    confidence: float
    hand_position_pixel: Tuple[float, float]
    observation_mode: str = "DIRECTLY_OBSERVED"  # "DIRECTLY_OBSERVED" or "TEMPORALLY_ESTIMATED"
    timing_uncertainty_sec: float = 0.00417      # Timing uncertainty (+/- seconds)
    timing_uncertainty_frames: float = 0.5       # Timing uncertainty (+/- frames)
    supporting_detections_count: int = 0
    separation_velocity_px_s: float = 0.0
    ball_to_hand_distance_px: float = 0.0
    is_valid: bool = True
    status: str = "MEASURED"                    # "MEASURED" or "NOT_MEASURED"
    rejection_reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 5),
            "c1_pixel": [round(c, 2) for c in self.c1_pixel],
            "confidence": round(self.confidence, 4),
            "hand_position_pixel": [round(c, 2) for c in self.hand_position_pixel],
            "observation_mode": self.observation_mode,
            "timing_uncertainty_sec": round(self.timing_uncertainty_sec, 5),
            "timing_uncertainty_frames": round(self.timing_uncertainty_frames, 2),
            "supporting_detections_count": self.supporting_detections_count,
            "separation_velocity_px_s": round(self.separation_velocity_px_s, 2),
            "ball_to_hand_distance_px": round(self.ball_to_hand_distance_px, 2),
            "is_valid": self.is_valid,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
        }


class ReleaseDetector:
    """
    Identifies the transition frame where the ball leaves the hand.
    Analyzes ball-to-hand displacement, separation velocity, and sustained forward trajectory.
    """

    def __init__(
        self,
        hand_estimator: Optional[HandEstimator] = None,
        min_supporting_frames: int = 4,
        min_separation_velocity_px_s: float = 120.0,
        max_attached_hand_distance_px: float = 45.0,
        min_release_confidence: float = 0.45
    ):
        """
        Args:
            hand_estimator: Estimator for bowler hand position.
            min_supporting_frames: Minimum post-release detections required to confirm trajectory.
            min_separation_velocity_px_s: Minimum velocity at which ball departs hand (px/s).
            max_attached_hand_distance_px: Maximum distance between ball and hand during attachment.
            min_release_confidence: Minimum composite confidence to accept release.
        """
        self.hand_estimator = hand_estimator or HandEstimator()
        self.min_supporting = int(min_supporting_frames)
        self.min_sep_vel = float(min_separation_velocity_px_s)
        self.max_attached_dist = float(max_attached_hand_distance_px)
        self.min_conf = float(min_release_confidence)

    def detect_release(
        self,
        tracked_points: List[TrackStatePoint],
        bowler_history: Dict[int, BowlerObservation]
    ) -> ReleaseEvent:
        """
        Scans tracked points to identify the release frame t1.

        Enforces:
        - The first ball detection is not automatically assumed to be release.
        - Checks for continuous forward flight away from the bowler.
        - Supports directly observed release or temporally estimated release.
        - Never invents a release point when observations are insufficient.
        """
        # 1. Filter points backed by real camera detections
        real_points = [p for p in tracked_points if p.is_real_detection]

        if len(real_points) < self.min_supporting:
            logger.warning(
                f"Insufficient real ball detections ({len(real_points)} < {self.min_supporting}) to detect release."
            )
            return ReleaseEvent(
                frame_index=0,
                timestamp_sec=0.0,
                c1_pixel=(0.0, 0.0),
                confidence=0.0,
                hand_position_pixel=(0.0, 0.0),
                observation_mode="NOT_OBSERVED",
                timing_uncertainty_sec=0.0,
                timing_uncertainty_frames=0.0,
                supporting_detections_count=len(real_points),
                separation_velocity_px_s=0.0,
                ball_to_hand_distance_px=0.0,
                is_valid=False,
                status="NOT_MEASURED",
                rejection_reason=(
                    f"INSUFFICIENT_OBSERVATIONS: Found only {len(real_points)} real detections "
                    f"(minimum required: {self.min_supporting})."
                )
            )

        # 2. Analyze bowler delivery stride window
        bowler_analyzer = BowlerAnalyzer()
        for b_obs in bowler_history.values():
            bowler_analyzer.add_observation(b_obs)
        win_start, win_end = bowler_analyzer.find_delivery_window()

        # 3. Direct Observation Search Pass
        best_direct_release: Optional[ReleaseEvent] = None
        best_direct_score = 0.0

        # Scan candidate frames for separation:
        # Note: we do NOT simply pick the first detected frame!
        for i in range(len(real_points) - 2):
            pt = real_points[i]
            frame_idx = pt.frame_index
            t1 = pt.timestamp_sec
            u1, v1 = pt.u, pt.v

            # Estimate bowler hand at this frame
            bowler_obs = bowler_history.get(frame_idx)
            if bowler_obs is not None:
                hx, hy, hand_conf = self.hand_estimator.estimate_hand_position(bowler_obs)
                hand_dist = float(np.hypot(u1 - hx, v1 - hy))
            else:
                # Bowler not visible in this specific frame; search adjacent frames within +/- 2
                nearby_bowlers = [
                    bowler_history[k] for k in range(frame_idx - 2, frame_idx + 3) if k in bowler_history
                ]
                if nearby_bowlers:
                    b_adj = nearby_bowlers[0]
                    hx, hy, hand_conf = self.hand_estimator.estimate_hand_position(b_adj)
                    hand_dist = float(np.hypot(u1 - hx, v1 - hy))
                else:
                    # Bowler is not detected anywhere near this candidate frame; direct observation cannot be verified
                    continue

            # Ball must be in hand proximity at release frame
            if hand_dist > self.max_attached_dist:
                # Ball is too far from hand; this frame is already in late flight, not release
                continue

            # Evaluate next frame: ball must begin departing the hand
            next_pt = real_points[i + 1]
            dt_step = next_pt.timestamp_sec - t1
            if dt_step <= 1e-4:
                continue

            # Compute hand position at next frame
            bowler_next = bowler_history.get(next_pt.frame_index)
            if bowler_next is not None:
                h_next_x, h_next_y, _ = self.hand_estimator.estimate_hand_position(bowler_next)
            else:
                h_next_x, h_next_y = hx, hy

            dist_next = float(np.hypot(next_pt.u - h_next_x, next_pt.v - h_next_y))
            separation_delta = dist_next - hand_dist
            separation_vel = separation_delta / dt_step

            # Ball must separate from hand with positive velocity
            if separation_vel < self.min_sep_vel or separation_delta < 2.0:
                # Ball is still attached / moving with hand (pre-release)
                continue

            # Check forward flight continuity over subsequent 2-3 frames
            subsequent = real_points[i + 1: min(len(real_points), i + 4)]
            dt_flight = subsequent[-1].timestamp_sec - t1
            if dt_flight <= 1e-4:
                continue

            flight_disp = float(np.hypot(subsequent[-1].u - u1, subsequent[-1].v - v1))
            flight_speed = flight_disp / dt_flight

            if flight_speed < self.min_sep_vel:
                continue

            # Compute composite confidence
            proximity_score = float(np.exp(-hand_dist / 35.0))
            separation_score = min(1.0, separation_vel / 400.0)
            flight_score = min(1.0, flight_speed / 600.0)
            window_score = 1.0 if (win_start <= frame_idx <= win_end) else 0.75

            conf = float(
                0.30 * hand_conf +
                0.30 * proximity_score +
                0.25 * separation_score +
                0.15 * window_score
            )

            if conf > best_direct_score:
                best_direct_score = conf
                dt_unc = max(0.002, dt_step / 2.0)
                best_direct_release = ReleaseEvent(
                    frame_index=frame_idx,
                    timestamp_sec=t1,
                    c1_pixel=(u1, v1),
                    confidence=conf,
                    hand_position_pixel=(hx, hy),
                    observation_mode="DIRECTLY_OBSERVED",
                    timing_uncertainty_sec=dt_unc,
                    timing_uncertainty_frames=0.5,
                    supporting_detections_count=len(real_points) - i,
                    separation_velocity_px_s=separation_vel,
                    ball_to_hand_distance_px=hand_dist,
                    is_valid=(conf >= self.min_conf),
                    status="MEASURED" if conf >= self.min_conf else "NOT_MEASURED",
                    rejection_reason=None if conf >= self.min_conf else f"Low release confidence ({conf:.2f} < {self.min_conf:.2f})"
                )

        if best_direct_release is not None and best_direct_release.is_valid:
            return best_direct_release

        # 4. Temporal Estimation Fallback (Occluded / Late First Detection)
        # If release occurred during occlusion and ball was first detected in early flight
        pt_first = real_points[0]
        first_frame = pt_first.frame_index
        first_t = pt_first.timestamp_sec

        # Search for bowler in the delivery window near the start of the ball track
        nearby_bowlers = [
            b for b in bowler_history.values()
            if abs(b.frame_index - first_frame) <= 6
        ]

        if nearby_bowlers and len(real_points) >= 3:
            # Find bowler closest to the first ball frame
            bowler_ref = min(nearby_bowlers, key=lambda b: abs(b.frame_index - first_frame))
            hx, hy, hand_conf = self.hand_estimator.estimate_hand_position(bowler_ref)

            dist_to_hand = float(np.hypot(pt_first.u - hx, pt_first.v - hy))

            # Only estimate if first detection is plausibly close to the bowler's release apex
            if dist_to_hand <= 120.0 and first_frame >= bowler_ref.frame_index:
                # Extrapolate velocity back to hand
                p2 = real_points[min(2, len(real_points) - 1)]
                dt_p = p2.timestamp_sec - first_t
                if dt_p > 1e-4:
                    vu = (p2.u - pt_first.u) / dt_p
                    vv = (p2.v - pt_first.v) / dt_p
                    speed_px_s = float(np.hypot(vu, vv))

                    if speed_px_s > self.min_sep_vel:
                        # Estimated release occurred slightly before first detection
                        dt_back = min(0.04, dist_to_hand / speed_px_s)
                        est_t1 = max(0.0, first_t - dt_back)
                        est_frame = max(0, int(round(first_frame - (dt_back / 0.00833))))

                        conf = float(0.55 * hand_conf + 0.25 * np.exp(-dist_to_hand / 70.0) + 0.20)
                        conf = min(0.75, conf) # Temporally estimated release capped at 0.75

                        return ReleaseEvent(
                            frame_index=est_frame,
                            timestamp_sec=est_t1,
                            c1_pixel=(hx, hy),
                            confidence=conf,
                            hand_position_pixel=(hx, hy),
                            observation_mode="TEMPORALLY_ESTIMATED",
                            timing_uncertainty_sec=max(0.008, dt_back),
                            timing_uncertainty_frames=max(1.0, dt_back / 0.00833),
                            supporting_detections_count=len(real_points),
                            separation_velocity_px_s=speed_px_s,
                            ball_to_hand_distance_px=dist_to_hand,
                            is_valid=(conf >= self.min_conf),
                            status="MEASURED" if conf >= self.min_conf else "NOT_MEASURED",
                            rejection_reason=None if conf >= self.min_conf else f"Temporally estimated confidence ({conf:.2f}) < threshold ({self.min_conf:.2f})"
                        )

        # 5. Observations Insufficient: DO NOT INVENT A RELEASE POINT
        logger.warning("Could not reliably identify bowling release point. Returning NOT_MEASURED.")
        return ReleaseEvent(
            frame_index=real_points[0].frame_index,
            timestamp_sec=real_points[0].timestamp_sec,
            c1_pixel=(real_points[0].u, real_points[0].v),
            confidence=0.0,
            hand_position_pixel=(0.0, 0.0),
            observation_mode="NOT_IDENTIFIED",
            timing_uncertainty_sec=0.0,
            timing_uncertainty_frames=0.0,
            supporting_detections_count=len(real_points),
            separation_velocity_px_s=0.0,
            ball_to_hand_distance_px=0.0,
            is_valid=False,
            status="NOT_MEASURED",
            rejection_reason=(
                "AMBIGUOUS_RELEASE: Could not isolate bowling release point. "
                "Ball flight cannot be linked to bowler hand separation."
            )
        )
