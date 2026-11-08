"""
Prediction Manager: Manages track lifecycle and distinguishes real detections from predictions.

CRITICAL ARCHITECTURAL MANDATES:
1. NEVER call a predicted position a real detection (is_real_detection = False for predictions).
2. NEVER allow a long prediction-only gap to create a fake continuous trajectory.
3. Use actual timestamps, including nonuniform frame intervals.
4. Preserve raw YOLO detections, accepted observations, rejected observations, and filtered states.
5. Record explicit rejection reasons on every non-accepted candidate.
6. Expose covariance and uncertainty for every state estimate.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import logging
import numpy as np

from ball_processing.ball_observations import BallObservation
from .kalman_2d import Kalman2D
from .data_association import DataAssociator

logger = logging.getLogger("CricketBallSpeed.PredictionManager")


@dataclass
class TrackStatePoint:
    """Holds point data along the ball track with full provenance and uncertainty."""
    frame_index: int
    timestamp_sec: float
    point_type: str                         # "OBSERVED", "FILTERED", "PREDICTED", "REJECTED"
    u: float
    v: float
    velocity_u: float
    velocity_v: float
    confidence: float
    observation: Optional[BallObservation] = None
    is_real_detection: bool = False         # STRICT: True ONLY when backed by a real sensor observation!
    position_uncertainty: Tuple[float, float] = (0.0, 0.0) # (sigma_u, sigma_v) in pixels
    velocity_uncertainty: Tuple[float, float] = (0.0, 0.0) # (sigma_Vu, sigma_Vv) in px/s
    covariance_diag: List[float] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 5),
            "point_type": self.point_type,
            "u": round(self.u, 2),
            "v": round(self.v, 2),
            "velocity_u": round(self.velocity_u, 2),
            "velocity_v": round(self.velocity_v, 2),
            "confidence": round(self.confidence, 4),
            "is_real_detection": self.is_real_detection,
            "sigma_u_px": round(self.position_uncertainty[0], 3),
            "sigma_v_px": round(self.position_uncertainty[1], 3),
            "sigma_Vu_px_s": round(self.velocity_uncertainty[0], 2),
            "sigma_Vv_px_s": round(self.velocity_uncertainty[1], 2),
            "rejection_reason": self.rejection_reason
        }


class PredictionManager:
    """
    Coordinates 2D Kalman tracking across video frames.
    Enforces maximum prediction gap thresholds and preserves complete audit trails.
    """

    def __init__(
        self,
        max_consecutive_predictions: int = 3,
        min_hits_to_confirm: int = 3,
        process_noise_std: float = 80.0,
        measurement_noise_std: float = 2.0,
        mahalanobis_gate_threshold: float = 3.5,
        max_pixel_distance: float = 90.0
    ):
        """
        Args:
            max_consecutive_predictions: Maximum allowed prediction-only frames before terminating track.
                Strictly prevents long prediction gaps from creating fake trajectories.
            min_hits_to_confirm: Minimum detections required to confirm a legitimate bowling track.
            process_noise_std: Acceleration spectral density in px/s^2.
            measurement_noise_std: Measurement noise in pixels.
            mahalanobis_gate_threshold: Statistical gate threshold.
            max_pixel_distance: Kinematic Euclidean distance ceiling.
        """
        self.max_pred_gap = max(1, int(max_consecutive_predictions))
        self.min_hits = max(1, int(min_hits_to_confirm))
        self.q_std = float(process_noise_std)
        self.r_std = float(measurement_noise_std)

        self.kalman: Optional[Kalman2D] = None
        self.associator = DataAssociator(
            mahalanobis_gate_threshold=mahalanobis_gate_threshold,
            max_pixel_distance=max_pixel_distance
        )

        self.consecutive_misses: int = 0
        self.total_hits: int = 0
        self.is_active: bool = False
        self.is_confirmed: bool = False
        self._last_timestamp: float = 0.0

        # Structured historical stores
        self.history: List[TrackStatePoint] = []
        self.raw_observations_history: Dict[int, List[BallObservation]] = {}
        self.accepted_observations_history: Dict[int, BallObservation] = {}
        self.rejected_observations_history: Dict[int, List[BallObservation]] = {}

    def initialize_track(self, initial_obs: BallObservation):
        """Initializes a new track from a valid starting candidate."""
        self.kalman = Kalman2D(
            initial_u=initial_obs.center[0],
            initial_v=initial_obs.center[1],
            process_noise_std=self.q_std,
            measurement_noise_std=self.r_std
        )
        self.consecutive_misses = 0
        self.total_hits = 1
        self.is_active = True
        self.is_confirmed = (self.total_hits >= self.min_hits)
        self._last_timestamp = float(initial_obs.timestamp_sec)

        initial_obs.mark_accepted()
        self.accepted_observations_history[initial_obs.frame_index] = initial_obs

        sig_pos = self.kalman.get_position_uncertainty()
        sig_vel = self.kalman.get_velocity_uncertainty()

        pt = TrackStatePoint(
            frame_index=initial_obs.frame_index,
            timestamp_sec=initial_obs.timestamp_sec,
            point_type="OBSERVED",
            u=initial_obs.center[0],
            v=initial_obs.center[1],
            velocity_u=0.0,
            velocity_v=0.0,
            confidence=initial_obs.confidence,
            observation=initial_obs,
            is_real_detection=True, # Validated real detection
            position_uncertainty=sig_pos,
            velocity_uncertainty=sig_vel,
            covariance_diag=[float(self.kalman.P[i, i]) for i in range(4)]
        )
        self.history.append(pt)

    def step(
        self,
        frame_idx: int,
        timestamp_sec: float,
        candidates: List[BallObservation]
    ) -> Optional[TrackStatePoint]:
        """
        Advances track state by predicting and updating with observations.
        Uses actual timestamps, handling nonuniform frame intervals seamlessly.

        Enforces:
        - Never call a predicted position a real detection.
        - Never allow a long prediction-only gap to create a fake continuous trajectory.
        """
        # Store raw observations for full auditability
        self.raw_observations_history[frame_idx] = list(candidates)

        if not self.is_active or self.kalman is None:
            if candidates:
                # Select best candidate by confidence and initialize
                best_cand = max(candidates, key=lambda c: c.confidence)
                self.initialize_track(best_cand)
                # Any other candidates in this initialization frame are marked rejected
                other_cands = [c for c in candidates if c is not best_cand]
                for oc in other_cands:
                    oc.mark_rejected("Subsumed during track initialization")
                self.rejected_observations_history[frame_idx] = other_cands
                return self.history[-1]
            return None

        # Compute actual dt from timestamps (handles nonuniform frame intervals)
        dt = timestamp_sec - self._last_timestamp
        if dt <= 0.0:
            # Fallback for identical/monotonic timestamp anomaly
            dt = 1.0 / 120.0
        self._last_timestamp = timestamp_sec

        # 1. Kalman Predict
        pred_state = self.kalman.predict(dt)
        pred_u, pred_v = float(pred_state[0]), float(pred_state[1])

        # 2. Data Association
        matched_obs, rejected_cands = self.associator.associate(self.kalman, candidates)
        self.rejected_observations_history[frame_idx] = rejected_cands

        if matched_obs is not None:
            # 3. Measurement Update
            upd_state, d_m = self.kalman.update(
                matched_obs.center[0], matched_obs.center[1], confidence=matched_obs.confidence
            )
            matched_obs.mark_accepted()
            self.accepted_observations_history[frame_idx] = matched_obs

            self.consecutive_misses = 0
            self.total_hits += 1
            if self.total_hits >= self.min_hits:
                self.is_confirmed = True

            sig_pos = self.kalman.get_position_uncertainty()
            sig_vel = self.kalman.get_velocity_uncertainty()

            pt = TrackStatePoint(
                frame_index=frame_idx,
                timestamp_sec=timestamp_sec,
                point_type="FILTERED",
                u=float(upd_state[0]),
                v=float(upd_state[1]),
                velocity_u=float(upd_state[2]),
                velocity_v=float(upd_state[3]),
                confidence=matched_obs.confidence,
                observation=matched_obs,
                is_real_detection=True, # STRICT: Backed by real sensor detection
                position_uncertainty=sig_pos,
                velocity_uncertainty=sig_vel,
                covariance_diag=[float(self.kalman.P[i, i]) for i in range(4)],
                extra={"mahalanobis": d_m, "dt_sec": dt}
            )
            self.history.append(pt)
            return pt

        else:
            # Missed observation: enter prediction-only state
            self.consecutive_misses += 1

            # REQUIREMENT: Never allow a long prediction-only gap to create a fake continuous trajectory
            if self.consecutive_misses > self.max_pred_gap:
                logger.warning(
                    f"Frame {frame_idx}: Track terminated. Exceeded max consecutive predictions "
                    f"({self.consecutive_misses} > {self.max_pred_gap}). Preventing fake trajectory interpolation."
                )
                self.is_active = False
                return None

            sig_pos = self.kalman.get_position_uncertainty()
            sig_vel = self.kalman.get_velocity_uncertainty()

            pt = TrackStatePoint(
                frame_index=frame_idx,
                timestamp_sec=timestamp_sec,
                point_type="PREDICTED",
                u=pred_u,
                v=pred_v,
                velocity_u=float(pred_state[2]),
                velocity_v=float(pred_state[3]),
                confidence=max(0.05, 0.50 - 0.12 * self.consecutive_misses),
                observation=None,
                is_real_detection=False, # STRICT REQUIREMENT: NEVER MARK PREDICTED POINT AS REAL DETECTION!
                position_uncertainty=sig_pos,
                velocity_uncertainty=sig_vel,
                covariance_diag=[float(self.kalman.P[i, i]) for i in range(4)],
                extra={"consecutive_misses": self.consecutive_misses, "dt_sec": dt}
            )
            self.history.append(pt)
            return pt

    def get_real_detected_points(self) -> List[TrackStatePoint]:
        """Returns exclusively points backed by verified camera detections."""
        return [pt for pt in self.history if pt.is_real_detection]

    def get_predicted_points(self) -> List[TrackStatePoint]:
        """Returns exclusively prediction-only points."""
        return [pt for pt in self.history if not pt.is_real_detection]

    def get_all_points(self) -> List[TrackStatePoint]:
        """Returns full historical trajectory (both real detections and valid bounded predictions)."""
        return list(self.history)

    def get_all_rejected_observations(self) -> List[BallObservation]:
        """Returns flattened list of all rejected observations across all frames."""
        rejected = []
        for frame_idx in sorted(self.rejected_observations_history.keys()):
            rejected.extend(self.rejected_observations_history[frame_idx])
        return rejected

    def get_audit_summary(self) -> Dict[str, Any]:
        """Returns comprehensive track lifecycle audit report."""
        real_pts = self.get_real_detected_points()
        pred_pts = self.get_predicted_points()
        total_rejected = sum(len(v) for v in self.rejected_observations_history.values())
        return {
            "total_track_points": len(self.history),
            "real_detected_count": len(real_pts),
            "predicted_count": len(pred_pts),
            "total_rejected_candidates": total_rejected,
            "is_confirmed": self.is_confirmed,
            "is_active": self.is_active,
            "consecutive_misses": self.consecutive_misses,
            "max_allowed_prediction_gap": self.max_pred_gap
        }
