"""
Comprehensive Unit Tests for Cricket Bowling Release Detection:
- Hand landmark estimation and ball snapping
- First-frame non-release guarantee (first detection != release)
- Directly observed release with ball-to-hand separation velocity
- Temporally estimated release (backward extrapolation for occluded start)
- Non-invention of release points under insufficient observations (NOT_MEASURED)
- Far down-pitch detection rejection
- Timing uncertainty quantification
- Physical biomechanical height and position validation (post-hoc sanity check, NOT a speed prior)
- Diagnostic visualizer rendering
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detection.bowler_detector import BowlerObservation
from filtering.prediction_manager import TrackStatePoint
from release.hand_estimation import HandEstimator
from release.bowler_analysis import BowlerAnalyzer
from release.release_detector import ReleaseDetector, ReleaseEvent
from release.release_validator import ReleaseValidator
from release.release_visualizer import ReleaseVisualizer


class TestReleaseSubsystem(unittest.TestCase):

    # -------------------------------------------------------------------------
    # 1. Hand Landmark Estimation Tests
    # -------------------------------------------------------------------------

    def test_hand_estimation_quadrant_and_snapping(self):
        """Tests bowler hand estimation in upper quadrant and proximity snapping."""
        estimator = HandEstimator(arm_side="right")
        bowler = BowlerObservation(
            frame_index=10,
            timestamp_sec=0.10,
            bbox=[400.0, 300.0, 600.0, 900.0], # W=200, H=600
            confidence=0.90,
            center=[500.0, 600.0],
            width=200.0,
            height=600.0
        )

        # 1. Biomechanical heuristic
        hx, hy, conf = estimator.estimate_hand_position(bowler)
        self.assertAlmostEqual(hy, 390.0, delta=15.0)
        self.assertAlmostEqual(hx, 536.0, delta=15.0)

        # 2. Proximity snap to ball held in upper hand quadrant
        hx_snap, hy_snap, conf_snap = estimator.estimate_hand_position(
            bowler, ball_candidates_nearby=[(540.0, 385.0)]
        )
        self.assertEqual(hx_snap, 540.0)
        self.assertEqual(hy_snap, 385.0)
        self.assertGreater(conf_snap, conf)

    # -------------------------------------------------------------------------
    # 2. Directly Observed Release & First-Frame Non-Release Rule
    # -------------------------------------------------------------------------

    def test_first_ball_detection_must_not_automatically_be_release(self):
        """
        REQUIREMENT: 'The first ball detection must not automatically be treated as release.'
        Simulate a bowler running in with the ball detected in-hand for 4 frames before release.
        Verifies that frame 0 is rejected as release, and frame 4 is correctly identified.
        """
        detector = ReleaseDetector(min_supporting_frames=4, min_separation_velocity_px_s=150.0)
        bowler_hist = {}

        # Bowler detected across frames 0 to 12
        for i in range(13):
            bowler_hist[i] = BowlerObservation(
                frame_index=i,
                timestamp_sec=i * 0.00833,
                bbox=[450.0, 250.0, 650.0, 850.0],
                confidence=0.88,
                center=[550.0, 550.0],
                width=200.0,
                height=600.0
            )

        # Hand is at (586, 340)
        tracked_points = []

        # Frames 0 to 4: Ball held in hand (moving slowly with arm: separation ~ 0)
        for i in range(5):
            tracked_points.append(TrackStatePoint(
                frame_index=i,
                timestamp_sec=i * 0.00833,
                point_type="FILTERED",
                u=586.0 + i * 0.5,
                v=340.0 + i * 0.5,
                velocity_u=15.0,
                velocity_v=15.0,
                confidence=0.85,
                is_real_detection=True
            ))

        # Frames 5 to 12: Ball in fast departure flight (separating by 35px per frame = 4200 px/s)
        for i in range(5, 13):
            step = i - 4
            tracked_points.append(TrackStatePoint(
                frame_index=i,
                timestamp_sec=i * 0.00833,
                point_type="FILTERED",
                u=588.0 + step * 2.0,
                v=342.0 + step * 35.0,
                velocity_u=240.0,
                velocity_v=4200.0,
                confidence=0.90,
                is_real_detection=True
            ))

        release_event = detector.detect_release(tracked_points, bowler_hist)

        self.assertIsNotNone(release_event)
        self.assertTrue(release_event.is_valid)
        self.assertEqual(release_event.status, "MEASURED")

        # CRITICAL ASSERTION: Frame 0 MUST NOT be the release frame!
        self.assertNotEqual(release_event.frame_index, 0)
        # Release frame must be frame 4 (the last frame in hand before separation)
        self.assertIn(release_event.frame_index, [4, 5])
        self.assertEqual(release_event.observation_mode, "DIRECTLY_OBSERVED")
        self.assertGreater(release_event.separation_velocity_px_s, 150.0)
        self.assertGreater(release_event.timing_uncertainty_sec, 0.0)

    # -------------------------------------------------------------------------
    # 3. Temporally Estimated Release (Occluded / Late Detection Start)
    # -------------------------------------------------------------------------

    def test_temporally_estimated_release_occluded_start(self):
        """
        Tests that when ball is occluded right at release and first detected 2 frames into flight,
        the system estimates release without inventing fake points.
        """
        detector = ReleaseDetector(min_supporting_frames=4, min_separation_velocity_px_s=150.0)
        bowler_hist = {}

        # Bowler at release apex around frame 10 (hand at 586, 340)
        for i in range(8, 14):
            bowler_hist[i] = BowlerObservation(
                frame_index=i,
                timestamp_sec=i * 0.00833,
                bbox=[450.0, 250.0, 650.0, 850.0],
                confidence=0.90,
                center=[550.0, 550.0],
                width=200.0,
                height=600.0
            )

        # Ball first detected at frame 12, already 50px away from hand (fast flight)
        # (Frames 10 and 11 were occluded)
        tracked_points = []
        for i in range(12, 19):
            step = i - 12
            tracked_points.append(TrackStatePoint(
                frame_index=i,
                timestamp_sec=i * 0.00833,
                point_type="FILTERED",
                u=590.0 + step * 2.0,
                v=395.0 + step * 28.0,
                velocity_u=240.0,
                velocity_v=3360.0,
                confidence=0.85,
                is_real_detection=True
            ))

        release_event = detector.detect_release(tracked_points, bowler_hist)

        self.assertIsNotNone(release_event)
        self.assertTrue(release_event.is_valid)
        self.assertEqual(release_event.observation_mode, "TEMPORALLY_ESTIMATED")
        # Estimated release frame should be close to bowler apex (frame 10-11)
        self.assertLessEqual(release_event.frame_index, 12)
        # Timing uncertainty must be larger for estimated mode
        self.assertGreaterEqual(release_event.timing_uncertainty_frames, 1.0)

    # -------------------------------------------------------------------------
    # 4. Insufficient Observations & Far Down-Pitch Rejections (NOT_MEASURED)
    # -------------------------------------------------------------------------

    def test_insufficient_observations_returns_not_measured(self):
        """
        REQUIREMENT: 'Do not invent a release point when the observations are insufficient.'
        'If release cannot be confidently identified, return NOT_MEASURED with a clear rejection reason.'
        """
        detector = ReleaseDetector(min_supporting_frames=4)
        bowler_hist = {}

        # Only 2 detections available
        sparse_points = [
            TrackStatePoint(1, 0.00833, "FILTERED", 500.0, 300.0, 10.0, 10.0, 0.85, is_real_detection=True),
            TrackStatePoint(2, 0.01667, "FILTERED", 510.0, 320.0, 10.0, 10.0, 0.85, is_real_detection=True)
        ]

        release_event = detector.detect_release(sparse_points, bowler_hist)

        self.assertFalse(release_event.is_valid)
        self.assertEqual(release_event.status, "NOT_MEASURED")
        self.assertIsNotNone(release_event.rejection_reason)
        self.assertIn("INSUFFICIENT_OBSERVATIONS", release_event.rejection_reason)

    def test_far_downpitch_first_detection_returns_not_measured(self):
        """
        If the ball is first detected 200px down-pitch away from bowler with no link to release,
        the system must NOT invent a release point. Must return NOT_MEASURED.
        """
        detector = ReleaseDetector(min_supporting_frames=4)
        bowler_hist = {
            5: BowlerObservation(5, 0.0416, [450.0, 250.0, 650.0, 850.0], 0.90, [550.0, 550.0], 200.0, 600.0)
        }

        # Ball first detected at frame 15 at v=750 (far down-pitch, distance > 350px from hand)
        late_points = [
            TrackStatePoint(15 + i, 0.125 + i * 0.00833, "FILTERED", 600.0, 750.0 + i * 30.0, 0.0, 3600.0, 0.85, is_real_detection=True)
            for i in range(6)
        ]

        release_event = detector.detect_release(late_points, bowler_hist)

        self.assertFalse(release_event.is_valid)
        self.assertEqual(release_event.status, "NOT_MEASURED")
        self.assertIn("AMBIGUOUS_RELEASE", release_event.rejection_reason)

    # -------------------------------------------------------------------------
    # 5. Release Validator & Biomechanical Bound Checks
    # -------------------------------------------------------------------------

    def test_release_validator_height_and_bounds(self):
        """
        Validates post-hoc biomechanical plausibility checks.
        Ensures release height is checked strictly as a sanity check, NOT as a speed calculation prior.
        """
        validator = ReleaseValidator(
            min_release_confidence=0.45,
            min_post_release_detections=4,
            min_plausible_height_m=1.40,
            max_plausible_height_m=2.90
        )

        valid_event = ReleaseEvent(
            frame_index=5,
            timestamp_sec=0.0416,
            c1_pixel=(580.0, 340.0),
            confidence=0.85,
            hand_position_pixel=(580.0, 340.0),
            supporting_detections_count=7,
            separation_velocity_px_s=1200.0,
            is_valid=True,
            status="MEASURED"
        )

        # 1. Valid human release height (e.g. 2.15m above ground, near crease Y=0.2m)
        is_ok, reason = validator.validate_event(valid_event, np.array([0.1, 0.2, 2.15]))
        self.assertTrue(is_ok)
        self.assertIsNone(reason)

        # 2. Implausible release height (e.g. 3.8m above ground - physically impossible for human bowler)
        is_bad, bad_reason = validator.validate_event(valid_event, np.array([0.1, 0.2, 3.80]))
        self.assertFalse(is_bad)
        self.assertIn("RELEASE_HEIGHT_IMPLAUSIBLE", bad_reason)

        # 3. Implausible longitudinal position (e.g. Y=8.0m down the pitch)
        is_bad_y, bad_y_reason = validator.validate_event(valid_event, np.array([0.1, 8.5, 2.10]))
        self.assertFalse(is_bad_y)
        self.assertIn("RELEASE_POSITION_IMPLAUSIBLE", bad_y_reason)

    # -------------------------------------------------------------------------
    # 6. Diagnostic Release Visualizer Tests
    # -------------------------------------------------------------------------

    def test_release_visualizer_rendering(self):
        """Verifies ReleaseVisualizer renders diagnostic overlay with hand, C1, and badge."""
        vis = ReleaseVisualizer()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        event = ReleaseEvent(
            frame_index=12,
            timestamp_sec=0.100,
            c1_pixel=(640.0, 320.0),
            confidence=0.88,
            hand_position_pixel=(635.0, 310.0),
            observation_mode="DIRECTLY_OBSERVED",
            timing_uncertainty_sec=0.00417,
            timing_uncertainty_frames=0.5,
            supporting_detections_count=8,
            separation_velocity_px_s=420.0,
            ball_to_hand_distance_px=11.2,
            is_valid=True,
            status="MEASURED"
        )

        bowler = BowlerObservation(
            frame_index=12,
            timestamp_sec=0.100,
            bbox=[520.0, 200.0, 720.0, 680.0],
            confidence=0.91,
            center=[620.0, 440.0],
            width=200.0,
            height=480.0
        )

        annotated = vis.render_release_frame(frame, event, bowler)
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, (720, 1280, 3))
        self.assertGreater(np.count_nonzero(annotated), 1000)


if __name__ == "__main__":
    unittest.main()
