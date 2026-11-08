"""
Unit tests for AI detection subsystem:
- Class mapping extraction and resolution
- Model path and device validation
- Overlapping tiled inference and image upscaling
- Sub-pixel coordinate restoration to original video frame coordinates
- Overlapping tile duplicate removal (NMS) and source merging
- Low-confidence ball candidate preservation
- Non-conflation of bowler bounding box with hand landmarks
- Immutability guarantee that stumps detection never alters manual calibration
- Detection visualizer rendering
"""

import unittest
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detection.tiled_inference import (
    compute_tiles,
    restore_tile_coordinates,
    apply_nms,
    merge_candidate_observations,
    TiledInferenceEngine
)
from detection.class_mapper import ModelClassResolver
from detection.model_validator import ModelValidator
from detection.ball_detector import BallDetector
from detection.bowler_detector import BowlerDetector, BowlerObservation
from detection.stumps_detector import StumpsDetector, StumpsObservation, CalibrationSecurityError
from detection.detection_manager import DetectionManager
from detection.detection_visualizer import DetectionVisualizer

from ball_processing.ball_observations import BallObservation
from ball_processing.bbox_refinement import BoundingBoxRefiner
from ball_processing.apparent_diameter import ApparentDiameterEstimator
from ball_processing.observation_quality import ObservationQualityAssessor
from ball_processing.outlier_rejection import OutlierRejector


class TestDetection(unittest.TestCase):

    # -------------------------------------------------------------------------
    # 1. Existing Core Tests (Preserved)
    # -------------------------------------------------------------------------

    def test_compute_tiles_coverage(self):
        """Verifies tiled slicing covers entire frame with overlap."""
        w, h = 1920, 1080
        tiles = compute_tiles(w, h, tile_w=640, tile_h=640, overlap_x=0.20, overlap_y=0.20)
        self.assertGreater(len(tiles), 4)

        # Ensure last tile reaches bottom-right corner
        max_x2 = max(t[2] for t in tiles)
        max_y2 = max(t[3] for t in tiles)
        self.assertEqual(max_x2, w)
        self.assertEqual(max_y2, h)

    def test_apply_nms(self):
        """Tests Non-Maximum Suppression on duplicate detections."""
        boxes = np.array([
            [100.0, 100.0, 120.0, 120.0],
            [101.0, 99.0, 121.0, 119.0],  # Overlapping duplicate
            [300.0, 300.0, 320.0, 320.0]   # Distinct candidate
        ], dtype=np.float32)
        scores = np.array([0.90, 0.75, 0.85], dtype=np.float32)

        keep = apply_nms(boxes, scores, iou_threshold=0.40)
        self.assertEqual(len(keep), 2)
        self.assertIn(0, keep)
        self.assertIn(2, keep)

    def test_bbox_refinement_synthetic(self):
        """Tests sub-pixel centroid refinement on synthetic ball circle."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 6, (255, 255, 255), -1)

        raw_obs = BallObservation.from_bbox(
            frame_idx=1,
            timestamp_sec=0.016,
            bbox=[42.0, 42.0, 60.0, 60.0],
            confidence=0.85
        )

        refiner = BoundingBoxRefiner(padding_px=6)
        refined = refiner.refine(img, raw_obs)

        self.assertAlmostEqual(refined.center[0], 50.0, delta=0.5)
        self.assertAlmostEqual(refined.center[1], 50.0, delta=0.5)

    def test_apparent_diameter_depth(self):
        """Tests depth estimation from apparent diameter formula."""
        estimator = ApparentDiameterEstimator(physical_diameter_m=0.0715)
        focal_length = 1800.0
        target_depth = 10.0
        apparent_diam_px = (focal_length * 0.0715) / target_depth

        est_depth, unc = estimator.estimate_depth_from_diameter(apparent_diam_px, focal_length)
        self.assertAlmostEqual(est_depth, target_depth, places=3)
        self.assertGreater(unc, 0.0)

    def test_observation_quality_assessor(self):
        """Tests quality scoring penalizing non-circular artifacts."""
        assessor = ObservationQualityAssessor()

        good_obs = BallObservation.from_bbox(1, 0.0, [100.0, 100.0, 120.0, 120.0], 0.85)
        bad_obs = BallObservation.from_bbox(1, 0.0, [100.0, 100.0, 250.0, 110.0], 0.85)

        score_good = assessor.assess(good_obs, 1920, 1080)
        score_bad = assessor.assess(bad_obs, 1920, 1080)

        self.assertGreater(score_good, score_bad)
        self.assertGreater(score_good, 0.6)

    def test_outlier_rejector(self):
        """Tests filtering of impossible physical jumps."""
        rejector = OutlierRejector(max_interframe_displacement_px=100.0)

        obs1 = BallObservation.from_bbox(1, 0.0, [500.0, 500.0, 515.0, 515.0], 0.80)
        obs1.quality_score = 0.8
        obs1.mark_accepted()

        obs_valid = BallObservation.from_bbox(2, 0.00833, [510.0, 512.0, 525.0, 527.0], 0.80)
        obs_valid.quality_score = 0.8

        obs_jump = BallObservation.from_bbox(2, 0.00833, [1200.0, 800.0, 1215.0, 815.0], 0.80)
        obs_jump.quality_score = 0.8

        accepted = rejector.filter_candidates([obs_valid, obs_jump], last_accepted=obs1)
        self.assertIn(obs_valid, accepted)
        self.assertNotIn(obs_jump, accepted)
        self.assertEqual(obs_jump.acceptance_status, "REJECTED")

    # -------------------------------------------------------------------------
    # 2. Class Mapping Resolution Tests
    # -------------------------------------------------------------------------

    def test_class_mapping_resolution(self):
        """Verifies dynamic class mapping resolution from model metadata."""
        # Case A: Standard COCO model where class 32 is 'sports ball' and class 0 is 'person'
        coco_names = {0: "person", 1: "bicycle", 32: "sports ball", 37: "baseball"}
        res_ball = ModelClassResolver.resolve_classes(
            model={"names": coco_names},
            target_keywords=["ball", "sports ball", "cricket_ball"],
            entity_name="ball"
        )
        self.assertIn(32, res_ball.target_class_ids)
        self.assertFalse(res_ball.is_fallback)

        res_bowler = ModelClassResolver.resolve_classes(
            model={"names": coco_names},
            target_keywords=["bowler", "person"],
            entity_name="bowler"
        )
        self.assertIn(0, res_bowler.target_class_ids)

        # Case B: Custom cricket model
        custom_names = {0: "cricket_stumps", 1: "cricket_ball", 2: "batsman", 3: "bowler"}
        res_stumps = ModelClassResolver.resolve_classes(
            model={"names": custom_names},
            target_keywords=["stumps", "wicket"],
            entity_name="stumps"
        )
        self.assertEqual(res_stumps.target_class_ids, [0])

        res_cricket_ball = ModelClassResolver.resolve_classes(
            model={"names": custom_names},
            target_keywords=["ball", "cricket_ball"],
            entity_name="ball"
        )
        self.assertEqual(res_cricket_ball.target_class_ids, [1])

        res_custom_bowler = ModelClassResolver.resolve_classes(
            model={"names": custom_names},
            target_keywords=["bowler"],
            entity_name="bowler"
        )
        self.assertEqual(res_custom_bowler.target_class_ids, [3])

    # -------------------------------------------------------------------------
    # 3. Model Validator Tests
    # -------------------------------------------------------------------------

    def test_model_validator_reports_missing_mandatory(self):
        """Verifies validator correctly detects missing mandatory model paths."""
        fake_paths = {
            "ball": "nonexistent_dir/nonexistent_ball.pt",
            "bowler": "nonexistent_dir/nonexistent_bowler.pt",
            "stumps": "nonexistent_dir/nonexistent_stumps.pt",
            "pitch": "nonexistent_dir/nonexistent_pitch.pt"
        }
        report = ModelValidator.validate_models(fake_paths, requested_device="cpu")
        self.assertFalse(report.is_valid)
        self.assertEqual(len(report.missing_mandatory), 3)
        self.assertIn("ball", report.missing_mandatory)
        self.assertIn("bowler", report.missing_mandatory)
        self.assertIn("stumps", report.missing_mandatory)
        # Pitch is optional, so it must not be in missing_mandatory
        self.assertNotIn("pitch", report.missing_mandatory)

    def test_model_device_resolution(self):
        """Tests device resolution for cpu and auto."""
        dev, cuda_avail, _ = ModelValidator.resolve_device("cpu")
        self.assertEqual(dev, "cpu")

        dev_auto, _, _ = ModelValidator.resolve_device("auto")
        self.assertIn(dev_auto, ["cpu", "cuda:0"])

    # -------------------------------------------------------------------------
    # 4. Coordinate Restoration & Upscaling Tests
    # -------------------------------------------------------------------------

    def test_coordinate_restoration_with_upscaling(self):
        """
        Tests exact sub-pixel restoration of detected bounding box from upscaled tile
        back to original video coordinates.
        """
        tile_x1, tile_y1 = 640.0, 360.0
        upscale_factor = 2.0

        # Suppose a tiny ball is located at (740.0, 460.0) in the original frame,
        # with width=10.0, height=10.0 (bbox [735.0, 455.0, 745.0, 465.0]).
        # In the unscaled tile, bbox is [95.0, 95.0, 105.0, 105.0].
        # In the 2x upscaled crop, bbox is [190.0, 190.0, 210.0, 210.0].
        upscaled_crop_bbox = [190.0, 190.0, 210.0, 210.0]

        restored_bbox = restore_tile_coordinates(
            crop_bbox=upscaled_crop_bbox,
            tile_x1=tile_x1,
            tile_y1=tile_y1,
            upscale_factor=upscale_factor
        )

        expected_bbox = [735.0, 455.0, 745.0, 465.0]
        for r, e in zip(restored_bbox, expected_bbox):
            self.assertAlmostEqual(r, e, places=5)

        # Check center and apparent diameter
        cx = (restored_bbox[0] + restored_bbox[2]) / 2.0
        cy = (restored_bbox[1] + restored_bbox[3]) / 2.0
        self.assertAlmostEqual(cx, 740.0, places=5)
        self.assertAlmostEqual(cy, 460.0, places=5)

    def test_tiny_object_coordinate_restoration_4k(self):
        """
        Tests tiny 6-pixel ball coordinate restoration in 4K resolution (3840x2160).
        """
        engine = TiledInferenceEngine(
            tile_width=640,
            tile_height=640,
            overlap_x=0.25,
            overlap_y=0.25,
            upscale_factor=1.5
        )

        # Frame 3840x2160
        tiles = engine.get_tiles(3840, 2160)
        self.assertGreater(len(tiles), 15)

        # Pick tile starting at (1440, 960)
        tile = (1440, 960, 2080, 1600)
        # Object inside tile at original position: (1500.2, 1000.4)
        orig_x1 = 1497.2
        orig_y1 = 997.4
        orig_x2 = 1503.2
        orig_y2 = 1003.4

        # Crop coordinate (unscaled)
        c_x1 = orig_x1 - tile[0]
        c_y1 = orig_y1 - tile[1]
        c_x2 = orig_x2 - tile[0]
        c_y2 = orig_y2 - tile[1]

        # In 1.5x upscaled crop:
        u_crop = [c_x1 * 1.5, c_y1 * 1.5, c_x2 * 1.5, c_y2 * 1.5]

        restored = restore_tile_coordinates(u_crop, tile[0], tile[1], upscale_factor=1.5)
        self.assertAlmostEqual(restored[0], orig_x1, places=4)
        self.assertAlmostEqual(restored[1], orig_y1, places=4)
        self.assertAlmostEqual(restored[2], orig_x2, places=4)
        self.assertAlmostEqual(restored[3], orig_y2, places=4)

    # -------------------------------------------------------------------------
    # 5. Overlapping Tile Duplicate Removal & Source Merging Tests
    # -------------------------------------------------------------------------

    def test_overlapping_tile_duplicate_removal_with_source_merging(self):
        """
        Tests that when a ball is detected across multiple overlapping tiles and full-frame,
        duplicates are merged via NMS while recording all corroborating sources.
        """
        # Ball located at (550, 400)
        obs_full = BallObservation.from_bbox(
            frame_idx=10,
            timestamp_sec=0.083,
            bbox=[545.0, 395.0, 555.0, 405.0],
            confidence=0.70,
            detection_source="full_frame"
        )
        obs_tile1 = BallObservation.from_bbox(
            frame_idx=10,
            timestamp_sec=0.083,
            bbox=[544.5, 394.8, 555.2, 405.1],
            confidence=0.88,
            detection_source="tile_x0_y0"
        )
        obs_tile2 = BallObservation.from_bbox(
            frame_idx=10,
            timestamp_sec=0.083,
            bbox=[545.2, 395.1, 554.8, 404.9],
            confidence=0.82,
            detection_source="tile_x512_y0"
        )

        candidates = [obs_full, obs_tile1, obs_tile2]
        merged = merge_candidate_observations(candidates, iou_threshold=0.35)

        # All 3 represent the exact same ball; should be merged into 1
        self.assertEqual(len(merged), 1)

        primary = merged[0]
        # Highest confidence (0.88 from tile1) should be chosen as primary
        self.assertEqual(primary.detection_source, "tile_x0_y0")
        self.assertAlmostEqual(primary.confidence, 0.88, places=2)

        # Audit trail must record all 3 contributing sources
        merged_srcs = primary.extra.get("merged_sources", [])
        self.assertEqual(len(merged_srcs), 3)
        self.assertIn("full_frame", merged_srcs)
        self.assertIn("tile_x0_y0", merged_srcs)
        self.assertIn("tile_x512_y0", merged_srcs)

    # -------------------------------------------------------------------------
    # 6. Low-Confidence Ball Preservation Tests
    # -------------------------------------------------------------------------

    def test_low_confidence_ball_preservation(self):
        """
        Requirement: 'Do not discard a ball merely because its confidence is lower than an arbitrary threshold.'
        Verifies BallDetector retains low-confidence flight candidates (e.g., conf 0.14) for temporal verification.
        """
        # Create a mock predictor returning one low-conf candidate (0.14) and one high-conf candidate (0.85)
        def mock_predict(img):
            return [
                ([50.0, 50.0, 62.0, 62.0], 0.14, 0),  # Low conf motion blur candidate
                ([200.0, 200.0, 215.0, 215.0], 0.85, 0)
            ]

        detector = BallDetector(
            confidence_threshold=0.20,
            min_candidate_confidence=0.10, # Configured low floor
            use_tiling=False,
            mock_predictor=mock_predict
        )

        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        detections = detector.detect(frame, frame_idx=1, timestamp_sec=0.016)

        # Low-confidence candidate must NOT have been discarded!
        self.assertEqual(len(detections), 2)
        confs = [d.confidence for d in detections]
        self.assertIn(0.14, confs)
        self.assertIn(0.85, confs)

    # -------------------------------------------------------------------------
    # 7. Bowler Detector Non-Hand Constraint Test
    # -------------------------------------------------------------------------

    def test_bowler_detector_never_confused_with_hand_landmark(self):
        """
        Requirement: 'Do not confuse a bowler bounding box with a precise bowling-hand landmark.'
        Verifies BowlerDetector explicitly sets is_hand_landmark=False and landmark_type='body_bbox'.
        """
        def mock_bowler_predict(img):
            # Returns a person bbox in the foreground
            return [([100.0, 200.0, 300.0, 800.0], 0.90, 0)]

        bowler_det = BowlerDetector(
            confidence_threshold=0.35,
            mock_predictor=mock_bowler_predict
        )

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        bowler_obs = bowler_det.detect(frame, frame_idx=5, timestamp_sec=0.040)

        self.assertIsNotNone(bowler_obs)
        # Architectural assertion
        self.assertFalse(bowler_obs.is_hand_landmark)
        self.assertEqual(bowler_obs.landmark_type, "body_bbox")
        self.assertEqual(bowler_obs.width, 200.0)
        self.assertEqual(bowler_obs.height, 600.0)

    # -------------------------------------------------------------------------
    # 8. Stumps Detector Non-Override Guarantee Test
    # -------------------------------------------------------------------------

    def test_stumps_detector_calibration_non_override(self):
        """
        Requirement: 'Do not allow stumps detection to silently change the manual calibration.'
        Verifies that stumps drift checking NEVER alters or overwrites the manual calibration points.
        """
        stumps_det = StumpsDetector()

        manual_calibration_points = {
            "near_base": (960.0, 850.0),
            "near_top": (960.0, 680.0),
            "far_base": (960.0, 480.0),
            "far_top": (960.0, 430.0),
        }
        original_copy = dict(manual_calibration_points)

        stump_dets = [
            StumpsObservation(1, 0.0, [950.0, 675.0, 970.0, 852.0], 0.92, "near"),
            StumpsObservation(1, 0.0, [954.0, 428.0, 966.0, 481.0], 0.88, "far")
        ]

        # Verify against calibration
        drift_rep = stumps_det.verify_against_calibration(stump_dets, manual_calibration_points)

        # 1. Manual points must remain completely unchanged
        self.assertEqual(manual_calibration_points, original_copy)
        self.assertTrue(drift_rep["calibration_preserved"])
        self.assertFalse(drift_rep["is_drift_detected"])
        self.assertLess(drift_rep["max_drift_px"], 10.0)

        # 2. Test security violation if mutation is attempted
        class MaliciousMutator(dict):
            def __getitem__(self, key):
                val = super().__getitem__(key)
                # Tamper with point
                super().__setitem__(key, (999.0, 999.0))
                return val

        tampered_points = MaliciousMutator(original_copy)
        with self.assertRaises(CalibrationSecurityError):
            stumps_det.verify_against_calibration(stump_dets, tampered_points)

    # -------------------------------------------------------------------------
    # 9. Detection Manager Integration Test
    # -------------------------------------------------------------------------

    def test_detection_manager_orchestration(self):
        """
        Verifies DetectionManager orchestrates all detectors, preserves raw candidates,
        and records full metadata (cx, cy, w, h, apparent_diameter_px, frame_index, timestamp).
        """
        def mock_ball_predict(img):
            return [([100.0, 100.0, 116.0, 116.0], 0.80, 0)]

        def mock_bowler_predict(img):
            return [([400.0, 300.0, 600.0, 900.0], 0.91, 0)]

        def mock_stumps_predict(img):
            return [([950.0, 700.0, 970.0, 850.0], 0.89, 0)]

        ball_det = BallDetector(use_tiling=False, mock_predictor=mock_ball_predict)
        bowler_det = BowlerDetector(mock_predictor=mock_bowler_predict)
        stumps_det = StumpsDetector(mock_predictor=mock_stumps_predict)

        mgr = DetectionManager(
            ball_detector=ball_det,
            bowler_detector=bowler_det,
            stumps_detector=stumps_det,
            stumps_interval=1
        )

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        res = mgr.process_frame(frame, frame_idx=0, timestamp_sec=0.0)

        self.assertEqual(len(res["ball_candidates"]), 1)
        ball = res["ball_candidates"][0]
        self.assertEqual(ball.center, [108.0, 108.0])
        self.assertEqual(ball.width, 16.0)
        self.assertEqual(ball.height, 16.0)
        self.assertEqual(ball.apparent_diameter_px, 16.0)
        self.assertEqual(ball.frame_index, 0)
        self.assertEqual(ball.timestamp_sec, 0.0)

        # Raw candidates preserved
        self.assertEqual(len(res["raw_ball_candidates"]), 1)
        self.assertEqual(len(mgr.raw_ball_observations[0]), 1)

        # Bowler and stumps
        self.assertIsNotNone(res["bowler"])
        self.assertFalse(res["bowler"].is_hand_landmark)
        self.assertEqual(len(res["stumps"]), 1)

        # Summary
        summary = mgr.get_summary()
        self.assertEqual(summary["total_frames_processed"], 1)
        self.assertEqual(summary["frames_with_ball"], 1)

    # -------------------------------------------------------------------------
    # 10. Detection Visualizer Rendering Test
    # -------------------------------------------------------------------------

    def test_detection_visualizer_rendering(self):
        """Verifies DetectionVisualizer renders annotated diagnostic frame with overlays."""
        vis = DetectionVisualizer(show_tile_grid=True, show_raw_candidates=True)

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        ball = BallObservation.from_bbox(1, 0.016, [600.0, 350.0, 614.0, 364.0], 0.88, "full_frame")
        raw_ball = BallObservation.from_bbox(1, 0.016, [600.0, 350.0, 614.0, 364.0], 0.88, "tile_x0_y0")
        bowler = BowlerObservation(1, 0.016, [200.0, 150.0, 350.0, 550.0], 0.92, [275.0, 350.0], 150.0, 400.0)
        stump = StumpsObservation(1, 0.016, [630.0, 480.0, 650.0, 560.0], 0.89, "far")
        tiles = [(0, 0, 640, 640), (480, 0, 1120, 640)]

        annotated = vis.render_detection_frame(
            frame_bgr=frame,
            frame_idx=1,
            timestamp_sec=0.016,
            merged_balls=[ball],
            raw_balls=[raw_ball],
            bowler=bowler,
            stumps=[stump],
            tile_boxes=tiles
        )

        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, (720, 1280, 3))
        # Ensure image was drawn on (not all black)
        self.assertGreater(np.count_nonzero(annotated), 1000)


if __name__ == "__main__":
    unittest.main()
