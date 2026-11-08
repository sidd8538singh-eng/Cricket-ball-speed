"""
Unit tests for coordinate systems, world geometry, and coordinate manager.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from calibration.world_coordinates import WorldPitchGeometry, CALIBRATION_POINT_KEYS
from calibration.coordinate_manager import CoordinateManager


class TestCoordinates(unittest.TestCase):
    def setUp(self):
        self.geom = WorldPitchGeometry()
        self.coord_mgr = CoordinateManager(original_width=1920, original_height=1080)

    def test_canonical_point_keys_order(self):
        """Strict ordering of the 8 canonical calibration points."""
        expected = [
            "near_top",
            "near_base",
            "far_top",
            "far_base",
            "pitch_left_near",
            "pitch_right_near",
            "pitch_left_far",
            "pitch_right_far"
        ]
        self.assertEqual(CALIBRATION_POINT_KEYS, expected)

    def test_world_coordinates_dimensions(self):
        """Validates canonical 3D coordinates against ICC cricket standards."""
        pts = self.geom.get_world_coordinates()
        self.assertEqual(len(pts), 8)

        # Near stumps: base at (0,0,0), top at (0,0,0.711)
        np.testing.assert_allclose(pts["near_base"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(pts["near_top"], [0.0, 0.0, 0.711])

        # Far stumps: base at (0, 20.12, 0), top at (0, 20.12, 0.711)
        np.testing.assert_allclose(pts["far_base"], [0.0, 20.12, 0.0])
        np.testing.assert_allclose(pts["far_top"], [0.0, 20.12, 0.711])

        # Pitch width: 3.048m (half-width = 1.524m)
        half_w = 3.048 / 2.0
        np.testing.assert_allclose(pts["pitch_left_near"], [-half_w, 0.0, 0.0])
        np.testing.assert_allclose(pts["pitch_right_near"], [+half_w, 0.0, 0.0])
        np.testing.assert_allclose(pts["pitch_left_far"], [-half_w, 20.12, 0.0])
        np.testing.assert_allclose(pts["pitch_right_far"], [+half_w, 20.12, 0.0])

    def test_coordinate_manager_roundtrip(self):
        """Validates display zoom/pan to original pixel round-trip transformation."""
        orig_x, orig_y = 540.0, 360.0
        scale = 2.5
        pan_x, pan_y = -100.0, -150.0

        disp_x, disp_y = self.coord_mgr.original_to_display(orig_x, orig_y, scale, pan_x, pan_y)
        recov_x, recov_y = self.coord_mgr.display_to_original(disp_x, disp_y, scale, pan_x, pan_y)

        self.assertAlmostEqual(orig_x, recov_x, delta=0.5)
        self.assertAlmostEqual(orig_y, recov_y, delta=0.5)

    def test_tile_offset_mapping(self):
        """Verifies local tile bounding box maps to global frame coordinates."""
        tile_box = [10.0, 20.0, 30.0, 45.0]
        offset_x, offset_y = 600, 400
        global_box = self.coord_mgr.tile_to_original(tile_box, offset_x, offset_y)
        self.assertEqual(global_box, (610.0, 420.0, 630.0, 445.0))


if __name__ == "__main__":
    unittest.main()
