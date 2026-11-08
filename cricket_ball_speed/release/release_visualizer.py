"""
Release Visualizer: Renders visual diagnostics for bowling release events.
Displays bowler hand landmarks, release point C1, ball-to-hand separation vector,
timing uncertainty bounds, and release audit status.
"""

from typing import Optional, Dict, Any, Tuple, List
import cv2
import numpy as np

from .release_detector import ReleaseEvent
from detection.bowler_detector import BowlerObservation


class ReleaseVisualizer:
    """
    Renders diagnostic computer vision overlays for bowling release detection.
    """

    COLOR_HAND = (0, 255, 255)         # Yellow
    COLOR_C1 = (0, 255, 0)              # Bright Green
    COLOR_VECTOR = (0, 140, 255)        # Orange
    COLOR_BOWLER = (255, 128, 0)        # Blue/Teal
    COLOR_ALERT = (0, 0, 255)           # Red
    COLOR_TEXT = (255, 255, 255)        # White

    def __init__(
        self,
        show_hand: bool = True,
        show_c1: bool = True,
        show_separation_vector: bool = True,
        font_scale: float = 0.5,
        line_thickness: int = 2
    ):
        self.show_hand = show_hand
        self.show_c1 = show_c1
        self.show_sep = show_separation_vector
        self.font_scale = font_scale
        self.line_thickness = line_thickness

    def render_release_frame(
        self,
        frame_bgr: np.ndarray,
        release_event: ReleaseEvent,
        bowler_obs: Optional[BowlerObservation] = None
    ) -> np.ndarray:
        """
        Overlays release telemetry, hand landmarks, C1 coordinates, and timing uncertainty on frame.
        """
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]

        # 1. Bowler Bounding Box
        if bowler_obs is not None:
            bx1, by1, bx2, by2 = [int(round(c)) for c in bowler_obs.bbox]
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), self.COLOR_BOWLER, 1, cv2.LINE_AA)

        # 2. Bowler Hand Landmark
        hx, hy = [int(round(c)) for c in release_event.hand_position_pixel]
        if self.show_hand and hx > 0 and hy > 0:
            cv2.drawMarker(vis, (hx, hy), self.COLOR_HAND, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
            cv2.circle(vis, (hx, hy), 6, self.COLOR_HAND, 1, cv2.LINE_AA)
            cv2.putText(
                vis, "Hand Apex", (hx + 10, hy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.COLOR_HAND, 1, cv2.LINE_AA
            )

        # 3. Release Point C1
        c1x, c1y = [int(round(c)) for c in release_event.c1_pixel]
        if self.show_c1 and c1x > 0 and c1y > 0:
            cv2.circle(vis, (c1x, c1y), 8, self.COLOR_C1, self.line_thickness, cv2.LINE_AA)
            cv2.circle(vis, (c1x, c1y), 3, self.COLOR_C1, -1)
            cv2.putText(
                vis, f"C1 ({c1x}, {c1y})", (c1x + 10, c1y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.COLOR_C1, 2, cv2.LINE_AA
            )

        # 4. Separation Vector from Hand to C1
        if self.show_sep and (hx > 0 and hy > 0) and (c1x > 0 and c1y > 0):
            cv2.line(vis, (hx, hy), (c1x, c1y), self.COLOR_VECTOR, 1, cv2.LINE_AA)

        # 5. Diagnostic Release HUD Banner
        hud = vis.copy()
        cv2.rectangle(hud, (0, h - 54), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(hud, 0.80, vis, 0.20, 0, vis)

        if release_event.is_valid and release_event.status == "MEASURED":
            mode_tag = "DIRECT" if release_event.observation_mode == "DIRECTLY_OBSERVED" else "ESTIMATED"
            unc_ms = release_event.timing_uncertainty_sec * 1000.0
            banner_text = (
                f"RELEASE DETECTED [{mode_tag}] | Frame: {release_event.frame_index} | "
                f"Time: {release_event.timestamp_sec:.3f}s (+/- {unc_ms:.1f}ms) | "
                f"Conf: {release_event.confidence:.2f} | "
                f"Sep Vel: {release_event.separation_velocity_px_s:.0f} px/s | "
                f"Hand Dist: {release_event.ball_to_hand_distance_px:.1f}px"
            )
            badge_color = self.COLOR_C1
        else:
            banner_text = (
                f"RELEASE NOT MEASURED | Reason: {release_event.rejection_reason or 'Insufficient observations'}"
            )
            badge_color = self.COLOR_ALERT

        cv2.putText(
            vis, banner_text, (16, h - 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, badge_color, 1, cv2.LINE_AA
        )

        return vis
