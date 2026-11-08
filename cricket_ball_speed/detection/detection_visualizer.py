"""
Detection Visualizer: Renders visual diagnostics for object detection.
Displays overlapping tile grids, raw vs merged ball candidates, apparent diameters,
bowler bounding box (clearly distinguished from hand landmark), and stumps (clearly
labeled as locked calibration drift check).
"""

from typing import List, Optional, Dict, Any, Tuple
import cv2
import numpy as np

from ball_processing.ball_observations import BallObservation
from .bowler_detector import BowlerObservation
from .stumps_detector import StumpsObservation


class DetectionVisualizer:
    """
    Renders diagnostic overlays for AI detection results.
    """

    COLOR_RAW_TILE = (0, 165, 255)       # Orange
    COLOR_RAW_FULL = (255, 191, 0)       # Deep Sky Blue
    COLOR_MERGED_BALL = (0, 255, 0)      # Bright Green
    COLOR_CENTER_DOT = (0, 0, 255)       # Red
    COLOR_BOWLER_BOX = (255, 128, 0)     # Blue/Teal
    COLOR_STUMPS_BOX = (255, 0, 255)     # Magenta
    COLOR_TILE_GRID = (120, 120, 120)    # Subtle Gray

    def __init__(
        self,
        show_tile_grid: bool = True,
        show_raw_candidates: bool = True,
        show_labels: bool = True,
        font_scale: float = 0.5,
        line_thickness: int = 2
    ):
        self.show_tile_grid = show_tile_grid
        self.show_raw_candidates = show_raw_candidates
        self.show_labels = show_labels
        self.font_scale = font_scale
        self.line_thickness = line_thickness

    def render_detection_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        merged_balls: List[BallObservation],
        raw_balls: Optional[List[BallObservation]] = None,
        bowler: Optional[BowlerObservation] = None,
        stumps: Optional[List[StumpsObservation]] = None,
        tile_boxes: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> np.ndarray:
        """
        Renders a full diagnostic detection frame with all overlays.
        """
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]

        # 1. Tile Grid (Subtle overlay)
        if self.show_tile_grid and tile_boxes:
            grid_overlay = vis.copy()
            for idx, (tx1, ty1, tx2, ty2) in enumerate(tile_boxes):
                cv2.rectangle(grid_overlay, (tx1, ty1), (tx2, ty2), self.COLOR_TILE_GRID, 1)
                cv2.putText(
                    grid_overlay,
                    f"Tile {idx}",
                    (tx1 + 6, ty1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA
                )
            cv2.addWeighted(grid_overlay, 0.45, vis, 0.55, 0, vis)

        # 2. Raw Ball Candidates (pre-deduplication)
        if self.show_raw_candidates and raw_balls:
            for c in raw_balls:
                bx1, by1, bx2, by2 = [int(round(v)) for v in c.bbox]
                is_full = "full_frame" in c.detection_source
                color = self.COLOR_RAW_FULL if is_full else self.COLOR_RAW_TILE
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_4)

        # 3. Merged / Confirmed Ball Detections
        for ball in merged_balls:
            bx1, by1, bx2, by2 = [int(round(v)) for v in ball.bbox]
            cx, cy = int(round(ball.center[0])), int(round(ball.center[1]))
            radius = max(3, int(round(ball.apparent_diameter_px / 2.0)))

            # Ball bounding box and center dot
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), self.COLOR_MERGED_BALL, self.line_thickness)
            cv2.circle(vis, (cx, cy), 3, self.COLOR_CENTER_DOT, -1)
            cv2.circle(vis, (cx, cy), radius, self.COLOR_MERGED_BALL, 1)

            if self.show_labels:
                src_info = ball.extra.get("merged_sources", [ball.detection_source])
                src_str = f"[{len(src_info)} srcs]" if len(src_info) > 1 else ball.detection_source
                label = f"Ball {ball.confidence:.2f} (d={ball.apparent_diameter_px:.1f}px) {src_str}"
                cv2.putText(
                    vis,
                    label,
                    (bx1, max(15, by1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    self.COLOR_MERGED_BALL,
                    1,
                    cv2.LINE_AA
                )

        # 4. Bowler Bounding Box (Explicitly labeled as body box, NOT hand)
        if bowler is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in bowler.bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), self.COLOR_BOWLER_BOX, self.line_thickness)
            if self.show_labels:
                label_bowler = f"Bowler Body [NOT Hand] ({bowler.confidence:.2f})"
                cv2.putText(
                    vis,
                    label_bowler,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    self.COLOR_BOWLER_BOX,
                    2,
                    cv2.LINE_AA
                )

        # 5. Stumps Bounding Boxes (Explicitly labeled as read-only calib check)
        if stumps:
            for st in stumps:
                sx1, sy1, sx2, sy2 = [int(round(v)) for v in st.bbox]
                cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), self.COLOR_STUMPS_BOX, self.line_thickness)
                if self.show_labels:
                    stump_lbl = f"Stumps ({st.stump_type}) [Calib Locked]"
                    cv2.putText(
                        vis,
                        stump_lbl,
                        (sx1, max(15, sy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        self.COLOR_STUMPS_BOX,
                        1,
                        cv2.LINE_AA
                    )

        # 6. Telemetry HUD Bar
        hud = vis.copy()
        cv2.rectangle(hud, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.addWeighted(hud, 0.75, vis, 0.25, 0, vis)

        raw_count = len(raw_balls) if raw_balls else 0
        merged_count = len(merged_balls)
        telemetry_text = (
            f"Frame: {frame_idx:04d} | Time: {timestamp_sec:.3f}s | "
            f"Ball Candidates: {merged_count} merged ({raw_count} raw) | "
            f"Bowler: {'DETECTED' if bowler else 'None'} | "
            f"Stumps: {len(stumps) if stumps else 0}"
        )
        cv2.putText(
            vis,
            telemetry_text,
            (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return vis
