"""
Bounding Box Refinement: Sub-pixel center refinement and contour analysis.
Extracts localized image patch around ball candidate to compute precise sub-pixel centroid.
Preserves raw observations intact without modification.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from .ball_observations import BallObservation


class BoundingBoxRefiner:
    """
    Refines crude YOLO ball bounding boxes into sub-pixel accurate centroids
    using image moments, Otsu/adaptive thresholding, and contour circularity.
    """

    def __init__(self, padding_px: int = 4):
        self.padding = padding_px

    def refine(self, frame_bgr: np.ndarray, raw_obs: BallObservation) -> BallObservation:
        """
        Produces a new refined BallObservation while leaving raw_obs untouched.
        """
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = raw_obs.bbox

        # Expand patch with padding
        px1 = max(0, int(round(x1 - self.padding)))
        py1 = max(0, int(round(y1 - self.padding)))
        px2 = min(w, int(round(x2 + self.padding)))
        py2 = min(h, int(round(y2 + self.padding)))

        patch = frame_bgr[py1:py2, px1:px2]
        if patch.size == 0 or patch.shape[0] < 3 or patch.shape[1] < 3:
            return raw_obs

        # Convert to grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # High-contrast threshold (ball is typically bright red, white, or pink against pitch/grass)
        # Using Otsu or inverted Otsu depending on contrast
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if ball is darker than background
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        refined_cx = raw_obs.center[0]
        refined_cy = raw_obs.center[1]
        refined_diam = raw_obs.apparent_diameter_px

        if contours:
            # Find contour closest to original center
            local_cx_target = raw_obs.center[0] - px1
            local_cy_target = raw_obs.center[1] - py1
            
            best_contour = None
            min_dist = float("inf")
            for c in contours:
                M = cv2.moments(c)
                if M["m00"] > 1e-4:
                    mcx = M["m10"] / M["m00"]
                    mcy = M["m01"] / M["m00"]
                    dist = np.hypot(mcx - local_cx_target, mcy - local_cy_target)
                    if dist < min_dist:
                        min_dist = dist
                        best_contour = (mcx, mcy, c)

            if best_contour is not None and min_dist < max(raw_obs.width, raw_obs.height):
                mcx, mcy, c = best_contour
                refined_cx = float(mcx + px1)
                refined_cy = float(mcy + py1)
                
                # Fit minimum enclosing circle for refined diameter
                _, radius = cv2.minEnclosingCircle(c)
                if radius > 1.0:
                    refined_diam = float(radius * 2.0)

        # Build refined observation
        half_w = (raw_obs.width / 2.0)
        half_h = (raw_obs.height / 2.0)
        refined_bbox = [
            refined_cx - half_w,
            refined_cy - half_h,
            refined_cx + half_w,
            refined_cy + half_h
        ]

        refined_obs = BallObservation(
            frame_index=raw_obs.frame_index,
            timestamp_sec=raw_obs.timestamp_sec,
            bbox=refined_bbox,
            confidence=raw_obs.confidence,
            center=[refined_cx, refined_cy],
            width=raw_obs.width,
            height=raw_obs.height,
            apparent_diameter_px=refined_diam,
            detection_source="refined",
            acceptance_status="PENDING",
            quality_score=raw_obs.quality_score,
            extra={"raw_center": raw_obs.center, "raw_bbox": raw_obs.bbox}
        )
        return refined_obs
