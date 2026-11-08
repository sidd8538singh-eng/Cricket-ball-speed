"""
Tiled Inference: Overlapping image slicing, optional upscaling, coordinate restoration,
and cross-tile Non-Maximum Suppression (NMS) for small cricket ball detection.
Enables accurate detection of distant tiny balls (4-20 pixels) in high-resolution frames (1080p, 4K).
"""

from typing import List, Tuple, Dict, Any, Callable, Optional
import numpy as np
import cv2
import logging

from ball_processing.ball_observations import BallObservation

logger = logging.getLogger("CricketBallSpeed.TiledInference")


def compute_tiles(
    img_w: int,
    img_h: int,
    tile_w: int = 640,
    tile_h: int = 640,
    overlap_x: float = 0.20,
    overlap_y: float = 0.20
) -> List[Tuple[int, int, int, int]]:
    """
    Computes overlapping tile bounding boxes [x1, y1, x2, y2] covering the entire frame.
    Guarantees full coverage of frame boundaries without clipping.
    """
    tile_w = min(tile_w, img_w)
    tile_h = min(tile_h, img_h)

    stride_x = max(1, int(round(tile_w * (1.0 - overlap_x))))
    stride_y = max(1, int(round(tile_h * (1.0 - overlap_y))))

    tiles: List[Tuple[int, int, int, int]] = []
    y1 = 0
    while y1 < img_h:
        y2 = min(y1 + tile_h, img_h)
        if y2 - y1 < tile_h and y1 > 0:
            y1 = max(0, img_h - tile_h)
            y2 = img_h

        x1 = 0
        while x1 < img_w:
            x2 = min(x1 + tile_w, img_w)
            if x2 - x1 < tile_w and x1 > 0:
                x1 = max(0, img_w - tile_w)
                x2 = img_w

            tiles.append((int(x1), int(y1), int(x2), int(y2)))
            if x2 >= img_w:
                break
            x1 += stride_x

        if y2 >= img_h:
            break
        y1 += stride_y

    return tiles


def restore_tile_coordinates(
    crop_bbox: List[float],
    tile_x1: float,
    tile_y1: float,
    upscale_factor: float = 1.0
) -> List[float]:
    """
    Restores bounding box [x1, y1, x2, y2] from a potentially upscaled tile crop
    back into original video frame pixel coordinates.

    Mathematical mapping:
        x_orig = tile_x1 + (x_crop / upscale_factor)
        y_orig = tile_y1 + (y_crop / upscale_factor)
    """
    s = max(1e-6, float(upscale_factor))
    x1, y1, x2, y2 = crop_bbox
    orig_x1 = float(tile_x1) + (float(x1) / s)
    orig_y1 = float(tile_y1) + (float(y1) / s)
    orig_x2 = float(tile_x1) + (float(x2) / s)
    orig_y2 = float(tile_y1) + (float(y2) / s)
    return [orig_x1, orig_y1, orig_x2, orig_y2]


def apply_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.40
) -> List[int]:
    """
    Standard Non-Maximum Suppression (NMS) over bounding boxes [x1, y1, x2, y2].
    Returns indices of kept boxes in descending score order.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Computes Intersection over Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


def merge_candidate_observations(
    candidates: List[BallObservation],
    iou_threshold: float = 0.35
) -> List[BallObservation]:
    """
    Removes duplicate detections resulting from overlapping tiles and full-frame inference.
    Preserves auditability: records contributing detection sources in extra['merged_sources'].
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        c = candidates[0]
        if "merged_sources" not in c.extra:
            c.extra["merged_sources"] = [c.detection_source]
        return [c]

    boxes = np.array([c.bbox for c in candidates], dtype=np.float32)
    scores = np.array([c.confidence for c in candidates], dtype=np.float32)

    keep_indices = apply_nms(boxes, scores, iou_threshold=iou_threshold)
    kept_observations: List[BallObservation] = []

    for idx in keep_indices:
        primary = candidates[idx]
        merged_sources = [primary.detection_source]

        # Find all other candidates that overlapped with this primary candidate
        for other_idx, other in enumerate(candidates):
            if other_idx != idx and compute_iou(primary.bbox, other.bbox) > iou_threshold:
                merged_sources.append(other.detection_source)

        primary.extra["merged_sources"] = list(set(merged_sources))
        primary.extra["corroboration_count"] = len(primary.extra["merged_sources"])
        kept_observations.append(primary)

    return kept_observations


class TiledInferenceEngine:
    """
    Orchestrates tiled inference over large video frames.
    Slices frames, applies optional upscaling, invokes detector callback,
    restores coordinates to original video space, and deduplicates candidates.
    """

    def __init__(
        self,
        tile_width: int = 640,
        tile_height: int = 640,
        overlap_x: float = 0.20,
        overlap_y: float = 0.20,
        upscale_factor: float = 1.0,
        nms_iou_threshold: float = 0.35,
        min_confidence: float = 0.12
    ):
        self.tile_w = tile_width
        self.tile_h = tile_height
        self.overlap_x = overlap_x
        self.overlap_y = overlap_y
        self.upscale_factor = max(1.0, float(upscale_factor))
        self.nms_iou = nms_iou_threshold
        self.min_conf = min_confidence

    def get_tiles(self, img_w: int, img_h: int) -> List[Tuple[int, int, int, int]]:
        """Returns tile bounding boxes for the given image dimensions."""
        return compute_tiles(img_w, img_h, self.tile_w, self.tile_h, self.overlap_x, self.overlap_y)

    def extract_and_prepare_tile(
        self,
        frame_bgr: np.ndarray,
        tile_coords: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, float]:
        """
        Extracts tile crop and applies optional upscaling.
        Returns: (processed_tile_crop, active_upscale_factor)
        """
        tx1, ty1, tx2, ty2 = tile_coords
        crop = frame_bgr[ty1:ty2, tx1:tx2]

        if self.upscale_factor > 1.001:
            h, w = crop.shape[:2]
            new_w = int(round(w * self.upscale_factor))
            new_h = int(round(h * self.upscale_factor))
            upscaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            return upscaled, self.upscale_factor

        return crop, 1.0

    def process_tiled_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        predict_fn: Callable[[np.ndarray], List[Tuple[List[float], float, int]]],
        target_classes: Optional[List[int]] = None
    ) -> Tuple[List[BallObservation], List[BallObservation]]:
        """
        Executes tiled inference across all overlapping tiles.
        predict_fn signature: frame_or_crop -> List of (bbox_xyxy, confidence, class_id)

        Returns:
            (merged_candidates, raw_candidates)
        """
        h, w = frame_bgr.shape[:2]
        tiles = self.get_tiles(w, h)
        raw_candidates: List[BallObservation] = []

        for tile_coords in tiles:
            tx1, ty1, tx2, ty2 = tile_coords
            crop, scale = self.extract_and_prepare_tile(frame_bgr, tile_coords)

            detections = predict_fn(crop)
            for xyxy_crop, conf, cls_id in detections:
                if target_classes and cls_id not in target_classes:
                    continue
                if conf < self.min_conf:
                    continue

                # Restore to original video frame coordinates
                global_xyxy = restore_tile_coordinates(xyxy_crop, tx1, ty1, upscale_factor=scale)
                source_tag = f"tile_x{tx1}_y{ty1}" if scale <= 1.0 else f"tile_x{tx1}_y{ty1}_up{scale:.1f}x"

                obs = BallObservation.from_bbox(
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    bbox=global_xyxy,
                    confidence=conf,
                    detection_source=source_tag
                )
                raw_candidates.append(obs)

        merged_candidates = merge_candidate_observations(raw_candidates, iou_threshold=self.nms_iou)
        return merged_candidates, raw_candidates
