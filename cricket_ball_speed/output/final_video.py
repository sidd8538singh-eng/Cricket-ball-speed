"""
Final Processed Video Generator.
Produces clean, broadcast-ready output video showing ONLY:
  1. Calculated bowling speed in km/h.
  2. Date and time.

STRICT VISUAL CONSTRAINTS:
- The speed badge is positioned in the upper-left region (not touching the top edge, not in the middle-left).
- Strictly OMITS ball markers, bounding boxes, frame counters, calibration lines, and diagnostic labels.
- Freezes the final frame for a configurable duration (default 1.5s) to allow viewer comprehension.
- Independent of detection and optimization pipelines (operates on plain data dicts or result objects).
"""

from typing import Optional, Tuple, Dict, Any, Union
import datetime
import os
import cv2
import numpy as np


class FinalVideoRenderer:
    """
    Renders clean broadcast-style speed telemetry graphics onto video frames.
    """

    def __init__(
        self,
        freeze_seconds: float = 1.5,
        margin_left_ratio: float = 0.035,   # ~67px at 1080p
        margin_top_ratio: float = 0.065,    # ~70px at 1080p (safe broadcast margin, not touching top)
        badge_width_ratio: float = 0.170,   # ~326px at 1080p
        badge_height_ratio: float = 0.090   # ~97px at 1080p
    ):
        self.freeze_sec = max(0.0, float(freeze_seconds))
        self.ml_ratio = margin_left_ratio
        self.mt_ratio = margin_top_ratio
        self.bw_ratio = badge_width_ratio
        self.bh_ratio = badge_height_ratio

    def compute_badge_rect(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """Computes resolution-independent upper-left badge coordinates."""
        bx = int(round(frame_w * self.ml_ratio))
        by = int(round(frame_h * self.mt_ratio))
        bw = max(240, int(round(frame_w * self.bw_ratio)))
        bh = max(80, int(round(frame_h * self.bh_ratio)))
        return bx, by, bw, bh

    def render_badge(
        self,
        frame: np.ndarray,
        speed_text: str,
        timestamp_str: str,
        is_measured: bool = True
    ) -> np.ndarray:
        """
        Overlays the clean broadcast speed badge in the upper-left area.
        Contains strictly speed and date/time. Zero diagnostics.
        """
        h, w = frame.shape[:2]
        bx, by, bw, bh = self.compute_badge_rect(w, h)
        out_frame = frame.copy()

        # Premium semi-transparent dark glass card
        overlay = out_frame.copy()
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (18, 22, 28), -1)

        # Left accent stripe: vivid green for MEASURED, deep coral for NOT MEASURED
        stripe_color = (46, 204, 113) if is_measured else (70, 70, 230)
        cv2.rectangle(overlay, (bx, by), (bx + 6, by + bh), stripe_color, -1)

        cv2.addWeighted(overlay, 0.85, out_frame, 0.15, 0, out_frame)

        # Subtle card border
        cv2.rectangle(out_frame, (bx, by), (bx + bw, by + bh), (60, 68, 76), 1)

        # 1. Date and Time (Upper row of badge)
        cv2.putText(
            out_frame, timestamp_str, (bx + 16, by + int(bh * 0.30)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44 * (h / 1080.0), (190, 200, 210), 1, cv2.LINE_AA
        )

        # 2. Calculated Speed in km/h (Main visual anchor)
        val_color = (255, 255, 255) if is_measured else (180, 180, 245)
        font_scale = 1.05 * (h / 1080.0) if is_measured else 0.72 * (h / 1080.0)
        cv2.putText(
            out_frame, speed_text, (bx + 16, by + int(bh * 0.78)),
            cv2.FONT_HERSHEY_DUPLEX, max(0.5, font_scale), val_color, 2, cv2.LINE_AA
        )

        return out_frame


def generate_final_video(
    input_video_path: str,
    output_video_path: str,
    qc_result: Union[Any, Dict[str, Any], bool],
    release_speed: Optional[Union[Any, Dict[str, Any], float]] = None,
    fps: float = 60.0,
    freeze_last_frame_seconds: float = 1.5,
    custom_timestamp_str: Optional[str] = None
) -> str:
    """
    Renders the broadcast final video showing ONLY bowling speed and date/time.
    Supports decoupled inputs (dataclass objects, dicts, or raw values).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open input video: {input_video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    in_fps = float(cap.get(cv2.CAP_PROP_FPS))
    out_fps = in_fps if in_fps > 0 else fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, out_fps, (w, h))

    renderer = FinalVideoRenderer(freeze_seconds=freeze_last_frame_seconds)

    # Resolve measurement status (decoupled from specific types)
    if isinstance(qc_result, bool):
        is_measured = qc_result
    elif isinstance(qc_result, dict):
        is_measured = bool(qc_result.get("is_measured", False))
    else:
        is_measured = getattr(qc_result, "is_measured", False)

    # Resolve speed value (decoupled from specific types)
    speed_kmh = None
    if release_speed is not None:
        if isinstance(release_speed, (int, float)):
            speed_kmh = float(release_speed)
        elif isinstance(release_speed, dict):
            speed_kmh = release_speed.get("speed_km_h")
        else:
            speed_kmh = getattr(release_speed, "speed_km_h", None)

    if is_measured and speed_kmh is not None and speed_kmh > 0.0:
        speed_str = f"{speed_kmh:.1f} km/h"
    else:
        speed_str = "SPEED: NOT MEASURED"

    time_str = custom_timestamp_str or datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    last_frame = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            rendered = renderer.render_badge(
                frame, speed_str, time_str, is_measured=is_measured
            )
            writer.write(rendered)
            last_frame = rendered

        # Freeze last frame if configured
        if last_frame is not None and freeze_last_frame_seconds > 0:
            freeze_frames = int(round(out_fps * freeze_last_frame_seconds))
            for _ in range(freeze_frames):
                writer.write(last_frame)
    finally:
        cap.release()
        writer.release()

    return output_video_path
