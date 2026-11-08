"""
Manual Eight-Point Calibration Interface and Video Locker.
Standalone module providing interactive landmark selection, frame changing,
magnifier zoom/pan, Line A-B reference rendering, duplicate rejection,
structured JSON saving, video hash locking, and resolution mismatch detection.
"""

from typing import Optional, Dict, Any, Tuple, List
import os
import sys
import json
import datetime
import argparse
import cv2
import numpy as np

from .point_selector import (
    CalibrationPointManager,
    POINT_DESCRIPTIONS,
    POINT_SHORT_NAMES,
    POINT_COLORS
)
from .coordinate_manager import CoordinateManager
from .world_coordinates import WorldPitchGeometry, CALIBRATION_POINT_KEYS
from .camera_calibration import CameraCalibrator
from .calibration_validator import CalibrationValidator
from .camera_model import CameraModel
from .reprojection import compute_reprojection_report, ReprojectionReport


CALIBRATION_VERSION = "1.0.0"


class CalibrationResolutionMismatchError(ValueError):
    """Raised when the video resolution does not match the locked calibration resolution."""
    pass


class CalibrationLockError(RuntimeError):
    """Raised when video identity/hash does not match the locked calibration file."""
    pass


class ManualCalibrationUI:
    """
    OpenCV GUI for user landmark selection.
    Coordinates are strictly tracked and preserved in original video pixel resolution.
    Supports scrubbing video frames, zoom, pan, undo, reset, per-point replacement,
    drawing the imaginary Line A-B, and visual geometric preview.
    """

    def __init__(
        self,
        video_path: Optional[str] = None,
        frame_bgr: Optional[np.ndarray] = None,
        frame_idx: int = 0,
        frame_timestamp: float = 0.0,
        video_hash: Optional[str] = None,
        video_width: Optional[int] = None,
        video_height: Optional[int] = None,
        pitch_geometry: Optional[WorldPitchGeometry] = None,
        window_name: str = "Cricket Manual 8-Point Calibration"
    ):
        self.video_path = video_path
        self.geometry = pitch_geometry or WorldPitchGeometry()
        self.window_name = window_name

        # If video_path is provided, open video capture to allow changing frames
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames = 1
        self.fps = 60.0

        if video_path and os.path.isfile(video_path):
            self.cap = cv2.VideoCapture(video_path)
            if self.cap.isOpened():
                self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
                if self.fps <= 0 or np.isnan(self.fps):
                    self.fps = 60.0
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_bgr is None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame_bgr = self.cap.read()
                    if not ret or frame_bgr is None:
                        raise IOError(f"Could not read frame {frame_idx} from {video_path}")
            else:
                self.cap = None

        if frame_bgr is None:
            raise ValueError("Either a valid frame_bgr or video_path must be provided.")

        self.orig_frame = frame_bgr.copy()
        self.orig_h, self.orig_w = frame_bgr.shape[:2]
        if video_width:
            self.orig_w = video_width
        if video_height:
            self.orig_h = video_height

        self.frame_idx = frame_idx
        self.frame_timestamp = frame_timestamp if frame_timestamp > 0 else (frame_idx / self.fps)
        self.video_hash = video_hash or self._compute_or_default_hash()

        self.coord_mgr = CoordinateManager(self.orig_w, self.orig_h)
        self.point_mgr = CalibrationPointManager(self.orig_w, self.orig_h)

        # Viewport zoom & pan
        self.zoom_scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.mouse_orig_x = 0.0
        self.mouse_orig_y = 0.0

        self.active_replace_key: Optional[str] = None
        self.status_message: Optional[str] = None
        self.status_error: bool = False
        self.is_preview_mode: bool = False

    def _compute_or_default_hash(self) -> str:
        if self.video_path and os.path.isfile(self.video_path):
            import hashlib
            sha = hashlib.sha256()
            with open(self.video_path, "rb") as f:
                while chunk := f.read(65536):
                    sha.update(chunk)
            return sha.hexdigest()
        return "synthetic_or_unhashed_video"

    def set_frame(self, new_frame_idx: int) -> bool:
        """Loads and switches to a different video frame for calibration."""
        if self.cap is None or not self.cap.isOpened():
            self._set_status_msg("Cannot change frame: video source not attached.", error=True)
            return False

        new_frame_idx = max(0, min(new_frame_idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_idx)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._set_status_msg(f"Failed to read frame {new_frame_idx}.", error=True)
            return False

        self.orig_frame = frame.copy()
        self.frame_idx = new_frame_idx
        self.frame_timestamp = new_frame_idx / self.fps
        self._set_status_msg(f"Switched to frame {self.frame_idx} (t={self.frame_timestamp:.3f}s)")
        return True

    def _set_status_msg(self, msg: str, error: bool = False):
        self.status_message = msg
        self.status_error = error

    def _mouse_callback(self, event, x, y, flags, param):
        # Track original pixel coordinate under mouse
        orig_x, orig_y = self.coord_mgr.display_to_original(
            x, y, self.zoom_scale, self.pan_x, self.pan_y
        )
        self.mouse_orig_x = orig_x
        self.mouse_orig_y = orig_y

        if event == cv2.EVENT_LBUTTONDOWN:
            target_key = self.active_replace_key or self.point_mgr.next_required_point_key
            if target_key is not None:
                try:
                    self.point_mgr.add_point(orig_x, orig_y, target_key)
                    self._set_status_msg(f"Selected {target_key} at ({orig_x:.1f}, {orig_y:.1f})")
                    self.active_replace_key = None
                    if self.point_mgr.is_complete():
                        self.is_preview_mode = True
                except ValueError as ve:
                    self._set_status_msg(str(ve), error=True)
            else:
                self._set_status_msg("All 8 points set. Press [C] to Confirm or [1-8] to re-pick.")

        elif event == cv2.EVENT_RBUTTONDOWN or event == cv2.EVENT_MBUTTONDOWN:
            self.is_panning = True
            self.pan_start_x = x - self.pan_x
            self.pan_start_y = y - self.pan_y

        elif event == cv2.EVENT_MOUSEMOVE and self.is_panning:
            self.pan_x = x - self.pan_start_x
            self.pan_y = y - self.pan_start_y

        elif event == cv2.EVENT_RBUTTONUP or event == cv2.EVENT_MBUTTONUP:
            self.is_panning = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 1.15 if flags > 0 else 0.85
            old_scale = self.zoom_scale
            self.zoom_scale = float(np.clip(self.zoom_scale * delta, 0.5, 10.0))
            self.pan_x = x - (x - self.pan_x) * (self.zoom_scale / old_scale)
            self.pan_y = y - (y - self.pan_y) * (self.zoom_scale / old_scale)

    def _render_display_frame(self) -> np.ndarray:
        scaled_w = max(1, int(round(self.orig_w * self.zoom_scale)))
        scaled_h = max(1, int(round(self.orig_h * self.zoom_scale)))
        resized = cv2.resize(self.orig_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((self.orig_h, self.orig_w, 3), dtype=np.uint8)

        tx = int(round(self.pan_x))
        ty = int(round(self.pan_y))

        src_x1 = max(0, -tx)
        src_y1 = max(0, -ty)
        src_x2 = min(scaled_w, self.orig_w - tx)
        src_y2 = min(scaled_h, self.orig_h - ty)

        dst_x1 = max(0, tx)
        dst_y1 = max(0, ty)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]

        pts = self.point_mgr.points

        # -------------------------------------------------------------
        # DRAW IMAGINARY LINE A-B (Near Base to Far Base)
        # Requirement: "Draw the imaginary line connecting the centres of near_base and far_base."
        # -------------------------------------------------------------
        if pts["near_base"] is not None and pts["far_base"] is not None:
            nb_dx, nb_dy = self.coord_mgr.original_to_display(
                pts["near_base"][0], pts["near_base"][1], self.zoom_scale, self.pan_x, self.pan_y
            )
            fb_dx, fb_dy = self.coord_mgr.original_to_display(
                pts["far_base"][0], pts["far_base"][1], self.zoom_scale, self.pan_x, self.pan_y
            )

            # Draw prominent dashed / bright golden reference line
            cv2.line(canvas, (nb_dx, nb_dy), (fb_dx, fb_dy), (0, 215, 255), 2, cv2.LINE_AA)
            mid_x = (nb_dx + fb_dx) // 2
            mid_y = (nb_dy + fb_dy) // 2
            cv2.putText(
                canvas, "Line A-B (Longitudinal Pitch Axis)", (mid_x + 8, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA
            )

        # Draw Near Wicket upright (near_base to near_top)
        if pts["near_base"] is not None and pts["near_top"] is not None:
            nb_dx, nb_dy = self.coord_mgr.original_to_display(
                pts["near_base"][0], pts["near_base"][1], self.zoom_scale, self.pan_x, self.pan_y
            )
            nt_dx, nt_dy = self.coord_mgr.original_to_display(
                pts["near_top"][0], pts["near_top"][1], self.zoom_scale, self.pan_x, self.pan_y
            )
            cv2.line(canvas, (nb_dx, nb_dy), (nt_dx, nt_dy), (0, 200, 255), 3, cv2.LINE_AA)

        # Draw Far Wicket upright (far_base to far_top)
        if pts["far_base"] is not None and pts["far_top"] is not None:
            fb_dx, fb_dy = self.coord_mgr.original_to_display(
                pts["far_base"][0], pts["far_base"][1], self.zoom_scale, self.pan_x, self.pan_y
            )
            ft_dx, ft_dy = self.coord_mgr.original_to_display(
                pts["far_top"][0], pts["far_top"][1], self.zoom_scale, self.pan_x, self.pan_y
            )
            cv2.line(canvas, (fb_dx, fb_dy), (ft_dx, ft_dy), (255, 200, 0), 2, cv2.LINE_AA)

        # Draw Pitch Perimeter Polygon if corners selected
        pitch_keys = ["pitch_left_near", "pitch_right_near", "pitch_right_far", "pitch_left_far"]
        if all(pts[k] is not None for k in pitch_keys):
            poly_pts = []
            for k in pitch_keys:
                p_dx, p_dy = self.coord_mgr.original_to_display(
                    pts[k][0], pts[k][1], self.zoom_scale, self.pan_x, self.pan_y
                )
                poly_pts.append([p_dx, p_dy])
            poly_np = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [poly_np], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # Draw all clicked landmark markers
        for idx, key in enumerate(CALIBRATION_POINT_KEYS):
            pt = pts[key]
            if pt is not None:
                dx, dy = self.coord_mgr.original_to_display(
                    pt[0], pt[1], self.zoom_scale, self.pan_x, self.pan_y
                )
                if -50 <= dx <= self.orig_w + 50 and -50 <= dy <= self.orig_h + 50:
                    color = POINT_COLORS.get(key, (0, 255, 0))
                    # Crosshair marker
                    cv2.drawMarker(canvas, (dx, dy), color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
                    cv2.circle(canvas, (dx, dy), 6, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.circle(canvas, (dx, dy), 4, color, -1, cv2.LINE_AA)

                    # Text label with background pill
                    label = f"{idx + 1}: {POINT_SHORT_NAMES.get(key, key)}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                    cv2.rectangle(canvas, (dx + 8, dy - th - 6), (dx + 12 + tw, dy + 2), (20, 20, 20), -1)
                    cv2.putText(
                        canvas, label, (dx + 10, dy - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
                    )

        # -------------------------------------------------------------
        # HUD BARS: TOP HEADER, INSTRUCTIONS, & FOOTER
        # -------------------------------------------------------------
        cv2.rectangle(canvas, (0, 0), (self.orig_w, 72), (24, 24, 28), -1)
        cv2.line(canvas, (0, 72), (self.orig_w, 72), (60, 60, 65), 1)

        # Frame Info & Status
        frame_str = f"FRAME: {self.frame_idx}/{max(1, self.total_frames - 1)} (t={self.frame_timestamp:.3f}s) | Res: {self.orig_w}x{self.orig_h}"
        cv2.putText(canvas, frame_str, (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 180, 190), 1, cv2.LINE_AA)

        # Main Prompt
        if self.active_replace_key:
            target_desc = POINT_DESCRIPTIONS.get(self.active_replace_key, self.active_replace_key)
            prompt_str = f"REPLACING: {target_desc} - Click new location on pitch"
            cv2.putText(canvas, prompt_str, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
        elif not self.point_mgr.is_complete():
            next_k = self.point_mgr.next_required_point_key
            target_desc = POINT_DESCRIPTIONS.get(next_k, next_k)
            prompt_str = f"CLICK POINT: {target_desc}"
            cv2.putText(canvas, prompt_str, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            prompt_str = "ALL 8 POINTS COMPLETE. Review preview below. Press [C] or [Enter] to Confirm & Lock."
            cv2.putText(canvas, prompt_str, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2, cv2.LINE_AA)

        # Sub-pixel coordinate & Zoom readout (Top-right)
        cursor_str = f"X: {self.mouse_orig_x:.1f}  Y: {self.mouse_orig_y:.1f}  |  Zoom: {self.zoom_scale:.1f}x"
        (cw, _), _ = cv2.getTextSize(cursor_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(canvas, cursor_str, (self.orig_w - cw - 16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 220), 1, cv2.LINE_AA)

        # Bottom Controls Bar
        cv2.rectangle(canvas, (0, self.orig_h - 32), (self.orig_w, self.orig_h), (20, 20, 24), -1)
        help_bar = (
            "[Left/Right: Prev/Next Frame] [Wheel: Zoom] [Right-Drag: Pan] [U: Undo] [R: Reset] "
            "[1-8: Re-pick] [C: Confirm] [Q: Cancel]"
        )
        cv2.putText(canvas, help_bar, (16, self.orig_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 185, 190), 1, cv2.LINE_AA)

        # Alert banner if any error/status message exists
        if self.status_message:
            msg_color = (0, 80, 255) if self.status_error else (50, 180, 50)
            cv2.rectangle(canvas, (0, 72), (self.orig_w, 98), msg_color, -1)
            cv2.putText(
                canvas, self.status_message, (16, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )

        return canvas

    def run_interactive(self) -> bool:
        """
        Runs the interactive calibration GUI window.
        Allows zooming, panning, scrubbing frames, clicking points, undo, reset,
        re-picking individual points, and confirming calibration.
        Returns True if 8 points confirmed, False if user canceled.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        disp_w = min(1280, self.orig_w)
        disp_h = min(720, self.orig_h)
        cv2.resizeWindow(self.window_name, disp_w, disp_h)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        confirmed = False
        while True:
            display_img = self._render_display_frame()
            cv2.imshow(self.window_name, display_img)
            key = cv2.waitKey(20)

            if key == -1:
                continue

            key_char = key & 0xFF

            if key_char in (ord('q'), ord('Q'), 27): # Quit
                break
            elif key_char in (ord('u'), ord('U')): # Undo
                undone = self.point_mgr.undo()
                if undone:
                    self._set_status_msg(f"Undid point {undone}")
                    self.is_preview_mode = False
            elif key_char in (ord('r'), ord('R')): # Reset
                self.point_mgr.reset()
                self.active_replace_key = None
                self.is_preview_mode = False
                self._set_status_msg("Reset all 8 calibration points.")
            elif key_char in (ord('+'), ord('=')): # Zoom in
                self.zoom_scale = min(self.zoom_scale * 1.25, 10.0)
            elif key_char in (ord('-'), ord('_')): # Zoom out
                self.zoom_scale = max(self.zoom_scale / 1.25, 0.5)
            elif ord('1') <= key_char <= ord('8'): # Re-pick point 1-8
                idx = key_char - ord('1')
                self.active_replace_key = CALIBRATION_POINT_KEYS[idx]
                self._set_status_msg(f"Replacing landmark {idx + 1}/8: {self.active_replace_key}")
            elif key_char in (ord('c'), ord('C'), 13): # Confirm (C or Enter)
                if self.point_mgr.is_complete():
                    confirmed = True
                    break
                else:
                    self._set_status_msg("Cannot confirm: Please click all 8 points first.", error=True)
            elif key_char in (ord('n'), ord('N')) or key == 2555904 or key == 83: # Next frame (N or Right arrow)
                self.set_frame(self.frame_idx + 1)
            elif key_char in (ord('p'), ord('P')) or key == 2424832 or key == 81: # Prev frame (P or Left arrow)
                self.set_frame(self.frame_idx - 1)

        cv2.destroyWindow(self.window_name)
        if self.cap is not None:
            self.cap.release()

        return confirmed


def save_calibration_file(
    calibration_path: str,
    video_hash: str,
    video_path: str,
    frame_idx: int,
    frame_timestamp: float,
    points_dict: Dict[str, Tuple[float, float]],
    camera_model: CameraModel,
    reproj_dict: Dict[str, Any],
    geometry: WorldPitchGeometry
) -> str:
    """
    Saves locked calibration JSON file with full audit metadata.
    Enforces video identity, resolution locking, and float point preservation.
    """
    os.makedirs(os.path.dirname(os.path.abspath(calibration_path)), exist_ok=True)

    # Format points as exact float lists
    exact_points = {
        k: [float(points_dict[k][0]), float(points_dict[k][1])]
        for k in CALIBRATION_POINT_KEYS
    }

    payload = {
        "calibration_version": CALIBRATION_VERSION,
        "video_identity": video_hash,
        "video_path": os.path.abspath(video_path),
        "video_resolution": [int(camera_model.width), int(camera_model.height)],
        "selected_frame_index": int(frame_idx),
        "selected_frame_timestamp_sec": round(float(frame_timestamp), 4),
        "calibration_points_pixel": exact_points,
        "camera_parameters": camera_model.to_dict(),
        "reprojection_report": reproj_dict,
        "physical_dimensions": {
            "pitch_length_crease_to_crease_m": geometry.pitch_length_crease_to_crease_m,
            "wicket_distance_m": geometry.wicket_distance_m,
            "pitch_width_m": geometry.pitch_width_m,
            "stump_height_m": geometry.stump_height_m,
            "ball_diameter_m": geometry.ball_diameter_m,
        },
        "is_locked": True,
        "locked_at_iso": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return calibration_path


def load_calibration_file(
    calibration_path: str,
    expected_video_path: Optional[str] = None,
    expected_video_hash: Optional[str] = None,
    expected_resolution: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """
    Loads and validates locked calibration metadata from JSON.
    Detects video identity mismatches and resolution mismatches.
    Never alters manually supplied coordinates.
    """
    if not os.path.isfile(calibration_path):
        raise FileNotFoundError(f"Calibration file not found at: {calibration_path}")

    with open(calibration_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Normalize and verify schema completeness
    vid_id = data.get("video_identity") or data.get("video_hash") or data.get("entity_id")
    if vid_id:
        data["video_identity"] = vid_id

    if "video_resolution" not in data:
        w = data.get("image_width") or (data.get("camera_parameters", {}).get("image_width"))
        h = data.get("image_height") or (data.get("camera_parameters", {}).get("image_height"))
        if w is not None and h is not None:
            data["video_resolution"] = [int(w), int(h)]

    if "calibration_points_pixel" not in data and "points_dict" in data:
        data["calibration_points_pixel"] = data["points_dict"]

    required_keys = ["video_identity", "video_resolution", "calibration_points_pixel", "camera_parameters"]
    for rk in required_keys:
        if rk not in data:
            raise ValueError(f"Corrupt calibration file: missing required key '{rk}'.")

    # 2. Check resolution mismatch
    stored_w, stored_h = data["video_resolution"]
    if expected_resolution is not None:
        exp_w, exp_h = expected_resolution
        if (stored_w != exp_w) or (stored_h != exp_h):
            raise CalibrationResolutionMismatchError(
                f"Video resolution mismatch: current video is {exp_w}x{exp_h}, "
                f"but locked calibration was recorded at {stored_w}x{stored_h}."
            )

    # 3. Check video hash locking
    if expected_video_hash is not None:
        stored_hash = data.get("video_identity")
        if stored_hash and stored_hash != expected_video_hash:
            raise CalibrationLockError(
                f"Calibration hash mismatch: video hash '{expected_video_hash[:10]}...' "
                f"does not match calibration locked identity '{stored_hash[:10]}...'."
            )

    # 4. Check points exist (stump points minimum required for calibration)
    pts = data["calibration_points_pixel"]
    required_stump_keys = ["near_top", "near_base", "far_top", "far_base"]
    for pk in required_stump_keys:
        if pk not in pts or pts[pk] is None:
            raise ValueError(f"Calibration incomplete: stump point '{pk}' is missing.")

    return data


def run_manual_calibration_cli(video_path: str, output_calib_path: Optional[str] = None) -> str:
    """
    Standalone function to run 8-point manual calibration on a video and lock the results.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    import hashlib
    sha = hashlib.sha256()
    with open(video_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    video_hash = sha.hexdigest()

    ui = ManualCalibrationUI(video_path=video_path, video_hash=video_hash)
    confirmed = ui.run_interactive()
    if not confirmed:
        print("Calibration cancelled by user.")
        sys.exit(1)

    geom = ui.geometry
    calibrator = CameraCalibrator(ui.orig_w, ui.orig_h, geom)
    pts_dict = ui.point_mgr.get_points_dict()
    cam_model, reproj_report = calibrator.calibrate(pts_dict)

    validator = CalibrationValidator()
    is_valid, failures, _ = validator.validate_calibration(cam_model, reproj_report, ui.point_mgr)
    if not is_valid:
        print(f"Calibration validation failed with errors: {failures}")
        sys.exit(1)

    out_path = output_calib_path or f"storage/calibration/{video_hash}.json"
    saved = save_calibration_file(
        calibration_path=out_path,
        video_hash=video_hash,
        video_path=video_path,
        frame_idx=ui.frame_idx,
        frame_timestamp=ui.frame_timestamp,
        points_dict=pts_dict,
        camera_model=cam_model,
        reproj_dict=reproj_report.to_dict(),
        geometry=geom
    )

    print(f"\nCalibration confirmed and locked successfully!")
    print(f"Locked File:           {saved}")
    print(f"Camera Height:         {cam_model.camera_height_m:.2f} m")
    print(f"Distance Behind Stumps:{cam_model.camera_distance_behind_near_stumps_m:.2f} m")
    print(f"Mean Reprojection Err: {reproj_report.mean_error_px:.2f} px")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cricket 8-Point Manual Calibration Module")
    parser.add_argument("--video", type=str, required=True, help="Path to cricket video file")
    parser.add_argument("--output", type=str, default=None, help="Output calibration JSON path")
    args = parser.parse_args()

    run_manual_calibration_cli(args.video, args.output)
