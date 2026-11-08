"""
Diagnostic Video Generator.
Renders comprehensive computer vision engineering telemetry:
  - Raw ball detections, accepted detections, and rejected candidates with explicit rejection reasons.
  - Bowling release point C1 and down-pitch baseline point C2.
  - Connected OBSERVED trajectory trail (strictly connecting verified real detections).
  - PREDICTED positions rendered in a visibly distinct style (never mixed with real detections).
  - 3D fitted ballistic trajectory reprojected onto the video ground frame.
  - High-information Telemetry HUD with quality gates, residuals, speeds, and rejection reasons.

DECOUPLING PRINCIPLE:
Operates cleanly on either domain dataclasses or plain dictionary structures.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import os
import cv2
import numpy as np


class DiagnosticVideoRenderer:
    """
    Renders diagnostic overlays, trajectory trails, and telemetry HUD onto frames.
    """

    @staticmethod
    def draw_dashed_line(
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 1,
        dash_len: int = 6
    ):
        """Draws a dashed line between two points."""
        dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
        if dist < 1e-3:
            return
        n_dashes = int(dist / dash_len)
        for i in range(0, n_dashes, 2):
            start_ratio = i / max(1, n_dashes)
            end_ratio = min(1.0, (i + 1) / max(1, n_dashes))
            p_start = (
                int(round(pt1[0] + (pt2[0] - pt1[0]) * start_ratio)),
                int(round(pt1[1] + (pt2[1] - pt1[1]) * start_ratio)),
            )
            p_end = (
                int(round(pt1[0] + (pt2[0] - pt1[0]) * end_ratio)),
                int(round(pt1[1] + (pt2[1] - pt1[1]) * end_ratio)),
            )
            cv2.line(img, p_start, p_end, color, thickness, cv2.LINE_AA)

    @classmethod
    def render_frame_diagnostics(
        cls,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        frame_raw_dets: List[Any],
        observed_centers: List[Tuple[int, int]],
        predicted_centers: List[Tuple[int, int]],
        release_event: Optional[Any],
        c1_c2_result: Optional[Any],
        fitted_curve_pts: List[Tuple[int, int]],
        hud_info: Dict[str, Any]
    ) -> np.ndarray:
        """Renders all diagnostic features onto a single frame."""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # ---------------------------------------------------------------------
        # 1. Raw, Accepted, and Rejected Ball Detections
        # ---------------------------------------------------------------------
        for det in frame_raw_dets:
            # Handle both object attributes and dict keys
            bbox = getattr(det, "bbox", None) or (det.get("bbox") if isinstance(det, dict) else None)
            if bbox is None:
                continue
            bx1, by1, bx2, by2 = [int(round(c)) for c in bbox]
            status = getattr(det, "acceptance_status", None) or (det.get("acceptance_status") if isinstance(det, dict) else "ACCEPTED")
            conf = getattr(det, "confidence", 0.0) if not isinstance(det, dict) else det.get("confidence", 0.0)
            diam = getattr(det, "apparent_diameter_px", 0.0) if not isinstance(det, dict) else det.get("apparent_diameter_px", 0.0)
            reason = getattr(det, "rejection_reason", None) or (det.get("rejection_reason") if isinstance(det, dict) else None)

            if status == "ACCEPTED":
                # Solid Green bounding box for verified ball detection
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (46, 204, 113), 2)
                lbl = f"Ball: {conf:.2f} (d={diam:.1f}px)"
                cv2.putText(overlay, lbl, (bx1, max(14, by1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (46, 204, 113), 1, cv2.LINE_AA)
            else:
                # Solid Red bounding box and X marker for rejected candidate
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 230), 1)
                cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                cv2.drawMarker(overlay, (cx, cy), (0, 0, 230), cv2.MARKER_TILTED_CROSS, 8, 1)
                rej_lbl = f"REJ: {reason[:18]}" if reason else "REJECTED"
                cv2.putText(overlay, rej_lbl, (bx1, max(14, by1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 230), 1, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # 2. Observed Trajectory Trail (STRICT: Real Detections ONLY)
        # ---------------------------------------------------------------------
        if len(observed_centers) > 1:
            for i in range(len(observed_centers) - 1):
                cv2.line(overlay, observed_centers[i], observed_centers[i + 1], (0, 230, 255), 2, cv2.LINE_AA)
        for pt in observed_centers:
            cv2.circle(overlay, pt, 4, (46, 204, 113), -1, cv2.LINE_AA)  # Solid green dot

        # ---------------------------------------------------------------------
        # 3. Predicted Positions (VISIBLY DIFFERENT STYLE: Never mixed with real)
        # ---------------------------------------------------------------------
        for pt in predicted_centers:
            # Rendered as hollow yellow-orange diamond with "PRED" label
            cv2.drawMarker(overlay, pt, (0, 165, 255), cv2.MARKER_DIAMOND, 10, 1, cv2.LINE_AA)
            cv2.putText(overlay, "PRED", (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 165, 255), 1, cv2.LINE_AA)

        if len(predicted_centers) > 1:
            for i in range(len(predicted_centers) - 1):
                cls.draw_dashed_line(overlay, predicted_centers[i], predicted_centers[i + 1], (0, 165, 255), 1, 5)

        # ---------------------------------------------------------------------
        # 4. Release Point C1
        # ---------------------------------------------------------------------
        if release_event is not None:
            c1_px = getattr(release_event, "c1_pixel", None) or (release_event.get("c1_pixel") if isinstance(release_event, dict) else None)
            rel_frame = getattr(release_event, "frame_index", -1) if not isinstance(release_event, dict) else release_event.get("frame_index", -1)
            mode = getattr(release_event, "observation_mode", "Direct") if not isinstance(release_event, dict) else release_event.get("observation_mode", "Direct")
            if c1_px is not None and frame_idx >= rel_frame:
                c1_u, c1_v = int(round(c1_px[0])), int(round(c1_px[1]))
                cv2.drawMarker(overlay, (c1_u, c1_v), (255, 0, 255), cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)
                cv2.putText(overlay, f"C1: Release ({mode[:6]})", (c1_u + 10, c1_v + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 0, 255), 1, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # 5. C2 Reference Point (Baseline)
        # ---------------------------------------------------------------------
        if c1_c2_result is not None:
            c2_frame = getattr(c1_c2_result, "c2_frame_index", 0) if not isinstance(c1_c2_result, dict) else c1_c2_result.get("c2_frame_index", 0)
            c2_dist = getattr(c1_c2_result, "distance_3d_m", 0.0) if not isinstance(c1_c2_result, dict) else c1_c2_result.get("distance_3d_m", 0.0)
            c2_px = hud_info.get("c2_projected_pixel")
            if c2_px is not None and frame_idx >= c2_frame:
                c2_u, c2_v = int(round(c2_px[0])), int(round(c2_px[1]))
                cv2.drawMarker(overlay, (c2_u, c2_v), (255, 255, 0), cv2.MARKER_SQUARE, 14, 2, cv2.LINE_AA)
                cv2.putText(overlay, f"C2: Baseline ({c2_dist:.1f}m)", (c2_u + 10, c2_v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 0), 1, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # 6. Fitted 3D Ballistic Trajectory Curve
        # ---------------------------------------------------------------------
        if len(fitted_curve_pts) > 1:
            for i in range(len(fitted_curve_pts) - 1):
                cv2.line(overlay, fitted_curve_pts[i], fitted_curve_pts[i + 1], (180, 50, 240), 2, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # 7. Diagnostic Telemetry HUD (Upper-Right Glass Panel)
        # ---------------------------------------------------------------------
        hud_w, hud_h = 420, 210
        hx1, hy1 = w - hud_w - 20, 20
        hx2, hy2 = w - 20, 20 + hud_h

        # Semi-transparent dark background
        hud_bg = overlay.copy()
        cv2.rectangle(hud_bg, (hx1, hy1), (hx2, hy2), (18, 22, 28), -1)
        cv2.addWeighted(hud_bg, 0.85, overlay, 0.15, 0, overlay)
        cv2.rectangle(overlay, (hx1, hy1), (hx2, hy2), (70, 78, 86), 1)

        # Status & Quality Score
        status_str = hud_info.get("status", "NOT_MEASURED")
        score = hud_info.get("quality_score", 0.0)
        status_col = (46, 204, 113) if status_str == "MEASURED" else (60, 60, 230)
        cv2.putText(overlay, f"STATUS: {status_str}  (QC Score: {score:.0f}/100)", (hx1 + 16, hy1 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.52, status_col, 1, cv2.LINE_AA)

        # Speeds
        rel_spd_str = hud_info.get("release_speed_str", "N/A")
        c1c2_spd_str = hud_info.get("c1_c2_speed_str", "N/A")
        cv2.putText(overlay, f"Release Speed: {rel_spd_str}", (hx1 + 16, hy1 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"C1-C2 Baseline: {c1c2_spd_str}", (hx1 + 16, hy1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 220, 230), 1, cv2.LINE_AA)

        # Trajectory Fit Quality
        rmse_str = hud_info.get("rmse_str", "N/A")
        unc_str = hud_info.get("uncertainty_str", "N/A")
        cv2.putText(overlay, f"Traj RMSE: {rmse_str} | Speed Unc: {unc_str}", (hx1 + 16, hy1 + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 200), 1, cv2.LINE_AA)

        # Frame telemetry
        fps = hud_info.get("fps", 60.0)
        cv2.putText(overlay, f"Frame: {frame_idx:04d} | PTS: {timestamp_sec:.4f}s | FPS: {fps:.1f}", (hx1 + 16, hy1 + 128), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 180), 1, cv2.LINE_AA)

        # Observations count
        det_cnt = hud_info.get("detected_count", 0)
        pred_cnt = hud_info.get("predicted_count", 0)
        cv2.putText(overlay, f"Observed Balls: {det_cnt} | Kalman Preds: {pred_cnt}", (hx1 + 16, hy1 + 152), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 180), 1, cv2.LINE_AA)

        # Rejection reason if unmeasured
        if status_str != "MEASURED":
            rej = hud_info.get("rejection_reason", "QC Gates Failed")
            cv2.putText(overlay, f"REJ: {rej[:40]}", (hx1 + 16, hy1 + 184), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 140, 255), 1, cv2.LINE_AA)

        return overlay


def generate_diagnostic_video(
    input_video_path: str,
    output_video_path: str,
    camera_model: Optional[Any],
    raw_observations: Optional[Dict[int, List[Any]]] = None,
    tracked_points: Optional[List[Any]] = None,
    release_event: Optional[Any] = None,
    c1_c2_result: Optional[Any] = None,
    trajectory_result: Optional[Any] = None,
    release_speed: Optional[Any] = None,
    qc_result: Optional[Any] = None,
    fps: float = 60.0
) -> str:
    """
    Renders the diagnostic engineering video.
    Completely decoupled from model/inference engines; operates on recorded points or dicts.
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

    raw_obs_map = raw_observations or {}
    points_list = tracked_points or []

    # Pre-index tracked points by frame
    track_by_frame: Dict[int, List[Any]] = {}
    for tp in points_list:
        f_idx = getattr(tp, "frame_index", None) if not isinstance(tp, dict) else tp.get("frame_index")
        if f_idx is not None:
            track_by_frame.setdefault(f_idx, []).append(tp)

    # Pre-calculate projected 3D fitted trajectory points
    fitted_curve_pts: List[Tuple[int, int]] = []
    if trajectory_result is not None and camera_model is not None:
        pts_3d = getattr(trajectory_result, "fitted_3d_points", None) or (trajectory_result.get("fitted_3d_points") if isinstance(trajectory_result, dict) else None)
        if pts_3d:
            for pt in pts_3d:
                pos = getattr(pt, "position_xyz", None) if not isinstance(pt, dict) else pt.get("position_xyz")
                if pos is not None and hasattr(camera_model, "project_point"):
                    try:
                        u_p, v_p = camera_model.project_point(np.array(pos, dtype=np.float64))
                        fitted_curve_pts.append((int(round(u_p)), int(round(v_p))))
                    except Exception:
                        pass

    # Pre-calculate C2 projected pixel
    c2_projected_px = None
    if c1_c2_result is not None and camera_model is not None:
        c2_pos = getattr(c1_c2_result, "c2_pos_m", None) or (c1_c2_result.get("c2_pos_m") if isinstance(c1_c2_result, dict) else None)
        if c2_pos is not None and hasattr(camera_model, "project_point"):
            try:
                c2_u, c2_v = camera_model.project_point(np.array(c2_pos, dtype=np.float64))
                c2_projected_px = (float(c2_u), float(c2_v))
            except Exception:
                pass

    # Extract HUD metadata
    status_str = "MEASURED"
    quality_score = 100.0
    rej_reason = None
    if qc_result is not None:
        if isinstance(qc_result, dict):
            status_str = qc_result.get("status", "NOT_MEASURED")
            quality_score = float(qc_result.get("quality_score", 0.0))
            rej_reason = qc_result.get("rejection_reason")
        else:
            status_str = getattr(qc_result, "status", "NOT_MEASURED")
            quality_score = float(getattr(qc_result, "quality_score", 0.0))
            rej_reason = getattr(qc_result, "rejection_reason", None)

    rel_spd_str = "N/A"
    unc_str = "N/A"
    if release_speed is not None and status_str == "MEASURED":
        s_val = getattr(release_speed, "speed_km_h", None) if not isinstance(release_speed, dict) else release_speed.get("speed_km_h")
        u_val = getattr(release_speed, "uncertainty_km_h", None) if not isinstance(release_speed, dict) else release_speed.get("uncertainty_km_h")
        if s_val is not None:
            rel_spd_str = f"{s_val:.1f} km/h"
        if u_val is not None:
            unc_str = f"+-{u_val:.1f} km/h"

    c1c2_spd_str = "N/A"
    if c1_c2_result is not None:
        c1c2_val = getattr(c1_c2_result, "average_speed_km_h", None) if not isinstance(c1_c2_result, dict) else c1_c2_result.get("average_speed_km_h")
        if c1c2_val is not None:
            c1c2_spd_str = f"{c1c2_val:.1f} km/h"

    rmse_str = "N/A"
    if trajectory_result is not None:
        rmse_val = getattr(trajectory_result, "rmse_pixels", None) if not isinstance(trajectory_result, dict) else trajectory_result.get("rmse_pixels")
        if rmse_val is not None:
            rmse_str = f"{rmse_val:.2f} px"

    hud_info = {
        "status": status_str,
        "quality_score": quality_score,
        "release_speed_str": rel_spd_str,
        "c1_c2_speed_str": c1c2_spd_str,
        "rmse_str": rmse_str,
        "uncertainty_str": unc_str,
        "fps": out_fps,
        "rejection_reason": rej_reason,
        "c2_projected_pixel": c2_projected_px,
    }

    observed_history: List[Tuple[int, int]] = []
    predicted_history: List[Tuple[int, int]] = []

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Update tracked history for this frame
            if frame_idx in track_by_frame:
                for tp in track_by_frame[frame_idx]:
                    u = getattr(tp, "u", None) if not isinstance(tp, dict) else tp.get("u")
                    v = getattr(tp, "v", None) if not isinstance(tp, dict) else tp.get("v")
                    is_real = getattr(tp, "is_real_detection", True) if not isinstance(tp, dict) else tp.get("is_real_detection", True)
                    if u is not None and v is not None:
                        pt_xy = (int(round(u)), int(round(v)))
                        if is_real:
                            observed_history.append(pt_xy)
                        else:
                            predicted_history.append(pt_xy)

            hud_info["detected_count"] = len(observed_history)
            hud_info["predicted_count"] = len(predicted_history)

            ts_sec = frame_idx / out_fps
            frame_dets = raw_obs_map.get(frame_idx, [])

            rendered = DiagnosticVideoRenderer.render_frame_diagnostics(
                frame=frame,
                frame_idx=frame_idx,
                timestamp_sec=ts_sec,
                frame_raw_dets=frame_dets,
                observed_centers=observed_history,
                predicted_centers=predicted_history,
                release_event=release_event,
                c1_c2_result=c1_c2_result,
                fitted_curve_pts=fitted_curve_pts,
                hud_info=hud_info
            )
            writer.write(rendered)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return output_video_path
