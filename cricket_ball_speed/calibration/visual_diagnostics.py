"""
Visual Calibration Diagnostics and Numerical Audit Reporter.
Generates diagnostic visualizations showing reprojected 3D pitch grids,
reprojection residual error vectors, Line A-B reference, wicket rods,
and complete camera telemetry HUD.
"""

from typing import Dict, Tuple, Any, Optional
import os
import json
import cv2
import numpy as np

from .camera_model import CameraModel
from .reprojection import ReprojectionReport
from .world_coordinates import WorldPitchGeometry, CALIBRATION_POINT_KEYS


def render_calibration_diagnostic_image(
    frame_bgr: np.ndarray,
    camera_model: CameraModel,
    reprojection_report: ReprojectionReport,
    measured_points: Dict[str, Tuple[float, float]],
    geometry: WorldPitchGeometry,
    residual_scale: float = 8.0,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Renders diagnostic visualization overlaid on the calibration frame:
      - 3D Pitch ground grid at 2m intervals
      - Line A-B (longitudinal pitch reference)
      - Measured landmarks vs 3D reprojected projections
      - Magnified residual vectors (error direction and magnitude)
      - Calibration Telemetry HUD
    """
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]

    # 1. Draw 3D Ground Pitch Grid (Z = 0)
    L = geometry.wicket_distance_m
    half_w = geometry.pitch_width_m / 2.0

    # Longitudinal grid lines (X = -half_w, 0, +half_w)
    for x_m in [-half_w, 0.0, half_w]:
        grid_pts = []
        for y_m in np.linspace(0.0, L, 25):
            u, v = camera_model.project_point(np.array([x_m, y_m, 0.0]))
            grid_pts.append((int(round(u)), int(round(v))))
        for i in range(len(grid_pts) - 1):
            color = (0, 215, 255) if abs(x_m) < 1e-3 else (0, 180, 0)
            thickness = 2 if abs(x_m) < 1e-3 else 1
            cv2.line(canvas, grid_pts[i], grid_pts[i + 1], color, thickness, cv2.LINE_AA)

    # Lateral grid lines (every 2m along Y from 0 to L)
    for y_m in np.arange(0.0, L + 0.1, 2.0):
        u1, v1 = camera_model.project_point(np.array([-half_w, y_m, 0.0]))
        u2, v2 = camera_model.project_point(np.array([+half_w, y_m, 0.0]))
        color = (255, 255, 255) if (abs(y_m) < 0.1 or abs(y_m - L) < 0.1) else (0, 140, 0)
        thickness = 2 if (abs(y_m) < 0.1 or abs(y_m - L) < 0.1) else 1
        cv2.line(canvas, (int(round(u1)), int(round(v1))), (int(round(u2)), int(round(v2))), color, thickness, cv2.LINE_AA)

    # 2. Draw Wicket Rods (Z = 0 to 0.711m)
    # Near Wicket
    nu_b, nv_b = camera_model.project_point(np.array([0.0, 0.0, 0.0]))
    nu_t, nv_t = camera_model.project_point(np.array([0.0, 0.0, geometry.stump_height_m]))
    cv2.line(canvas, (int(round(nu_b)), int(round(nv_b))), (int(round(nu_t)), int(round(nv_t))), (0, 200, 255), 3, cv2.LINE_AA)

    # Far Wicket
    fu_b, fv_b = camera_model.project_point(np.array([0.0, L, 0.0]))
    fu_t, fv_t = camera_model.project_point(np.array([0.0, L, geometry.stump_height_m]))
    cv2.line(canvas, (int(round(fu_b)), int(round(fv_b))), (int(round(fu_t)), int(round(fv_t))), (255, 200, 0), 2, cv2.LINE_AA)

    # 3. Draw Measured Points vs Reprojected Points & Error Residuals
    world_dict = geometry.get_world_coordinates()
    for key in CALIBRATION_POINT_KEYS:
        if key not in measured_points:
            continue
        meas_u, meas_v = measured_points[key]
        proj_u, proj_v = camera_model.project_point(world_dict[key])

        mu, mv = int(round(meas_u)), int(round(meas_v))
        pu, pv = int(round(proj_u)), int(round(proj_v))

        # Measured: Green crosshair
        cv2.drawMarker(canvas, (mu, mv), (0, 255, 0), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

        # Reprojected: Red circle
        cv2.circle(canvas, (pu, pv), 5, (0, 0, 255), 2, cv2.LINE_AA)

        # Magnified error residual vector (yellow arrow)
        du = (proj_u - meas_u) * residual_scale
        dv = (proj_v - meas_v) * residual_scale
        arrow_tip = (int(round(meas_u + du)), int(round(meas_v + dv)))
        cv2.arrowedLine(canvas, (mu, mv), arrow_tip, (0, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)

        # Point label and error px
        err_px = reprojection_report.per_point_errors.get(key, 0.0)
        label = f"{key} ({err_px:.1f}px)"
        cv2.putText(canvas, label, (mu + 8, mv - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Telemetry HUD (Top-Left Card)
    hud_w, hud_h = 420, 200
    overlay = canvas.copy()
    cv2.rectangle(overlay, (16, 16), (16 + hud_w, 16 + hud_h), (20, 20, 24), -1)
    cv2.rectangle(overlay, (16, 16), (16 + hud_w, 16 + hud_h), (70, 70, 75), 1)
    cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)

    status_color = (0, 255, 0) if reprojection_report.is_acceptable else (0, 0, 255)
    cv2.putText(canvas, "CAMERA CALIBRATION DIAGNOSTICS", (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"STATUS: {'VALID' if reprojection_report.is_acceptable else 'REJECTED'}", (310, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)
    cv2.line(canvas, (24, 48), (16 + hud_w - 8, 48), (60, 60, 65), 1)

    lines = [
        f"Focal Length f:    {camera_model.fx:.1f} px  (FOV: {camera_model.fov_x_deg:.1f}° x {camera_model.fov_y_deg:.1f}°)",
        f"Camera Height Z:    {camera_model.camera_height_m:.2f} m  (ICC setup: 2.0m)",
        f"Dist Behind Stumps: {camera_model.camera_distance_behind_near_stumps_m:.2f} m  (ICC setup: 1.5 - 2.5m)",
        f"Lateral Offset X:   {camera_model.camera_lateral_offset_m:.2f} m",
        f"Mean Reproj Error:  {reprojection_report.mean_error_px:.2f} px  (Max: {reprojection_report.max_error_px:.2f} px)",
        f"RMSE Reproj Error:  {reprojection_report.rmse_px:.2f} px",
        f"Quality Confidence: {reprojection_report.confidence_score * 100:.1f} / 100",
    ]

    for idx, text in enumerate(lines):
        cv2.putText(canvas, text, (28, 70 + idx * 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 220), 1, cv2.LINE_AA)

    # Save to disk if requested
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, canvas)

    return canvas


def generate_calibration_numerical_report(
    camera_model: CameraModel,
    reprojection_report: ReprojectionReport,
    geometry: WorldPitchGeometry,
    uncertainty_report: Optional[Any] = None
) -> Dict[str, Any]:
    """Generates structured numerical report suitable for JSON export or logging."""
    return {
        "camera_parameters": camera_model.to_dict(),
        "reprojection_report": reprojection_report.to_dict(),
        "physical_geometry": {
            "pitch_length_crease_to_crease_m": geometry.pitch_length_crease_to_crease_m,
            "wicket_distance_m": geometry.wicket_distance_m,
            "pitch_width_m": geometry.pitch_width_m,
            "stump_height_m": geometry.stump_height_m,
            "ball_diameter_m": geometry.ball_diameter_m,
            "near_wicket_offset_from_crease_m": geometry.near_wicket_offset_from_crease_m,
            "far_wicket_offset_from_crease_m": geometry.far_wicket_offset_from_crease_m,
        },
        "uncertainty_report": uncertainty_report.to_dict() if uncertainty_report else None,
    }
