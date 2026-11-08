"""
JSON Report Generator.
Serializes the complete delivery audit into a comprehensive, structured JSON document.

SAVES:
- Calibration: Intrinsics, extrinsics, distortion, optical center, reprojection metrics, and 8-point coordinates.
- Detections: Raw, accepted, and rejected candidate coordinates, confidences, apparent diameters, and reasons.
- Timestamps: Nonuniform timestamps, frame indices, and delta times.
- Release: Frame, timestamp, C1 image pixel, bowler hand position, separation metrics, and confidence.
- Reconstructed 3D Positions: World coordinates (X, Y, Z) in metres with uncertainties and provenance.
- Uncertainty: Depth observability, 3D covariance, speed uncertainty, and 95% confidence intervals.
- Trajectory: Ballistic parameter vector [X0, Y0, Z0, VX0, VY0, VZ0], RMSE pixels, and convergence.
- Speed: Instantaneous 3D release speed, C1-C2 baseline average, and frame-wise telemetry.
- Quality Control: 10 verification gates, pass/fail status, composite score, and rejection reasons.
"""

from typing import Dict, Any, Optional, List, Union
import os
import json
import numpy as np


def _to_serializable(val: Any) -> Any:
    """Recursively converts objects, dataclasses, and numpy arrays into JSON-serializable primitives."""
    if val is None:
        return None
    if isinstance(val, (int, float, str, bool)):
        return val
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, (list, tuple)):
        return [_to_serializable(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _to_serializable(v) for k, v in val.items()}
    if hasattr(val, "to_dict") and callable(getattr(val, "to_dict")):
        return _to_serializable(val.to_dict())
    if hasattr(val, "__dict__"):
        return _to_serializable(val.__dict__)
    return str(val)


def generate_json_report(
    output_path: str,
    metadata: Optional[Any] = None,
    camera_model: Optional[Any] = None,
    reprojection_report: Optional[Any] = None,
    release_event: Optional[Any] = None,
    trajectory_result: Optional[Any] = None,
    release_speed: Optional[Any] = None,
    c1_c2_result: Optional[Any] = None,
    qc_result: Optional[Any] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    framewise_report: Optional[Any] = None,
    raw_observations: Optional[Any] = None,
    reconstructed_3d_points: Optional[Any] = None
) -> str:
    """
    Serializes complete delivery analysis to a structured JSON audit document.
    Independent of internal classes; accepts dataclasses or plain dictionaries.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 1. Metadata
    meta_dict = _to_serializable(metadata) or {}

    # 2. Calibration
    calib_dict: Dict[str, Any] = {}
    if camera_model is not None:
        calib_dict["camera_parameters"] = _to_serializable(camera_model)
    if reprojection_report is not None:
        calib_dict["reprojection_report"] = _to_serializable(reprojection_report)

    # 3. Quality Control
    qc_dict = _to_serializable(qc_result) or {}

    # 4. Release Event
    rel_dict = _to_serializable(release_event)

    # 5. Trajectory Reconstruction
    traj_dict = _to_serializable(trajectory_result)

    # 6. Speed Measurements
    speed_meas: Dict[str, Any] = {
        "release_speed": _to_serializable(release_speed),
        "c1_c2_speed": _to_serializable(c1_c2_result),
        "framewise_speed": _to_serializable(framewise_report),
    }

    # 7. Uncertainty package
    uncertainty_dict: Dict[str, Any] = {}
    if release_speed is not None:
        u_speed = getattr(release_speed, "uncertainty_km_h", None)
        if u_speed is None and isinstance(release_speed, dict):
            u_speed = release_speed.get("uncertainty_km_h")
        ci_lower = getattr(release_speed, "ci_95_lower_km_h", None)
        if ci_lower is None and isinstance(release_speed, dict):
            ci_lower = release_speed.get("ci_95_lower_km_h")
        ci_upper = getattr(release_speed, "ci_95_upper_km_h", None)
        if ci_upper is None and isinstance(release_speed, dict):
            ci_upper = release_speed.get("ci_95_upper_km_h")

        uncertainty_dict["speed_uncertainty_kmh"] = u_speed
        if ci_lower is not None and ci_upper is not None:
            uncertainty_dict["confidence_interval_95_kmh"] = [ci_lower, ci_upper]

    if trajectory_result is not None:
        cov = getattr(trajectory_result, "covariance_matrix", None)
        if cov is None and isinstance(trajectory_result, dict):
            cov = trajectory_result.get("covariance_matrix")
        obs_score = getattr(trajectory_result, "depth_observability_score", None)
        if obs_score is None and isinstance(trajectory_result, dict):
            obs_score = trajectory_result.get("depth_observability_score")

        if cov is not None:
            uncertainty_dict["covariance_matrix"] = _to_serializable(cov)
        if obs_score is not None:
            try:
                uncertainty_dict["depth_observability_score"] = float(obs_score)
            except (ValueError, TypeError):
                uncertainty_dict["depth_observability_score"] = None

    # 8. Reconstructed 3D Points
    pts_3d_serializable = None
    if reconstructed_3d_points is not None:
        pts_3d_serializable = _to_serializable(reconstructed_3d_points)
    elif trajectory_result is not None:
        fitted_pts = getattr(trajectory_result, "fitted_3d_points", None)
        if fitted_pts is None and isinstance(trajectory_result, dict):
            fitted_pts = trajectory_result.get("fitted_3d_points")
        if fitted_pts is not None:
            pts_3d_serializable = _to_serializable(fitted_pts)

    # 9. Raw Observations & Candidate Detections
    obs_serializable = _to_serializable(raw_observations) if raw_observations is not None else None

    report = {
        "video_metadata": meta_dict,
        "quality_control": qc_dict,
        "speed_measurement": speed_meas,
        "release_event": rel_dict,
        "trajectory_reconstruction": traj_dict,
        "reconstructed_positions_3d": pts_3d_serializable,
        "detections": obs_serializable,
        "uncertainty": uncertainty_dict,
        "calibration": calib_dict,
        "extra_info": _to_serializable(extra_info) or {},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return output_path
