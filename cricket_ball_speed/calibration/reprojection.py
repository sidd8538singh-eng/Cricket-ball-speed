"""
Reprojection Error Evaluator.
Computes per-point, mean, median, RMSE, and maximum reprojection errors
for the 8 calibration landmarks and assesses calibration confidence.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from .camera_model import CameraModel
from .world_coordinates import CALIBRATION_POINT_KEYS


@dataclass
class ReprojectionReport:
    per_point_errors: Dict[str, float]
    mean_error_px: float
    median_error_px: float
    max_error_px: float
    worst_point: str
    confidence_score: float # 0.0 to 1.0
    is_acceptable: bool
    rmse_px: float = 0.0
    per_point_residuals_uv: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    uncertainty: Optional[Any] = None
    summary_diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_point_errors_px": {k: round(v, 3) for k, v in self.per_point_errors.items()},
            "mean_error_px": round(self.mean_error_px, 3),
            "median_error_px": round(self.median_error_px, 3),
            "rmse_px": round(self.rmse_px, 3),
            "max_error_px": round(self.max_error_px, 3),
            "worst_point": self.worst_point,
            "confidence_score": round(self.confidence_score, 4),
            "is_acceptable": self.is_acceptable,
            "uncertainty": self.uncertainty.to_dict() if hasattr(self.uncertainty, "to_dict") else None,
            "summary_diagnostics": self.summary_diagnostics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReprojectionReport":
        return cls(
            per_point_errors=data.get("per_point_errors_px", {}),
            mean_error_px=float(data.get("mean_error_px", 0.0)),
            median_error_px=float(data.get("median_error_px", 0.0)),
            max_error_px=float(data.get("max_error_px", 0.0)),
            worst_point=data.get("worst_point", "N/A"),
            confidence_score=float(data.get("confidence_score", 1.0)),
            is_acceptable=bool(data.get("is_acceptable", True)),
            rmse_px=float(data.get("rmse_px", 0.0)),
            summary_diagnostics=data.get("summary_diagnostics", {})
        )


def compute_reprojection_report(
    camera_model: CameraModel,
    world_points_dict: Dict[str, np.ndarray],
    image_points_dict: Dict[str, Tuple[float, float]],
    max_mean_error_px: float = 5.0,
    max_per_point_error_px: float = 9.0
) -> ReprojectionReport:
    """
    Computes Euclidean reprojection error across all calibrated landmarks.
    """
    errors: Dict[str, float] = {}
    residuals: Dict[str, Tuple[float, float]] = {}

    for key in CALIBRATION_POINT_KEYS:
        if key not in world_points_dict or key not in image_points_dict:
            continue
        pt_w = world_points_dict[key]
        pt_i_true = image_points_dict[key]
        u_proj, v_proj = camera_model.project_point(pt_w)

        du = float(u_proj - pt_i_true[0])
        dv = float(v_proj - pt_i_true[1])
        err = float(np.hypot(du, dv))

        errors[key] = err
        residuals[key] = (du, dv)

    if not errors:
        raise ValueError("No matching points found to compute reprojection errors.")

    err_values = np.array(list(errors.values()), dtype=np.float64)
    mean_err = float(np.mean(err_values))
    median_err = float(np.median(err_values))
    rmse = float(np.sqrt(np.mean(err_values ** 2)))
    max_err = float(np.max(err_values))
    worst_key = max(errors, key=errors.get)

    # Confidence score: exponential decay with error
    confidence = float(np.exp(-mean_err / 2.5))
    is_acceptable = (mean_err <= max_mean_error_px) and (max_err <= max_per_point_error_px)

    return ReprojectionReport(
        per_point_errors=errors,
        mean_error_px=mean_err,
        median_error_px=median_err,
        rmse_px=rmse,
        max_error_px=max_err,
        worst_point=worst_key,
        confidence_score=confidence,
        is_acceptable=is_acceptable,
        per_point_residuals_uv=residuals,
        summary_diagnostics={
            "near_wicket_mean_err_px": round(float((errors["near_top"] + errors["near_base"]) / 2.0), 3),
            "far_wicket_mean_err_px": round(float((errors["far_top"] + errors["far_base"]) / 2.0), 3),
            "pitch_corners_mean_err_px": round(float(np.mean([
                errors["pitch_left_near"], errors["pitch_right_near"],
                errors["pitch_left_far"], errors["pitch_right_far"]
            ])), 3),
        }
    )
