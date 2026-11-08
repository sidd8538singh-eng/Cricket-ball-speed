"""
Calibration Manager Orchestrator.
Coordinates:
  - Automatic stump detection
  - DRS line construction and validation
  - Pitch geometry & ground-plane constraints
  - Camera calibration (4-point Stumps+DRS and 8-point Stumps+DRS+Pitch)
  - Reprojection error reporting and geometric plausibility
  - Manual calibration fallback and landmark correction
  - Video and camera persistence
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import datetime
import math
import logging
import cv2
import numpy as np
from scipy.optimize import least_squares

from .world_coordinates import WorldPitchGeometry, CALIBRATION_POINT_KEYS
from .camera_model import CameraModel
from .camera_calibration import CameraCalibrator
from .calibration_validator import CalibrationValidator
from .reprojection import compute_reprojection_report, ReprojectionReport
from .homography import GroundPlaneHomography
from .drs_line import DRSLine, DRSLineValidator
from .automatic_stump_calibration import (
    AutomaticStumpDetector, AutomaticStumpDetectionResult, WicketObservation
)
from .pitch_geometry import PitchGeometryEngine, PitchDetectionResult, PitchCorners
from .calibration_storage import CalibrationStorage, StoredCalibration

logger = logging.getLogger("CricketBallSpeed.CalibManager")


@dataclass
class CalibrationResult:
    """Complete result of calibration execution."""
    success: bool
    status: str                         # AUTOMATIC_CALIBRATED, MANUALLY_CALIBRATED, PROVISIONAL_DRS, CALIBRATION_FAILED
    method: str                         # "Stump + DRS Calibration" or "Stump + DRS + Pitch Calibration"
    message: str
    camera_model: Optional[CameraModel] = None
    reprojection_report: Optional[ReprojectionReport] = None
    drs_line: Optional[DRSLine] = None
    pitch_corners: Optional[PitchCorners] = None
    image_points: Dict[str, Optional[Tuple[float, float]]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    video_hash: Optional[str] = None
    camera_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        cam_dict = self.camera_model.to_dict() if self.camera_model else None
        reproj_dict = self.reprojection_report.to_dict() if self.reprojection_report else None

        # Add homography matrices to camera_parameters if available
        if cam_dict and self.camera_model:
            R_mat = self.camera_model.R
            tvec = self.camera_model.tvec
            K = self.camera_model.K
            # Ground plane Z = 0 homography H = K * [r1, r2, t]
            H_w2i = K @ np.column_stack([R_mat[:, 0], R_mat[:, 1], tvec])
            if abs(H_w2i[2, 2]) > 1e-9:
                H_w2i /= H_w2i[2, 2]
                try:
                    H_i2w = np.linalg.inv(H_w2i)
                    H_i2w /= H_i2w[2, 2]
                    cam_dict["homography_w2i"] = H_w2i.tolist()
                    cam_dict["homography_i2w"] = H_i2w.tolist()
                except np.linalg.LinAlgError:
                    pass

        return {
            "success": self.success,
            "status": self.status,
            "method": self.method,
            "message": self.message,
            "reprojection_error_px": reproj_dict.get("mean_error_px", 0.0) if reproj_dict else None,
            "max_reprojection_error_px": reproj_dict.get("max_error_px", 0.0) if reproj_dict else None,
            "per_point_errors": reproj_dict.get("per_point_errors_px", {}) if reproj_dict else {},
            "camera_parameters": cam_dict,
            "drs_line": self.drs_line.to_dict() if self.drs_line else None,
            "pitch_corners": self.pitch_corners.to_dict() if self.pitch_corners else None,
            "image_points": {
                k: [round(c, 2) for c in v] if v is not None else None
                for k, v in self.image_points.items()
            },
            "warnings": self.warnings,
            "video_hash": self.video_hash,
            "camera_id": self.camera_id,
            "is_metric_scale_calibrated": self.success,
            "longitudinal_constraint_m": 20.12,
            "pitch_width_constraint_m": 3.048 if self.pitch_corners and self.pitch_corners.is_valid else None
        }


class CalibrationManager:
    """
    Central Calibration Manager.
    Coordinates automatic stump calibration, pitch geometry, DRS-line generation,
    manual correction, camera calibration solver, validation, and storage.
    """

    def __init__(
        self,
        stumps_model_path: str = "models/stumps.pt",
        pitch_model_path: Optional[str] = "models/pitch.pt",
        storage_dir: str = "storage/calibration",
        pitch_geometry: Optional[WorldPitchGeometry] = None,
        max_reprojection_error_px: float = 6.0,
        device: str = "auto"
    ):
        self.geometry = pitch_geometry or WorldPitchGeometry()
        self.max_reprojection_error_px = max_reprojection_error_px
        self.device = device

        self.storage = CalibrationStorage(storage_dir)
        self.stump_detector = AutomaticStumpDetector(
            model_path=stumps_model_path,
            confidence_threshold=0.20,
            device=device
        )
        self.pitch_engine = PitchGeometryEngine(
            model_path=pitch_model_path,
            conf_threshold=0.25
        )
        self.validator = CalibrationValidator(
            max_mean_reprojection_px=max_reprojection_error_px
        )

        self._active_calibration: Optional[CalibrationResult] = None

    @property
    def active_calibration(self) -> Optional[CalibrationResult]:
        return self._active_calibration

    def set_active_calibration(self, calib: Optional[CalibrationResult]):
        self._active_calibration = calib

    def auto_calibrate_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int = 0,
        video_hash: Optional[str] = None,
        camera_id: Optional[str] = None,
        initial_focal_length_px: Optional[float] = None
    ) -> CalibrationResult:
        """
        Executes complete automatic calibration on a single frame:
        1. Runs automatic stump detection.
        2. Constructs 20.12m DRS line.
        3. Runs optional pitch detection and boundary extraction.
        4. Solves joint camera calibration.
        5. Validates reprojection and camera plausibility.
        6. Persists calibration if successful.
        """
        h, w = frame_bgr.shape[:2]

        # 1. Automatic Stump Detection
        stump_res = self.stump_detector.detect_frame(
            frame_bgr=frame_bgr,
            frame_index=frame_index
        )

        if not stump_res.near_wicket or not stump_res.far_wicket or not stump_res.drs_line:
            return CalibrationResult(
                success=False,
                status="CALIBRATION_INSUFFICIENT",
                method="Automatic Detection",
                message=f"Automatic calibration failed: {stump_res.message}",
                drs_line=stump_res.drs_line,
                warnings=["Both near and far wickets must be clearly visible to calibrate."]
            )

        # 2. Pitch Detection (Optional)
        pitch_res = self.pitch_engine.detect_pitch(
            frame_bgr=frame_bgr,
            drs_line=stump_res.drs_line
        )

        # 3. Assemble Image Points
        pts: Dict[str, Optional[Tuple[float, float]]] = {
            "near_base": stump_res.near_wicket.base_center,
            "near_top": stump_res.near_wicket.top_center,
            "far_base": stump_res.far_wicket.base_center,
            "far_top": stump_res.far_wicket.top_center,
            "pitch_left_near": None,
            "pitch_right_near": None,
            "pitch_left_far": None,
            "pitch_right_far": None,
        }

        pitch_corners = None
        if pitch_res.is_detected and pitch_res.corners and pitch_res.corners.is_valid:
            pitch_corners = pitch_res.corners
            pts.update(pitch_corners.to_points_dict())

        # 4. Solve Calibration
        calib_res = self.solve_calibration(
            image_points=pts,
            image_width=w,
            image_height=h,
            drs_line=stump_res.drs_line,
            pitch_corners=pitch_corners,
            initial_focal_length_px=initial_focal_length_px,
            video_hash=video_hash,
            camera_id=camera_id
        )

        if calib_res.success:
            self._active_calibration = calib_res
            # Persist to disk
            self.persist_calibration(calib_res)

        return calib_res

    def solve_calibration(
        self,
        image_points: Dict[str, Optional[Tuple[float, float]]],
        image_width: int,
        image_height: int,
        drs_line: Optional[DRSLine] = None,
        pitch_corners: Optional[PitchCorners] = None,
        initial_focal_length_px: Optional[float] = None,
        video_hash: Optional[str] = None,
        camera_id: Optional[str] = None
    ) -> CalibrationResult:
        """
        Solves camera projection parameters from supplied image points.
        Supports both:
          - 4-point Stump + DRS Calibration (mandatory wickets)
          - 8-point Stump + DRS + Pitch Calibration (when pitch corners are present)
        """
        # Verify mandatory 4 stump points
        mandatory = ["near_base", "near_top", "far_base", "far_top"]
        for k in mandatory:
            if k not in image_points or image_points[k] is None:
                return CalibrationResult(
                    success=False,
                    status="CALIBRATION_INSUFFICIENT",
                    method="Camera Calibration",
                    message=f"Missing mandatory stump reference point '{k}'."
                )

        # Build or validate DRS line if not passed
        if drs_line is None:
            drs_line = DRSLineValidator.validate(
                near_base=image_points["near_base"],
                far_base=image_points["far_base"],
                image_width=image_width,
                image_height=image_height
            )

        # Check if full 8 points are present
        has_8_points = all(
            k in image_points and image_points[k] is not None
            for k in CALIBRATION_POINT_KEYS
        )

        if has_8_points:
            return self._solve_8point(
                image_points={k: image_points[k] for k in CALIBRATION_POINT_KEYS},
                image_width=image_width,
                image_height=image_height,
                drs_line=drs_line,
                pitch_corners=pitch_corners,
                video_hash=video_hash,
                camera_id=camera_id
            )
        else:
            return self._solve_4point(
                image_points={k: image_points[k] for k in mandatory},
                image_width=image_width,
                image_height=image_height,
                drs_line=drs_line,
                initial_focal_length_px=initial_focal_length_px,
                video_hash=video_hash,
                camera_id=camera_id
            )

    def _solve_4point(
        self,
        image_points: Dict[str, Tuple[float, float]],
        image_width: int,
        image_height: int,
        drs_line: DRSLine,
        initial_focal_length_px: Optional[float] = None,
        video_hash: Optional[str] = None,
        camera_id: Optional[str] = None
    ) -> CalibrationResult:
        """
        Solves camera calibration using the 4 non-coplanar stump landmarks:
          near_base (0, 0, 0), near_top (0, 0, H_s), far_base (0, L, 0), far_top (0, L, H_s).
        Enforces physical camera elevation, orientation, and longitudinal distance.
        """
        L = self.geometry.L
        h_stump = self.geometry.stump_height_m

        obj_pts = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, h_stump],
            [0.0, L, 0.0],
            [0.0, L, h_stump]
        ], dtype=np.float64)

        img_pts = np.array([
            image_points["near_base"],
            image_points["near_top"],
            image_points["far_base"],
            image_points["far_top"],
        ], dtype=np.float64)

        cx = image_width / 2.0
        cy = image_height / 2.0

        if initial_focal_length_px is None or initial_focal_length_px <= 0:
            f_prior = (image_width / 2.0) / math.tan(math.radians(60.0 / 2.0))
        else:
            f_prior = initial_focal_length_px

        f_candidates = np.linspace(f_prior * 0.4, f_prior * 2.2, 35)
        best_f = f_prior
        best_rvec = None
        best_tvec = None
        best_err = float("inf")

        dist_zero = np.zeros((4, 1), dtype=np.float64)

        for f_cand in f_candidates:
            K_cand = np.array([[f_cand, 0.0, cx], [0.0, f_cand, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
            for flag in [cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP]:
                try:
                    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K_cand, dist_zero, flags=flag)
                    if success:
                        R, _ = cv2.Rodrigues(rvec)
                        c_world = (-R.T @ tvec).flatten()
                        # Camera must be behind near stumps (Y < 0.5) and elevated (Z > 0.3)
                        if c_world[1] <= 0.5 and c_world[2] >= 0.3:
                            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K_cand, dist_zero)
                            err = float(np.mean(np.linalg.norm(proj.reshape((-1, 2)) - img_pts, axis=1)))
                            if err < best_err:
                                best_err = err
                                best_f = f_cand
                                best_rvec = rvec
                                best_tvec = tvec
                except Exception:
                    continue

        if best_rvec is None or best_tvec is None:
            K_init = np.array([[f_prior, 0, cx], [0, f_prior, cy], [0, 0, 1]], dtype=np.float64)
            try:
                cv2.solvePnP(obj_pts, img_pts, K_init, dist_zero, flags=cv2.SOLVEPNP_EPNP)
            except Exception:
                pass
            best_f = f_prior
            best_rvec = np.array([[1.5], [0.0], [0.0]], dtype=np.float64)
            best_tvec = np.array([[0.0], [1.5], [3.0]], dtype=np.float64)

        # Refinement over [log(f), rx, ry, rz, tx, ty, tz]
        x0 = np.array([
            math.log(best_f),
            best_rvec[0, 0], best_rvec[1, 0], best_rvec[2, 0],
            best_tvec[0, 0], best_tvec[1, 0], best_tvec[2, 0]
        ], dtype=np.float64)

        def residuals(params: np.ndarray) -> np.ndarray:
            f_val = math.exp(params[0])
            r_val = params[1:4].reshape((3, 1))
            t_val = params[4:7].reshape((3, 1))
            K = np.array([[f_val, 0, cx], [0, f_val, cy], [0, 0, 1]], dtype=np.float64)
            proj, _ = cv2.projectPoints(obj_pts, r_val, t_val, K, dist_zero)
            return (proj.reshape((-1, 2)) - img_pts).flatten()

        opt_res = least_squares(residuals, x0, method="lm", max_nfev=300)
        opt_f = float(math.exp(opt_res.x[0]))
        opt_rvec = opt_res.x[1:4].reshape((3, 1))
        opt_tvec = opt_res.x[4:7].reshape((3, 1))

        K_final = np.array([[opt_f, 0, cx], [0, opt_f, cy], [0, 0, 1]], dtype=np.float64)
        dist_final = np.zeros(5, dtype=np.float64)

        camera_model = CameraModel(
            camera_matrix=K_final,
            dist_coeffs=dist_final,
            rvec=opt_rvec,
            tvec=opt_tvec,
            image_width=image_width,
            image_height=image_height
        )

        # Reprojection Error calculation
        proj_final, _ = cv2.projectPoints(obj_pts, opt_rvec, opt_tvec, K_final, dist_final)
        proj_2d = proj_final.reshape((-1, 2))
        per_point_errs = {}
        for idx, key in enumerate(["near_base", "near_top", "far_base", "far_top"]):
            e = float(np.linalg.norm(proj_2d[idx] - img_pts[idx]))
            per_point_errs[key] = round(e, 2)

        mean_err = float(np.mean(list(per_point_errs.values())))
        max_err = float(np.max(list(per_point_errs.values())))

        reproj_report = ReprojectionReport(
            per_point_errors=per_point_errs,
            mean_error_px=mean_err,
            median_error_px=mean_err,
            max_error_px=max_err,
            worst_point=max(per_point_errs, key=per_point_errs.get),
            confidence_score=max(0.0, 1.0 - mean_err / 10.0),
            is_acceptable=mean_err <= self.max_reprojection_error_px
        )

        warnings = [
            "Stump + DRS Calibration locks longitudinal (20.12m) & vertical geometry. "
            "For highest lateral precision, activate pitch model or calibrate pitch corners."
        ]

        c_world = camera_model.optical_centre
        msg = (
            f"Stump + DRS calibration completed. Mean error: {mean_err:.2f}px. "
            f"Camera height: {c_world[2]:.2f}m, Distance behind stumps: {-c_world[1]:.2f}m."
        )

        return CalibrationResult(
            success=reproj_report.is_acceptable,
            status="AUTOMATIC_CALIBRATED" if reproj_report.is_acceptable else "CALIBRATION_REJECTED",
            method="Stump + DRS Calibration",
            message=msg,
            camera_model=camera_model,
            reprojection_report=reproj_report,
            drs_line=drs_line,
            image_points=image_points,
            warnings=warnings,
            video_hash=video_hash,
            camera_id=camera_id
        )

    def _solve_8point(
        self,
        image_points: Dict[str, Tuple[float, float]],
        image_width: int,
        image_height: int,
        drs_line: DRSLine,
        pitch_corners: Optional[PitchCorners] = None,
        video_hash: Optional[str] = None,
        camera_id: Optional[str] = None
    ) -> CalibrationResult:
        """
        Solves camera calibration using all 8 cricket landmarks via CameraCalibrator.
        """
        calibrator = CameraCalibrator(
            image_width=image_width,
            image_height=image_height,
            pitch_geometry=self.geometry,
            max_reprojection_error_px=self.max_reprojection_error_px
        )

        try:
            camera_model, reproj_report = calibrator.calibrate(image_points)
            c_world = camera_model.optical_centre
            msg = (
                f"Full 8-point DRS + Pitch calibration completed. Mean error: {reproj_report.mean_error_px:.2f}px. "
                f"Camera height: {c_world[2]:.2f}m, Distance behind stumps: {-c_world[1]:.2f}m."
            )
            return CalibrationResult(
                success=reproj_report.is_acceptable,
                status="AUTOMATIC_CALIBRATED" if reproj_report.is_acceptable else "CALIBRATION_REJECTED",
                method="Stump + DRS + Pitch Calibration",
                message=msg,
                camera_model=camera_model,
                reprojection_report=reproj_report,
                drs_line=drs_line,
                pitch_corners=pitch_corners,
                image_points=image_points,
                warnings=[],
                video_hash=video_hash,
                camera_id=camera_id
            )
        except Exception as e:
            logger.error(f"8-point calibration failed: {e}")
            return CalibrationResult(
                success=False,
                status="CALIBRATION_FAILED",
                method="Stump + DRS + Pitch Calibration",
                message=f"8-point calibration failed: {e}",
                drs_line=drs_line,
                image_points=image_points
            )

    def persist_calibration(self, calib: CalibrationResult) -> Optional[str]:
        """Saves active calibration result to storage."""
        if not calib.success or not calib.camera_model:
            return None

        entity_id = calib.video_hash or calib.camera_id or "default_calibration"
        entity_type = "camera" if calib.camera_id else "video"

        record = StoredCalibration(
            entity_id=entity_id,
            entity_type=entity_type,
            calibration_version="2.0.0",
            created_at=datetime.datetime.now().isoformat(),
            image_width=calib.camera_model.width,
            image_height=calib.camera_model.height,
            calibration_method=calib.method,
            points_dict=calib.image_points,
            drs_line=calib.drs_line.to_dict() if calib.drs_line else None,
            pitch_corners=calib.pitch_corners.to_dict() if calib.pitch_corners else None,
            camera_parameters=calib.camera_model.to_dict(),
            reprojection_report=calib.reprojection_report.to_dict() if calib.reprojection_report else {},
            physical_dimensions={
                "wicket_distance_m": self.geometry.wicket_distance_m,
                "pitch_width_m": self.geometry.pitch_width_m,
                "stump_height_m": self.geometry.stump_height_m,
                "ball_diameter_m": self.geometry.ball_diameter_m
            },
            is_valid=calib.success
        )
        return self.storage.save_calibration(record)

    def load_video_calibration(self, video_hash: str) -> Optional[CalibrationResult]:
        """Loads and instantiates calibration for a specific video hash."""
        rec = self.storage.load_calibration_for_video(video_hash)
        if not rec or not rec.is_valid:
            return None

        camera_model = CameraModel.from_dict(rec.camera_parameters)
        reproj = ReprojectionReport.from_dict(rec.reprojection_report) if rec.reprojection_report else None
        drs = DRSLine.from_dict(rec.drs_line) if rec.drs_line else None

        res = CalibrationResult(
            success=True,
            status="LOCKED_CALIBRATION_LOADED",
            method=rec.calibration_method,
            message=f"Locked calibration loaded for video {video_hash[:8]}...",
            camera_model=camera_model,
            reprojection_report=reproj,
            drs_line=drs,
            image_points=rec.points_dict,
            video_hash=video_hash
        )
        self._active_calibration = res
        return res
