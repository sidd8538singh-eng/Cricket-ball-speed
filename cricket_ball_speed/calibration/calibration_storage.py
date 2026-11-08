"""
Calibration Persistence and Management.
Stores and retrieves video-specific and camera-specific calibrations.
Guarantees calibration isolation by video SHA-256 hash and camera identity.
"""

from typing import Dict, Any, Optional
import os
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger("CricketBallSpeed.CalibStorage")


@dataclass
class StoredCalibration:
    """Encapsulates a persistent locked camera calibration record."""
    entity_id: str                      # Video SHA-256 hash or Camera ID
    entity_type: str                    # "video" or "camera"
    calibration_version: str            # e.g. "2.0.0"
    created_at: str
    image_width: int
    image_height: int
    calibration_method: str             # "automatic_stumps_drs", "automatic_stumps_pitch", "manual_8point"
    points_dict: Dict[str, Any]
    drs_line: Optional[Dict[str, Any]]
    pitch_corners: Optional[Dict[str, Any]]
    camera_parameters: Dict[str, Any]
    reprojection_report: Dict[str, Any]
    physical_dimensions: Dict[str, float]
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "calibration_version": self.calibration_version,
            "created_at": self.created_at,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "calibration_method": self.calibration_method,
            "points_dict": self.points_dict,
            "drs_line": self.drs_line,
            "pitch_corners": self.pitch_corners,
            "camera_parameters": self.camera_parameters,
            "reprojection_report": self.reprojection_report,
            "physical_dimensions": self.physical_dimensions,
            "is_valid": self.is_valid
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredCalibration":
        return cls(
            entity_id=data["entity_id"],
            entity_type=data.get("entity_type", "video"),
            calibration_version=data.get("calibration_version", "2.0.0"),
            created_at=data.get("created_at", ""),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            calibration_method=data.get("calibration_method", "automatic_stumps_drs"),
            points_dict=data.get("points_dict", {}),
            drs_line=data.get("drs_line"),
            pitch_corners=data.get("pitch_corners"),
            camera_parameters=data.get("camera_parameters", {}),
            reprojection_report=data.get("reprojection_report", {}),
            physical_dimensions=data.get("physical_dimensions", {}),
            is_valid=bool(data.get("is_valid", True))
        )


class CalibrationStorage:
    """
    Manages file-based storage of calibration files in storage/calibration/.
    Prevents cross-video calibration contamination.
    """

    def __init__(self, storage_dir: str = "storage/calibration"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def get_video_calibration_path(self, video_hash: str) -> str:
        safe_hash = "".join(c for c in video_hash if c.isalnum() or c in ("-", "_"))
        return os.path.join(self.storage_dir, f"{safe_hash}.json")

    def get_camera_calibration_path(self, camera_id: str) -> str:
        safe_cam = "".join(c for c in camera_id if c.isalnum() or c in ("-", "_"))
        return os.path.join(self.storage_dir, f"camera_{safe_cam}.json")

    def save_calibration(self, record: StoredCalibration) -> str:
        """Saves a calibration record to disk."""
        if record.entity_type == "camera":
            dest_path = self.get_camera_calibration_path(record.entity_id)
        else:
            dest_path = self.get_video_calibration_path(record.entity_id)

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2)

        logger.info(f"Saved locked calibration to {dest_path}")
        return dest_path

    def load_calibration_for_video(self, video_hash: str) -> Optional[StoredCalibration]:
        """Loads existing calibration if video hash matches exactly."""
        path = self.get_video_calibration_path(video_hash)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return StoredCalibration.from_dict(data)
        except Exception as e:
            logger.warning(f"Error loading calibration from {path}: {e}")
            return None

    def load_calibration_for_camera(self, camera_id: str) -> Optional[StoredCalibration]:
        """Loads existing calibration for a specific camera device."""
        path = self.get_camera_calibration_path(camera_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return StoredCalibration.from_dict(data)
        except Exception as e:
            logger.warning(f"Error loading camera calibration from {path}: {e}")
            return None

    def invalidate_camera_calibration(self, camera_id: str) -> bool:
        """Deletes or invalidates calibration when camera moves or vibrates excessively."""
        path = self.get_camera_calibration_path(camera_id)
        if os.path.isfile(path):
            try:
                os.remove(path)
                logger.info(f"Invalidated camera calibration for {camera_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to invalidate camera calibration: {e}")
        return False
