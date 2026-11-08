"""
Video Processing Service for Input Video Mode.
Handles video upload, metadata extraction, frame retrieval,
multi-frame stump detection, outlier filtering, automatic calibration frame selection,
and background execution of the complete CricketSpeedPipeline.
"""

from typing import Dict, List, Optional, Tuple, Any
import os
import sys
import uuid
import threading
import hashlib
import logging
import cv2
import numpy as np

try:
    from .drs_engine import DRSCalibrationEngine, FrameDetectionResult, DRSLine, DetectedWicket
except ImportError:
    from drs_engine import DRSCalibrationEngine, FrameDetectionResult, DRSLine, DetectedWicket

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
CBS_DIR = os.path.join(ROOT_DIR, "cricket_ball_speed")
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if CBS_DIR not in sys.path:
    sys.path.insert(0, CBS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from video.metadata import extract_video_metadata, VideoMetadata

try:
    from cricket_ball_speed.main import CricketSpeedPipeline
except ImportError:
    from main import CricketSpeedPipeline

logger = logging.getLogger("ZootaDRS.VideoService")


class VideoService:
    """Manages video storage, playback frame retrieval, and multi-frame processing."""

    def __init__(self, upload_dir: str = "storage/uploads", engine: Optional[DRSCalibrationEngine] = None):
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)
        self.task_dir = os.path.join(ROOT_DIR, "storage", "tasks")
        os.makedirs(self.task_dir, exist_ok=True)
        self.engine = engine or DRSCalibrationEngine()
        self._video_cache: Dict[str, cv2.VideoCapture] = {}

        # Background analysis tasks: task_id -> task status & results
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _save_task_to_disk(self, task_id: str, data: Dict[str, Any]):
        """Persists task state to disk for cross-worker process availability."""
        try:
            path = os.path.join(self.task_dir, f"{task_id}.json")
            temp_path = f"{path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, default=str)
            os.replace(temp_path, path)
        except Exception as e:
            logger.debug(f"Task {task_id} persistence to disk notice: {e}")

    def _load_task_from_disk(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Loads task state from disk if not present in memory."""
        path = os.path.join(self.task_dir, f"{task_id}.json")
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.debug(f"Task {task_id} disk read notice: {e}")
        return None

    def save_uploaded_file_stream(self, upload_file) -> Dict[str, Any]:
        """Streams uploaded video in chunks directly to disk to minimize memory footprint."""
        safe_name = os.path.basename(getattr(upload_file, "filename", "video.mp4") or "video.mp4")
        dest_path = os.path.join(self.upload_dir, safe_name)

        with open(dest_path, "wb") as f:
            while chunk := upload_file.file.read(1024 * 1024):
                f.write(chunk)

        metadata = self.get_video_metadata(dest_path)
        return {
            "file_name": safe_name,
            "file_path": dest_path,
            "metadata": metadata
        }

    def save_uploaded_video(self, filename: str, content: bytes) -> Dict[str, Any]:
        """Saves uploaded video file and extracts metadata."""
        safe_name = os.path.basename(filename)
        dest_path = os.path.join(self.upload_dir, safe_name)

        with open(dest_path, "wb") as f:
            f.write(content)

        metadata = self.get_video_metadata(dest_path)
        return {
            "file_name": safe_name,
            "file_path": dest_path,
            "metadata": metadata
        }

    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extracts complete metadata using OpenCV and hashlib."""
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = float(total_frames / fps) if fps > 0 else 0.0

        # Compute SHA256
        sha = hashlib.sha256()
        with open(video_path, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        file_hash = sha.hexdigest()
        cap.release()

        return {
            "file_name": os.path.basename(video_path),
            "file_path": video_path,
            "file_hash": file_hash,
            "width": w,
            "height": h,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_seconds": round(duration, 2)
        }

    def get_frame(self, video_path: str, frame_index: int) -> Tuple[bool, Optional[np.ndarray]]:
        """Extracts a specific frame from the video as BGR numpy array."""
        if not os.path.isfile(video_path):
            return False, None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        return ret, frame

    def auto_select_best_calibration_frame(
        self,
        video_path: str,
        max_samples: int = 15
    ) -> Dict[str, Any]:
        """
        Scans frames across the beginning of the video and selects the optimal
        frame for camera calibration (where both stumps and pitch are detected with highest clarity).
        """
        meta = self.get_video_metadata(video_path)
        total_f = meta["total_frames"]

        # Sample frames across first 20% of video or first 45 frames
        search_limit = min(total_f, 60)
        indices = np.linspace(0, max(0, search_limit - 1), min(search_limit, max_samples), dtype=int).tolist()

        best_score = -1.0
        best_frame_idx = 0
        best_detection: Optional[FrameDetectionResult] = None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "selected_frame_index": 0,
                "best_frame_index": 0,
                "timestamp_sec": 0.0,
                "quality_score": 0.0,
                "detection": None,
                "total_frames_checked": 0
            }

        try:
            for f_idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                det = self.engine.detect_frame(frame, frame_index=f_idx)
                score = 0.0
                if det.near_wicket and det.far_wicket:
                    score += 50.0
                    score += (det.near_wicket.confidence + det.far_wicket.confidence) * 15.0
                if det.drs_line and det.drs_line.is_valid:
                    score += 30.0
                if det.pitch_bbox:
                    score += 20.0

                if score > best_score:
                    best_score = score
                    best_frame_idx = f_idx
                    best_detection = det
        finally:
            cap.release()

        return {
            "selected_frame_index": best_frame_idx,
            "best_frame_index": best_frame_idx,
            "timestamp_sec": round(best_frame_idx / meta["fps"], 3) if meta["fps"] > 0 else 0.0,
            "quality_score": round(best_score, 1),
            "detection": best_detection.to_dict() if best_detection else None,
            "total_frames_checked": len(indices)
        }

    def process_multi_frame_video(
        self,
        video_path: str,
        sample_step: int = 2,
        max_samples: int = 40,
        conf_threshold: float = 0.20
    ) -> Dict[str, Any]:
        """
        Processes multiple frames across the video:
          1. Detects stumps on sampled frames.
          2. Rejects outliers and false positives.
          3. Computes stable near and far wicket geometry using median filtering.
          4. Constructs the multi-frame stabilized DRS line.
          5. Executes multi-frame calibration.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

        if total_frames <= max_samples:
            frame_indices = list(range(0, total_frames, max(1, sample_step)))
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_samples, dtype=int).tolist()

        near_bases = []
        near_tops = []
        far_bases = []
        far_tops = []
        per_frame_results = []

        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            res = self.engine.detect_frame(frame, frame_index=f_idx, custom_conf=conf_threshold)
            res_dict = res.to_dict()
            per_frame_results.append(res_dict)

            if res.near_wicket:
                near_bases.append(res.near_wicket.base_center)
                near_tops.append(res.near_wicket.top_center)
            if res.far_wicket:
                far_bases.append(res.far_wicket.base_center)
                far_tops.append(res.far_wicket.top_center)

        cap.release()

        # Robust median coordinates
        has_stable_near = len(near_bases) >= 1
        has_stable_far = len(far_bases) >= 1

        stable_points: Dict[str, Optional[Tuple[float, float]]] = {
            "near_base": None,
            "near_top": None,
            "far_base": None,
            "far_top": None,
        }

        if has_stable_near:
            nb_arr = np.array(near_bases)
            nt_arr = np.array(near_tops)
            stable_points["near_base"] = (float(np.median(nb_arr[:, 0])), float(np.median(nb_arr[:, 1])))
            stable_points["near_top"] = (float(np.median(nt_arr[:, 0])), float(np.median(nt_arr[:, 1])))

        if has_stable_far:
            fb_arr = np.array(far_bases)
            ft_arr = np.array(far_tops)
            stable_points["far_base"] = (float(np.median(fb_arr[:, 0])), float(np.median(fb_arr[:, 1])))
            stable_points["far_top"] = (float(np.median(ft_arr[:, 0])), float(np.median(ft_arr[:, 1])))

        # Construct Stable DRS line
        stable_drs_line = None
        if has_stable_near and has_stable_far:
            p_near = stable_points["near_base"]
            p_far = stable_points["far_base"]
            stable_drs_line = DRSLineValidator.validate(
                near_base=p_near,
                far_base=p_far,
                image_width=w,
                image_height=h
            )

        calibration_result = None
        if has_stable_near and has_stable_far and stable_drs_line and stable_drs_line.is_valid:
            valid_pts = {k: v for k, v in stable_points.items() if v is not None}
            calibration_result = self.engine.calibrate(
                image_points=valid_pts,
                image_width=w,
                image_height=h
            )

        return {
            "video_path": video_path,
            "total_frames_sampled": len(frame_indices),
            "near_detections_count": len(near_bases),
            "far_detections_count": len(far_bases),
            "stable_points": {k: [round(c, 2) for c in v] if v else None for k, v in stable_points.items()},
            "stable_drs_line": stable_drs_line.to_dict() if stable_drs_line else None,
            "per_frame_results": per_frame_results,
            "calibration": calibration_result,
            "summary_status": (
                "CALIBRATION_SUCCESSFUL" if calibration_result and calibration_result.get("success")
                else ("DRS_LINE_GENERATED" if stable_drs_line else "INSUFFICIENT_DETECTIONS")
            )
        }

    def start_video_analysis(
        self,
        video_path: str,
        custom_calibration: Optional[Dict[str, Any]] = None,
        radar_speed: Optional[float] = None
    ) -> str:
        """
        Launches full video delivery analysis in a background thread.
        Returns a unique task_id for tracking progress.
        """
        task_id = str(uuid.uuid4())
        initial_state = {
            "task_id": task_id,
            "video_path": video_path,
            "status": "processing",
            "progress_pct": 0.0,
            "stage": "Initializing bowling analysis engine...",
            "result": None,
            "error": None
        }
        with self._lock:
            self.tasks[task_id] = initial_state
        self._save_task_to_disk(task_id, initial_state)

        def worker():
            try:
                def on_progress(pct: float, stage: str):
                    with self._lock:
                        if task_id in self.tasks:
                            self.tasks[task_id]["progress_pct"] = round(pct, 1)
                            self.tasks[task_id]["stage"] = stage
                            self._save_task_to_disk(task_id, self.tasks[task_id])

                pipeline = CricketSpeedPipeline(
                    config_dir=os.path.join(CBS_DIR, "config")
                )
                output_dir = os.path.join(ROOT_DIR, "storage", "results")

                res = pipeline.process_delivery(
                    video_path=video_path,
                    custom_calibration_data=custom_calibration,
                    headless=True,
                    radar_speed=radar_speed,
                    output_dir=output_dir,
                    generate_videos=True,
                    progress_callback=on_progress
                )

                with self._lock:
                    if task_id in self.tasks:
                        self.tasks[task_id]["status"] = "completed"
                        self.tasks[task_id]["progress_pct"] = 100.0
                        self.tasks[task_id]["stage"] = "Analysis finished successfully."
                        self.tasks[task_id]["result"] = res
                        self._save_task_to_disk(task_id, self.tasks[task_id])

            except Exception as e:
                logger.error(f"Error in video analysis worker: {e}", exc_info=True)
                with self._lock:
                    if task_id in self.tasks:
                        self.tasks[task_id]["status"] = "failed"
                        self.tasks[task_id]["stage"] = f"Analysis error: {str(e)}"
                        self.tasks[task_id]["error"] = str(e)
                        self._save_task_to_disk(task_id, self.tasks[task_id])

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self.tasks.get(task_id)
        if not task:
            task = self._load_task_from_disk(task_id)
        if not task:
            return None
        return {
            "task_id": task["task_id"],
            "status": task["status"],
            "progress_pct": task["progress_pct"],
            "stage": task["stage"],
            "error": task.get("error")
        }

    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self.tasks.get(task_id)
        if not task:
            task = self._load_task_from_disk(task_id)
        if not task:
            return None
        return task.get("result")
