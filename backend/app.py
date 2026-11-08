"""
FastAPI Server for Zoota-Style Cricket Vision DRS & Calibration Platform.
Provides high-performance REST APIs and real-time WebSockets for Live Camera and Input Video Modes.
"""

from typing import Dict, List, Optional, Any, Tuple
import os
import sys
import io
import json
import base64
import logging
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
CBS_DIR = os.path.join(ROOT_DIR, "cricket_ball_speed")
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if CBS_DIR not in sys.path:
    sys.path.insert(0, CBS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from .drs_engine import DRSCalibrationEngine, DRSLine, DetectedWicket
    from .video_service import VideoService
    from .live_service import LiveCameraService
except ImportError:
    from drs_engine import DRSCalibrationEngine, DRSLine, DetectedWicket
    from video_service import VideoService
    from live_service import LiveCameraService

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
logger = logging.getLogger("ZootaDRS.Server")

# Create FastAPI app
app = FastAPI(
    title="Zoota Cricket Vision & DRS Calibration Platform",
    description="Professional Cricket Stump Detection, 20.12m DRS Line, Ball Tracking & Bowling Speed Analysis",
    version="2.0.0"
)

# CORS configuration: configurable via CORS_ORIGINS environment variable
cors_env = os.getenv("CORS_ORIGINS", "*")
allowed_origins = [o.strip() for o in cors_env.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
@app.get("/api/health")
def health_check():
    """Liveness/readiness probe for Docker, Kubernetes, and DigitalOcean."""
    return {
        "status": "healthy",
        "service": "cricket-ball-speed",
        "version": "2.0.0"
    }

# Initialize engines and services
engine = DRSCalibrationEngine()
video_service = VideoService(upload_dir=os.path.join(ROOT_DIR, "storage", "uploads"), engine=engine)
live_service = LiveCameraService(engine=engine)

# Active calibration cache
active_calibration: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------
class CalibrateRequest(BaseModel):
    image_points: Dict[str, Optional[List[float]]]
    image_width: int
    image_height: int
    initial_focal_length_px: Optional[float] = None
    video_hash: Optional[str] = None
    video_path: Optional[str] = None


class Base64FrameRequest(BaseModel):
    image_base64: str
    frame_index: Optional[int] = 0
    confidence_threshold: Optional[float] = 0.20


class AnalyzeVideoRequest(BaseModel):
    video_path: str
    radar_speed: Optional[float] = None
    custom_calibration: Optional[Dict[str, Any]] = None


class CameraConnectRequest(BaseModel):
    device_index: int = 0
    width: int = 1280
    height: int = 720


# -----------------------------------------------------------------------------
# System Status & Model Configuration Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/status")
def get_system_status():
    """Returns system status, loaded YOLO models, physical constants, and device info."""
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    except ImportError:
        cuda_avail = False
        device_name = "CPU"

    return {
        "status": "online",
        "version": "2.0.0",
        "device": {
            "cuda_available": cuda_avail,
            "device_name": device_name,
            "inference_device": engine.manager.device
        },
        "models": {
            "stumps": {
                "loaded": engine.is_stumps_loaded,
                "classes": [0],
                "class_names": ["stumps"],
                "path": engine.stumps_model_path,
                "confidence_threshold": engine.conf_threshold
            },
            "pitch": {
                "loaded": engine.is_pitch_loaded,
                "classes": [0],
                "class_names": ["Cricket_pitch"],
                "path": engine.pitch_model_path
            },
            "ball": {
                "path": os.path.join(CBS_DIR, "models", "ball.pt"),
                "exists": os.path.isfile(os.path.join(CBS_DIR, "models", "ball.pt"))
            },
            "bowler": {
                "path": os.path.join(CBS_DIR, "models", "bowler.pt"),
                "exists": os.path.isfile(os.path.join(CBS_DIR, "models", "bowler.pt"))
            }
        },
        "physical_dimensions": {
            "wicket_distance_m": engine.geometry.wicket_distance_m,
            "pitch_width_m": engine.geometry.pitch_width_m,
            "stump_height_m": engine.geometry.stump_height_m,
            "ball_diameter_m": engine.geometry.ball_diameter_m
        },
        "active_calibration_present": active_calibration is not None
    }


@app.get("/api/models/config")
def get_models_config():
    """Retrieves current model parameters."""
    return {
        "stumps_conf": engine.conf_threshold,
        "stumps_model": engine.stumps_model_path,
        "pitch_model": engine.pitch_model_path,
        "is_stumps_loaded": engine.is_stumps_loaded,
        "is_pitch_loaded": engine.is_pitch_loaded
    }


# -----------------------------------------------------------------------------
# Live Camera Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/camera/list")
def list_cameras():
    """Discovers available video capture devices."""
    devices = LiveCameraService.list_available_cameras()
    return {"cameras": devices}


@app.post("/api/camera/connect")
def connect_camera(req: CameraConnectRequest):
    """Connects to a specific camera device."""
    ok = live_service.start_camera(
        device_index=req.device_index,
        width=req.width,
        height=req.height
    )
    if not ok:
        raise HTTPException(status_code=500, detail=f"Could not connect to camera device {req.device_index}.")
    return {
        "status": "connected",
        "device_index": req.device_index,
        "width": live_service.width,
        "height": live_service.height,
        "fps": live_service.fps
    }


@app.post("/api/camera/disconnect")
def disconnect_camera():
    """Disconnects the active camera."""
    live_service.stop_camera()
    return {"status": "disconnected"}


@app.get("/api/camera/status")
def get_camera_status():
    """Returns current live camera state and calibration telemetry."""
    return {
        "is_running": live_service.is_running,
        "device_name": live_service.device_name,
        "device_index": live_service.camera_index,
        "width": live_service.width,
        "height": live_service.height,
        "fps": live_service.fps,
        "is_calibrated": live_service.is_calibrated,
        "is_stable": live_service.is_camera_stable,
        "drift_px": live_service.camera_drift_px
    }


@app.post("/api/live/process_frame")
def process_live_frame_base64(req: Base64FrameRequest):
    """Processes a base64 encoded frame from live camera client."""
    try:
        raw_b64 = req.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 frame: {e}")

    result = live_service.process_live_frame(frame_bgr)
    return result


# -----------------------------------------------------------------------------
# Frame Detection & Calibration Endpoints
# -----------------------------------------------------------------------------
@app.post("/api/detect/frame")
def detect_single_frame(
    file: Optional[UploadFile] = File(None),
    confidence: float = Form(0.20),
    frame_index: int = Form(0)
):
    """
    Accepts an uploaded image frame, runs stump detection,
    and returns detected wickets and automatically generated DRS line.
    """
    if file is None:
        raise HTTPException(status_code=400, detail="Image file must be provided.")

    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image encoding.")

    res = engine.detect_frame(frame_bgr, frame_index=frame_index, custom_conf=confidence)
    return res.to_dict()


@app.post("/api/detect/frame_base64")
def detect_single_frame_base64(req: Base64FrameRequest):
    """Detects stumps on a base64 encoded image frame."""
    try:
        raw_b64 = req.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("cv2.imdecode returned None")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {e}")

    res = engine.detect_frame(frame_bgr, frame_index=req.frame_index or 0, custom_conf=req.confidence_threshold)
    return res.to_dict()


@app.post("/api/calibrate")
def perform_calibration(req: CalibrateRequest):
    """
    Executes camera calibration using:
      - Stump + DRS Calibration (4 points: near_base, near_top, far_base, far_top)
      - Stump + DRS + Pitch Calibration (8 points: 4 stumps + 4 pitch corners)
    """
    global active_calibration
    pts = {}
    for k, v in req.image_points.items():
        if v is not None and len(v) >= 2:
            pts[k] = (float(v[0]), float(v[1]))
        else:
            pts[k] = None

    result = engine.calibrate(
        image_points=pts,
        image_width=req.image_width,
        image_height=req.image_height,
        initial_focal_length_px=req.initial_focal_length_px,
        video_hash=req.video_hash
    )

    if result.get("success"):
        active_calibration = {
            **result,
            "image_points": req.image_points,
            "image_width": req.image_width,
            "image_height": req.image_height,
            "video_hash": req.video_hash,
            "video_path": req.video_path,
        }

    return result


@app.get("/api/calibration/current")
def get_current_calibration():
    """Retrieves the active calibration result."""
    if active_calibration is None:
        return {"calibrated": False, "message": "No active calibration recorded."}
    return {"calibrated": True, "calibration": active_calibration}


@app.post("/api/calibration/confirm")
def confirm_calibration(req: Dict[str, Any]):
    """Confirms and locks calibration for the current session or video."""
    global active_calibration
    active_calibration = req
    return {"status": "confirmed", "message": "Calibration confirmed and locked."}


@app.post("/api/calibration/reset")
def reset_calibration():
    """Clears the active calibration."""
    global active_calibration
    active_calibration = None
    return {"status": "success", "message": "Active calibration cleared."}


@app.post("/api/calibration/save")
def save_calibration_to_disk(req: Dict[str, Any]):
    """Saves the calibration JSON to storage/calibration/."""
    try:
        calib_dir = os.path.join(ROOT_DIR, "storage", "calibration")
        os.makedirs(calib_dir, exist_ok=True)
        v_hash = req.get("video_hash") or "manual_calibration"
        filename = f"{v_hash}.json"
        dest_path = os.path.join(calib_dir, filename)

        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(req, f, indent=2)

        return {"status": "saved", "path": dest_path, "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save calibration: {e}")


# -----------------------------------------------------------------------------
# Input Video Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/sample_videos")
def list_sample_videos():
    """Discovers available video files in storage directories for instant one-click testing."""
    sample_files = []
    search_dirs = [
        os.path.join(CBS_DIR, "storage", "temp"),
        os.path.join(CBS_DIR, "storage", "results"),
        os.path.join(ROOT_DIR, "storage", "uploads")
    ]

    seen_paths = set()
    for sdir in search_dirs:
        if os.path.isdir(sdir):
            for fname in os.listdir(sdir):
                if fname.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
                    fpath = os.path.join(sdir, fname)
                    if fpath in seen_paths:
                        continue
                    seen_paths.add(fpath)
                    try:
                        # Lightweight header probe without full hash to keep fast
                        cap = cv2.VideoCapture(fpath)
                        if cap.isOpened():
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
                            tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            cap.release()
                            sample_files.append({
                                "name": fname,
                                "path": fpath,
                                "size_bytes": os.path.getsize(fpath),
                                "metadata": {
                                    "file_name": fname,
                                    "file_path": fpath,
                                    "width": w,
                                    "height": h,
                                    "fps": round(fps, 1),
                                    "total_frames": tot,
                                    "duration_seconds": round(tot / fps, 2) if fps > 0 else 0.0
                                }
                            })
                    except Exception:
                        pass

    return {"sample_videos": sample_files}


@app.post("/api/video/upload")
def upload_video_file(file: UploadFile = File(...)):
    """Uploads a cricket video for Input Video Mode analysis."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Valid video file must be provided.")
    res = video_service.save_uploaded_file_stream(file)
    return res


@app.get("/api/video/metadata")
def get_video_metadata_endpoint(video_path: str = Query(...)):
    """Retrieves full metadata for a video file."""
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail="Video file not found.")
    meta = video_service.get_video_metadata(video_path)
    return meta


@app.get("/api/video/stream")
def stream_video(video_path: str = Query(...)):
    """Streams a video file for HTML5 video playback."""
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail="Video file not found.")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"}
    )


@app.get("/api/video/frame_image")
def get_video_frame_image(video_path: str = Query(...), frame_index: int = Query(0)):
    """Extracts a video frame and returns it as a JPEG image."""
    ret, frame = video_service.get_frame(video_path, frame_index)
    if not ret or frame is None:
        raise HTTPException(status_code=404, detail="Frame could not be extracted.")

    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise HTTPException(status_code=500, detail="JPEG encoding failed.")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.post("/api/video/detect_frame")
def detect_video_frame(
    video_path: str = Form(...),
    frame_index: int = Form(0),
    confidence: float = Form(0.20)
):
    """Extracts a frame from video and runs stump detection and DRS line generation."""
    ret, frame = video_service.get_frame(video_path, frame_index)
    if not ret or frame is None:
        raise HTTPException(status_code=404, detail=f"Frame {frame_index} could not be extracted.")

    res = engine.detect_frame(frame, frame_index=frame_index, custom_conf=confidence)
    return res.to_dict()


@app.get("/api/video/best_calibration_frame")
def get_best_calibration_frame(video_path: str = Query(...)):
    """Automatically selects the best frame for wicket detection & calibration (GET)."""
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail="Video file not found.")
    res = video_service.auto_select_best_calibration_frame(video_path)
    return res


@app.post("/api/video/best_calibration_frame")
def post_best_calibration_frame(req: Dict[str, Any]):
    """Automatically selects the best frame for wicket detection & calibration (POST)."""
    video_path = req.get("video_path")
    if not video_path or not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail="Video file not found or invalid path.")
    res = video_service.auto_select_best_calibration_frame(video_path)
    return res


@app.post("/api/video/process_multi_frame")
def process_multi_frame_video(
    video_path: str = Form(...),
    sample_step: int = Form(2),
    max_samples: int = Form(40),
    confidence: float = Form(0.20)
):
    """Multi-frame video analysis across sampled frames."""
    res = video_service.process_multi_frame_video(
        video_path=video_path,
        sample_step=sample_step,
        max_samples=max_samples,
        conf_threshold=confidence
    )
    return res


# -----------------------------------------------------------------------------
# End-to-End Delivery Analysis Endpoints
# -----------------------------------------------------------------------------
@app.post("/api/video/analyze")
def start_video_delivery_analysis(req: AnalyzeVideoRequest):
    """
    Launches full delivery speed analysis in background.
    Returns task_id for progress polling.
    """
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=404, detail=f"Video file not found at: {req.video_path}")

    # Use custom calibration if supplied, else active_calibration
    calib_to_use = req.custom_calibration or active_calibration

    task_id = video_service.start_video_analysis(
        video_path=req.video_path,
        custom_calibration=calib_to_use,
        radar_speed=req.radar_speed
    )
    return {
        "status": "started",
        "task_id": task_id,
        "message": "Analysis started in background task."
    }


@app.get("/api/video/analysis_status/{task_id}")
def get_analysis_status(task_id: str):
    """Returns current analysis percentage and stage description."""
    status = video_service.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task ID not found.")
    return status


@app.get("/api/video/analysis_result/{task_id}")
def get_analysis_result(task_id: str):
    """Returns final speed, release speed, C1-C2 speed, QC audit, and output paths."""
    res = video_service.get_task_result(task_id)
    if not res:
        status = video_service.get_task_status(task_id)
        if status and status["status"] == "processing":
            return {"status": "processing", "progress_pct": status["progress_pct"]}
        elif status and status["status"] == "failed":
            raise HTTPException(status_code=500, detail=status.get("error", "Analysis failed."))
        raise HTTPException(status_code=404, detail="Analysis results not ready or task not found.")
    return res


@app.get("/api/video/download/{artifact_type}/{task_id}")
def download_analysis_artifact(artifact_type: str, task_id: str):
    """
    Downloads analysis outputs:
      - 'final': Final processed video MP4
      - 'diagnostic': Diagnostic video MP4
      - 'json': JSON audit report
      - 'excel': Excel XLSX summary report
    """
    res = video_service.get_task_result(task_id)
    if not res:
        raise HTTPException(status_code=404, detail="Results not ready.")

    key_map = {
        "final": ("final_video", "video/mp4", "final_delivery.mp4"),
        "diagnostic": ("diagnostic_video", "video/mp4", "diagnostic_delivery.mp4"),
        "json": ("json_report", "application/json", "delivery_report.json"),
        "excel": ("excel_report", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "delivery_summary.xlsx")
    }

    if artifact_type not in key_map:
        raise HTTPException(status_code=400, detail=f"Invalid artifact type '{artifact_type}'. Must be one of {list(key_map.keys())}.")

    dict_key, media_type, dl_name = key_map[artifact_type]
    file_path = res.get(dict_key)

    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"Artifact file {artifact_type} not found at {file_path}.")

    return FileResponse(
        file_path,
        media_type=media_type,
        filename=dl_name
    )


# -----------------------------------------------------------------------------
# WebSocket for Live Camera Streaming & Speed Dashboard
# -----------------------------------------------------------------------------
@app.websocket("/ws/live")
async def websocket_live_camera(websocket: WebSocket):
    """
    High-speed WebSocket connection for real-time live camera frame streaming.
    Receives JPEG/WebP frames, returns detected wickets, DRS line, and telemetry.
    """
    await websocket.accept()
    logger.info("Live Camera WebSocket connected.")
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                # Binary frame payload
                nparr = np.frombuffer(message["bytes"], np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    telemetry = live_service.process_live_frame(frame)
                    await websocket.send_json({
                        "type": "telemetry",
                        "data": telemetry
                    })

            elif "text" in message and message["text"]:
                payload = json.loads(message["text"])
                msg_type = payload.get("type", "frame")

                if msg_type == "frame":
                    raw_b64 = payload.get("image", "")
                    if "," in raw_b64:
                        raw_b64 = raw_b64.split(",")[1]
                    img_bytes = base64.b64decode(raw_b64)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        telemetry = live_service.process_live_frame(frame)
                        await websocket.send_json({
                            "type": "telemetry",
                            "data": telemetry
                        })

                elif msg_type == "calibrate":
                    pts = {}
                    for k, v in payload.get("image_points", {}).items():
                        if v is not None and len(v) >= 2:
                            pts[k] = (float(v[0]), float(v[1]))
                    calib_res = engine.calibrate(
                        image_points=pts,
                        image_width=int(payload.get("image_width", 1280)),
                        image_height=int(payload.get("image_height", 720))
                    )
                    await websocket.send_json({
                        "type": "calibration_result",
                        "data": calib_res
                    })

    except WebSocketDisconnect:
        logger.info("Live Camera WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Live Camera WebSocket error: {e}")


# -----------------------------------------------------------------------------
# Static Files & SPA Mounting
# -----------------------------------------------------------------------------
STATIC_DIR = os.path.join(ROOT_DIR, "frontend", "static")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    """Serves the main single page application."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"message": "Zoota DRS Vision API Online. Frontend initializing."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False)
