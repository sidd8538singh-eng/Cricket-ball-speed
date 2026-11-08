# PRO CORPORATE LEAGUE: Cricket Bowling Speed Analyzer
## Production-Grade Cricket Ball Speed Measurement System (Zoota Style)

A production-oriented computer vision and camera calibration system for calculating 3D cricket bowling velocity and release speed in km/h from fixed broadcast camera footage or live camera feeds behind the bowler.

---

## 1. System Overview

Inspired by Zoota's cricket analytics technology, this application provides high-precision speed estimation using:
1. **Automatic Stump Detection:** Fast detection of near and far cricket wickets using deep learning.
2. **DRS-Line Generation:** Automatic geometric estimation of the longitudinal reference line ($A \to B$) connecting wicket base centres across 20.12 metres.
3. **Optional Pitch Alignment:** Integration of pitch boundary segmentation to cross-verify pitch orientation and enforce lateral ground constraints.
4. **Camera Calibration:** Mathematical recovery of camera intrinsics ($f$) and extrinsics ($R, \mathbf{t}$) via 4-point Stump+DRS and 8-point Stump+DRS+Pitch solvers with reprojection error minimization.
5. **Real-time & Video Processing:** Dual operational modes for Live Camera capture and Input Video processing.
6. **Scientifically Defensible Physics:** 3D ballistic trajectory fitting under gravity, velocity vector conversion to km/h, 95% confidence intervals, and a rigorous 10-gate physical Quality Control audit.

---

## 2. Two Operational Modes

### Mode 1: LIVE CAMERA
Designed for live net practice, broadcast setups, or connected webcams/capture cards.
- **Hardware Management:** Discovers connected USB/UVC cameras, selects resolution and target FPS.
- **Continuous Calibration Monitoring:** Performs optical flow (Lucas-Kanade) motion analysis on pitch background features to detect camera drift, panning, or physical vibration.
- **Real-Time Stump & DRS Overlay:** Overlays bounding boxes, stump tops, bases, base centres $A$ and $B$, and a vibrant cyan DRS reference line.
- **Speed Telemetry:** Detects bowling release and ball trajectory in real-time, displaying `MEASURED` speed with confidence metrics or explicit `NOT_MEASURED` status.
- **WebSocket Streaming:** Bi-directional JSON/binary WebSocket at `/ws/live` delivering sub-50ms telemetry.

### Mode 2: INPUT VIDEO
Designed for recorded broadcast footage, coaching deliveries, and official match review.
- **Drag-and-Drop Ingestion:** Accepts MP4, MOV, AVI, and WebM videos up to 500 MB.
- **Automatic Best-Frame Discovery:** Scans early frames to find the frame with the highest stump detection confidence and visibility.
- **Interactive Scrubber & Zoom/Pan:** High-resolution HTML5 canvas frame inspection with mouse-wheel zoom, drag-to-pan, and 1:1 coordinate preservation.
- **Automatic Stump & DRS Calibration:** Solves camera parameters automatically without requiring manual user landmark clicks.
- **Manual Landmark Correction UI:** Fallback 8-point visual pin repositioning with instant live reprojection error feedback.
- **Background Processing Engine:** Multi-threaded asynchronous delivery processing with real-time percentage progress bar.
- **Comprehensive Outputs:** Automatically generates broadcast video, diagnostic telemetry video, JSON audit trail, and 17-column Excel summary.

---

## 3. Mathematical Principles & Coordinate Geometry

### 3.1 World Coordinate System
The system strictly adopts canonical MCC/ICC Law 8 physical dimensions:
- **Origin $A (0, 0, 0)$:** Ground-level centre of the near wicket base (bowling crease).
- **Target $B (0, L, 0)$:** Ground-level centre of the far wicket base, where $L = 20.12\text{ m}$ (22 yards).
- **Longitudinal $+Y$ Axis:** Direct line from near wicket base centre $A$ to far wicket base centre $B$.
- **Lateral $+X$ Axis:** Perpendicular axis to the right when facing the far wicket.
- **Vertical $+Z$ Axis:** Directed upwards from the ground plane.

Canonical 3D Landmark Positions ($X, Y, Z$ in metres):
- Near Wicket Base: $(0.0, 0.0, 0.0)$
- Near Wicket Top: $(0.0, 0.0, 0.711)$
- Far Wicket Base: $(0.0, 20.12, 0.0)$
- Far Wicket Top: $(0.0, 20.12, 0.711)$
- Pitch Near Left: $(-1.524, 0.0, 0.0)$
- Pitch Near Right: $(+1.524, 0.0, 0.0)$
- Pitch Far Left: $(-1.524, 20.12, 0.0)$
- Pitch Far Right: $(+1.524, 20.12, 0.0)$

### 3.2 4-Point Stump + DRS Calibration
Four vertical non-coplanar points (near base, near top, far base, far top) provide 8 independent equations for 7 unknowns ($f, r_x, r_y, r_z, t_x, t_y, t_z$). The system solves this non-linear optimization problem using Levenberg-Marquardt least-squares:
$$\min_{f, R, \mathbf{t}} \sum_{i=1}^{4} \left\| \mathbf{u}_i - \pi(K, R, \mathbf{t}, \mathbf{X}_i) \right\|^2$$
where $\pi(\cdot)$ represents perspective projection through the pinhole camera matrix $K$.

### 3.3 3D Ballistic Trajectory Reconstruction
Airborne ball coordinates are reconstructed along camera optical rays using kinematic constraints and apparent diameter depth priors. The trajectory is fitted to a 3D ballistic model with aerodynamic drag and gravity:
$$\mathbf{X}(t) = \mathbf{X}_0 + \mathbf{V}_0 t + \frac{1}{2}\mathbf{a} t^2$$
Instantaneous speed at release moment $t_1$ is computed from the 3D velocity vector:
$$v_{\text{release}} = \sqrt{v_x^2 + v_y^2 + v_z^2} \times 3.6\text{ km/h}$$

---

## 4. Quality Control & 10-Gate Validation

Every delivery must satisfy 10 rigorous physical gates before displaying a `MEASURED` speed:
1. **Calibration Status:** Valid camera parameters with mean reprojection error $< 8.0\text{ px}$.
2. **Camera Stability:** Optical flow background drift $< 5.0\text{ px}$.
3. **Ball Detection Count:** Minimum 4 verified ball observations.
4. **Track Continuity:** Continuous Kalman trajectory without large temporal gaps.
5. **Release Detection:** Legitimate ball-hand separation event within the bowling crease zone.
6. **Trajectory Convergence:** Non-linear least-squares Huber fit convergence.
7. **Trajectory RMSE:** Residual trajectory reprojection error $< 6.0\text{ px}$.
8. **Physical Speed Bounds:** Release speed strictly within $60.0 \le v \le 170.0\text{ km/h}$.
9. **Release Height Feasibility:** Release $Z$ coordinate between $1.4\text{ m}$ and $2.8\text{ m}$.
10. **Depth Observability:** Observability score $\ge 0.15$.

If any gate fails, the delivery is classified as `NOT_MEASURED` with an explicit reason code (e.g. `INSUFFICIENT_BALL_DETECTIONS`, `SPEED_OUT_OF_BOUNDS`).

---

## 5. Output Deliverables

Upon processing a delivery, the system produces 4 complete artifacts:
1. **Final Processed Video (`_final.mp4`):** Clean broadcast video with an upper-left speed telemetry badge (Release Speed, C1-C2 Speed, Status).
2. **Diagnostic Telemetry Video (`_diagnostic.mp4`):** Full computer vision HUD overlay showing 2D bounding boxes, DRS reference line, bowler hand keypoints, and 3D fitted trajectory arc.
3. **JSON Audit Report (`_report.json`):** Comprehensive document containing camera intrinsics/extrinsics, frame-by-frame candidate coordinates, Kalman covariance matrices, release metrics, and QC gate evaluations.
4. **Excel Summary Report (`_summary.xlsx`):** Formatted 17-column sports analytics spreadsheet compatible with radar gun comparison workflows.

---

## 6. Directory Structure

```
cricket-ball-speed/
├── main.py                          # Unified CLI and Server launcher
├── Dockerfile                       # Production Docker container
├── docker-compose.yml               # Container orchestration & storage mounts
├── DEPLOYMENT.md                    # DigitalOcean production deployment guide
├── requirements.txt                 # Production dependencies
├── .env.example                     # Environment template
│
├── backend/                         # FastAPI Web & WebSocket Server
│   ├── app.py                       # REST API & WebSocket endpoints
│   ├── drs_engine.py                # Bridge between API and CalibrationManager
│   ├── live_service.py              # Live camera streaming & stability supervisor
│   └── video_service.py             # Video upload, frame decoding & async runner
│
├── frontend/                        # Web Dashboard (Reference UI)
│   ├── index.html                   # Pro Corporate League interface
│   └── static/
│       ├── style.css                # Navy/white minimalist theme with dashed pink borders
│       └── app.js                   # Dual-mode state controller & canvas renderer
│
├── cricket_ball_speed/              # Core Computer Vision Engine
│   ├── config/                      # YAML configuration files
│   ├── models/                      # Trained YOLO weights (ball, bowler, stumps, pitch)
│   ├── video/                       # Video decoders, PTS manager, stability analyzer
│   ├── calibration/                 # DRS line, auto stump detector, 4/8-pt calibration
│   ├── detection/                   # Multi-model YOLO inference & tiling
│   ├── filtering/                   # 2D/3D Kalman filters & prediction manager
│   ├── release/                     # Bowler kinematic analysis & release detection
│   ├── reconstruction/              # Ray casting, depth estimation, Huber trajectory fit
│   ├── speed/                       # Velocity estimator, C1-C2 speed, confidence intervals
│   ├── validation/                  # 10-gate quality control & radar benchmarking
│   └── output/                      # Video renderers, JSON generator, Excel writer
│
├── nginx/                           # Production Nginx reverse proxy configuration
│   └── cricket-speed.conf           # 500MB upload, WebSockets & SSL proxy config
│
├── scripts/                         # Automation & deployment scripts
│   ├── deploy_digitalocean.sh       # One-click Ubuntu Droplet deployment
│   └── systemd/                     # Systemd service units
│
├── storage/                         # Isolated Persistent Storage
│   ├── calibration/                 # SHA-256 locked calibration JSON files
│   ├── results/                     # Output videos, JSON reports, Excel workbooks
│   ├── tasks/                       # Cross-process task state
│   ├── logs/                        # System logs
│   └── uploads/                     # Uploaded input video files
│
└── tests/                           # Comprehensive Test Suite
    ├── test_auto_calibration_and_drs.py # Stump detection & DRS line math tests
    ├── test_drs_engine.py           # Backend engine integration tests
    └── test_api.py                  # FastAPI REST endpoint integration tests
```

---

## 7. Installation & Quick Start

### 7.1 Local Development Setup
```bash
# Clone the repository
git clone https://github.com/sidd8538singh-eng/Cricket-ball-speed.git
cd Cricket-ball-speed

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 7.2 Launching the Web Application
```bash
python main.py server --port 8000
```
Open your browser to:
- **Application UI:** `http://localhost:8000`
- **Interactive API Docs:** `http://localhost:8000/docs`
- **Healthcheck:** `http://localhost:8000/health`

### 7.3 Production Deployment on DigitalOcean
For full deployment instructions on DigitalOcean (Droplet, Docker Compose, or App Platform), see [DEPLOYMENT.md](DEPLOYMENT.md).

**Quick One-Click Droplet Deploy:**
```bash
chmod +x scripts/deploy_digitalocean.sh
sudo bash scripts/deploy_digitalocean.sh
```

**Using Docker Compose:**
```bash
docker compose up -d
```

### 7.4 Running Test Suites (113 Total Tests)
```bash
# 1. Run all 94 core physics, geometry, and tracking unit tests
python cricket_ball_speed/tests/run_all.py

# 2. Run all 20 FastAPI REST endpoints & DRS engine integration tests
python -m unittest discover -s tests -p "test_*.py" -v
```
