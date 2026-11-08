# Cricket Bowling Speed Measurement System

A production-grade computer vision system for calculating 3D cricket bowling velocity and release speed in km/h from fixed broadcast camera footage behind the bowler.

---

## 1. System Architecture

```
cricket_ball_speed/
│
├── main.py                          # CLI orchestrator & end-to-end pipeline
├── requirements.txt                 # Python dependencies
│
├── config/
│   ├── settings.yaml                # Global logging, device, tracking & output settings
│   ├── camera_config.yaml           # Physical pitch dimensions, mounting nominals & tolerances
│   ├── model_config.yaml            # YOLO detection, confidence thresholds & tiling parameters
│   └── validation_config.yaml       # Quality control thresholds & radar benchmark tiers
│
├── models/
│   ├── ball.pt                      # YOLO model for cricket ball
│   ├── bowler.pt                    # YOLO model for bowler bounding box
│   ├── stumps.pt                    # YOLO model for wickets
│   └── pitch.pt                     # Optional YOLO model for pitch boundaries
│
├── video/
│   ├── video_loader.py              # Video stream coordinator & stability supervisor
│   ├── metadata.py                  # Metadata extraction (FPS, PTS, duration, codec, SHA256)
│   ├── frame_decoder.py             # Frame-by-frame sequential & random-access decoder
│   ├── timestamp_manager.py         # Hardware presentation timestamps (PTS) & VFR jitter manager
│   └── camera_stability.py         # Optical flow background displacement & zoom detector
│
├── calibration/
│   ├── manual_calibration_ui.py     # Interactive OpenCV UI (Zoom/Pan/Undo/Replace/Lock)
│   ├── point_selector.py            # 8-point landmark tracker & state machine
│   ├── coordinate_manager.py        # Zoom/pan <-> original pixel coordinate transformer
│   ├── world_coordinates.py         # Canonical ICC 3D coordinate system (A-B longitudinal axis)
│   ├── homography.py                # Ground-plane homography (strictly for Z=0 points)
│   ├── camera_model.py              # Pinhole & distorted camera model (K, R, t, ray casting)
│   ├── camera_calibration.py        # Levenberg-Marquardt solver for camera pose & focal length
│   ├── reprojection.py              # Landmark reprojection error & confidence evaluator
│   └── calibration_validator.py     # Perspective geometry & physical mounting validator
│
├── detection/
│   ├── ball_detector.py             # Full-frame & tiled small-object ball detector
│   ├── bowler_detector.py           # Bowler detector & spatial crease tracking
│   ├── stumps_detector.py           # Wicket detector (verification without overriding calibration)
│   ├── pitch_detector.py            # Optional pitch polygon segmentation
│   ├── tiled_inference.py           # Overlapping high-res image slicer & NMS merger
│   └── detection_manager.py         # Multi-model inference coordinator & raw audit logger
│
├── ball_processing/
│   ├── ball_observations.py         # Data structures holding raw/refined states & provenance
│   ├── bbox_refinement.py           # Sub-pixel centroid refinement via image moments
│   ├── apparent_diameter.py         # Depth prior estimation from apparent pixel diameter
│   ├── outlier_rejection.py         # Kinematic displacement & diameter jump filters
│   └── observation_quality.py       # Aspect ratio, circularity, and contrast scoring
│
├── filtering/
│   ├── kalman_2d.py                 # 2D image-space constant velocity Kalman filter
│   ├── kalman_3d.py                 # 3D world-space ballistic Kalman filter (gravity Z)
│   ├── data_association.py          # Mahalanobis distance gating
│   └── prediction_manager.py        # Track lifecycle: distinguishes real from predicted points
│
├── release/
│   ├── bowler_analysis.py           # Bowler approach & delivery stride window analyzer
│   ├── hand_estimation.py           # Upper-quadrant bowling hand localization
│   ├── release_detector.py          # Ball-hand separation & release moment (t1, C1) detector
│   └── release_validator.py         # Biomechanical release height & position validator
│
├── reconstruction/
│   ├── camera_rays.py               # 3D ray back-projection & multi-camera triangulation
│   ├── depth_estimation.py          # Depth estimation along rays using apparent ball diameter
│   ├── ball_3d.py                   # 3D position dataclass with uncertainty ellipsoids
│   ├── trajectory_fitting.py        # Robust non-linear Huber ballistic trajectory solver
│   └── uncertainty.py               # Jacobian covariance matrix & speed error propagation
│
├── speed/
│   ├── velocity_estimator.py        # 3D velocity vector to km/h conversion
│   ├── release_speed.py             # Instantaneous release speed at t1 with 95% CI
│   ├── c1_c2_speed.py               # Baseline average speed from C1 to C2 (5-8m down pitch)
│   ├── framewise_speed.py           # Frame-by-frame speed profile & moving average
│   └── speed_validator.py           # Physical range & aerodynamic consistency validator
│
├── validation/
│   ├── quality_control.py           # 8-gate QA pipeline (MEASURED / NOT_MEASURED)
│   ├── radar_validation.py          # Independent Doppler radar validation (MAE, RMSE, Bias)
│   ├── benchmark.py                 # Automated dataset benchmarking runner
│   └── rejection_reasons.py         # Canonical rejection codes and audit descriptions
│
├── output/
│   ├── final_video.py               # Broadcast video with upper-left speed badge
│   ├── diagnostic_video.py          # Telemetry overlay video with HUD, bboxes, & 3D arc
│   ├── json_report.py               # Comprehensive JSON audit trail
│   └── excel_report.py              # Multi-column formatted Excel workbook (.xlsx)
│
├── storage/
│   ├── calibration/                 # SHA-256 locked calibration JSON files
│   ├── results/                     # Video outputs, reports, and spreadsheets
│   └── logs/                        # System execution logs
│
└── tests/                           # 34 comprehensive unit and integration tests
    ├── run_all.py                   # Master test runner
    ├── test_video.py
    ├── test_calibration.py
    ├── test_coordinates.py
    ├── test_detection.py
    ├── test_kalman.py
    ├── test_release.py
    ├── test_reconstruction.py
    └── test_speed.py
```

---

## 2. Physical Setup & Canonical Dimensions

The system uses standard ICC and MCC cricket ground dimensions:
- **Pitch Length (Crease to Crease):** `20.12` metres (22 yards).
- **Pitch Width:** `3.048` metres (10 feet).
- **Stump Height:** `0.711` metres (28 inches).
- **Cricket Ball Diameter:** `0.0715` metres (7.15 cm).
- **Camera Mounting:**
  - Lens-centre height: `2.0` metres above ground (tolerance: `1.4m` - `2.8m`).
  - Distance behind near stumps ($A = (0,0,0)$): `1.5m` - `2.5m` ($Y = -2.0m$).
  - Target framerate: `120 FPS` preferred (`60 FPS` minimum).
  - Mount: Rigid, fixed focus and zoom.

### World Coordinate System
- **Origin $A = (0, 0, 0)$:** Centre of near wicket base at ground level.
- **Far Stumps $B = (0, L, 0)$:** Centre of far wicket base at ground level ($L = 20.12$ m).
- **$+Y$ Axis:** Longitudinal direction from near stumps to far stumps.
- **$+X$ Axis:** Lateral direction (to the right when looking down-pitch).
- **$+Z$ Axis:** Vertical direction pointing upwards from the ground plane.

---

## 3. Installation

### Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- NVIDIA GPU with CUDA recommended for real-time 120 FPS YOLO inference (CPU supported).

```bash
# Clone or navigate to the repository
cd cricket_ball_speed

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Model Loading Instructions

Place YOLO `.pt` model weights inside `models/`:
1. `models/ball.pt` - Custom trained YOLO model for small cricket ball detection.
2. `models/bowler.pt` - Person/bowler detection model.
3. `models/stumps.pt` - Cricket stumps detector.
4. `models/pitch.pt` - (Optional) Pitch boundary detector.

The system automatically detects whether [Ultralytics](https://github.com/ultralytics/ultralytics) is installed. If `.pt` files are absent, the detector initializes in fallback mode for testing and structural verification.

---

## 5. Manual Calibration Instructions

Manual calibration is mandatory for each newly uploaded video to avoid arbitrary focal length guessing:

```bash
python main.py --video path/to/delivery.mp4 --calibrate-only
```

An interactive OpenCV window opens displaying frame 0:
1. **Click the 8 landmarks in strict sequential order:**
   - `1/8: near_top` (near wicket top centre)
   - `2/8: near_base` (near wicket base centre on ground)
   - `3/8: far_top` (far wicket top centre)
   - `4/8: far_base` (far wicket base centre on ground)
   - `5/8: pitch_left_near` (pitch near-left corner)
   - `6/8: pitch_right_near` (pitch near-right corner)
   - `7/8: pitch_left_far` (pitch far-left corner)
   - `8/8: pitch_right_far` (pitch far-right corner)
2. **Interactive Controls:**
   - **Zoom:** Mouse wheel or `+` / `-` keys.
   - **Pan:** Right-click and drag (or middle-click drag).
   - **Undo:** Press `U` to undo the last clicked landmark.
   - **Reset:** Press `R` to clear all landmarks.
   - **Re-pick single point:** Press keys `1` through `8` to replace that specific point.
   - **Confirm:** Press `C` or `Enter` when all 8 points are placed.
   - **Quit:** Press `Q` or `Esc`.

Once confirmed, the camera parameters $(K, R, t)$ and reprojection report are computed and locked to `storage/calibration/{SHA256_HASH}.json`. Later pipeline runs on the same video automatically reuse the locked calibration without prompting.

---

## 6. Running the System

### Full Pipeline Execution
```bash
# Process a delivery with automatic or existing calibration
python main.py --video path/to/delivery.mp4

# Run headlessly using existing locked calibration
python main.py --video path/to/delivery.mp4 --headless

# Include independent radar gun speed for automated validation comparison
python main.py --video path/to/delivery.mp4 --headless --radar-speed 138.5
```

---

## 7. Running the Test Suite

The test suite contains 34 comprehensive tests validating video decoding, coordinate systems, camera calibration, homography, 2D/3D Kalman filtering, release detection, trajectory fitting, and radar benchmarking.

```bash
# Run all tests via master test runner
python tests/run_all.py

# Or via standard unittest
python -m unittest discover tests
```

---

## 8. Diagnostic and Output Files

For every processed delivery, four outputs are written to `storage/results/`:

1. **Final Processed Video (`*_final.mp4`):**
   - Clean broadcast display.
   - Modern speed badge rendered in the **upper-left region** (e.g. `142.4 km/h` or `NOT MEASURED`).
   - Displays delivery date/time.
   - Strictly free of bounding boxes, frame numbers, or crosshairs.
   - Automatically freezes the final frame for 1.5 seconds.
2. **Diagnostic Engineering Video (`*_diagnostic.mp4`):**
   - Displays raw detections (red/green bboxes).
   - Kalman filter predicted positions (cyan markers) vs confirmed detections (green dots).
   - Release point $C_1$ (magenta star) and baseline point $C_2$ (yellow diamond).
   - 3D ballistic trajectory arc reprojected onto the image.
   - Real-time telemetry HUD with speeds, trajectory RMSE, and QC failure explanations.
3. **Structured JSON Report (`*_report.json`):**
   - Full audit trail: video metadata, camera matrix, rotation, translation, reprojection errors, release event timestamp $t_1$, 3D coordinates, initial velocities $(V_X, V_Y, V_Z)$, and QC status.
4. **Excel Summary Report (`*_summary.xlsx`):**
   - Professional multi-column spreadsheet formatted via `openpyxl`.
   - Records calculated speed, C1-C2 speed, radar speed, difference, variance, reprojection error, detection counts, and rejection reasons.

---

## 9. Quality Control & Rejection Gates

Every result must pass 8 scientific quality gates before reporting speed. If any check fails, the delivery status is set to `NOT_MEASURED` with an explicit rejection reason:

| Gate | Criterion | Action on Failure |
|---|---|---|
| **Camera Stability** | Frame drift $\le 3.5$ px | `CAMERA_UNSTABLE` |
| **Video FPS** | FPS $\ge 59.0$ FPS | `FRAMERATE_TOO_LOW` |
| **Calibration Reprojection** | Mean landmark error $\le 6.0$ px | `REPROJECTION_ERROR_HIGH` |
| **Perspective Geometry** | Near wicket taller than far wicket | `PERSPECTIVE_GEOMETRY_INVALID` |
| **Ball Detection Count** | At least 5 confirmed detections | `INSUFFICIENT_BALL_DETECTIONS` |
| **Prediction Gap** | Max consecutive missing detections $\le 3$ | `LONG_PREDICTION_GAP` |
| **Release Confidence** | Release event confidence $\ge 0.50$ | `RELEASE_CONFIDENCE_LOW` |
| **Speed Uncertainty** | $95\%$ CI uncertainty $\le 8.5$ km/h | `SPEED_UNCERTAINTY_EXCESSIVE` |
| **Physical Sanity** | Instantaneous speed in $[30, 175]$ km/h | `SPEED_OUT_OF_RANGE` |

---

## 10. Known Technical Limitations

1. **Extreme Occlusion by Batsman/Umpire:** If the bowler's body or the non-striker completely occludes the ball for more than 4 consecutive frames right after release, the Kalman track terminates to avoid fabricating trajectories, returning `NOT_MEASURED`.
2. **Monocular Depth Ambiguity:** A single camera image point defines a 3D ray, not an isolated point. The system resolves this using multi-frame ballistic optimization under gravity combined with apparent ball diameter cues. While highly accurate with 10+ frames, accuracy degrades if fewer than 5 frames are detected.
3. **Framerate Constraints:** At 60 FPS, a 140 km/h delivery moves approximately 0.65 metres per frame. At 120 FPS, the ball moves only 0.32 metres per frame. 120 FPS is strongly recommended for sub-1 km/h precision.
