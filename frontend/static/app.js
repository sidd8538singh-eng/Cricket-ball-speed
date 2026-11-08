/**
 * PRO CORPORATE LEAGUE - CRICKET BOWLING SPEED ANALYZER
 * Client-side Controller for Live Camera & Input Video Modes
 */

(function () {
  'use strict';

  // State Management
  const state = {
    mode: 'live', // 'live' or 'video'
    systemOnline: false,

    // Live Camera State
    cameraStream: null,
    cameraRunning: false,
    cameraSocket: null,
    liveInterval: null,
    liveLatestTelemetry: null,
    liveCalibrated: false,

    // Video Mode State
    currentVideoPath: null,
    currentVideoHash: null,
    currentVideoMeta: null,
    currentFrameIndex: 0,
    isPlaying: false,
    playInterval: null,
    calibrationFrameIndex: 0,
    activeDetection: null,
    activeCalibration: null,
    isManualCorrectionActive: false,
    manualPoints: {
      near_top: null,
      near_base: null,
      far_top: null,
      far_base: null,
      pitch_left_near: null,
      pitch_right_near: null,
      pitch_left_far: null,
      pitch_right_far: null
    },
    activePointKey: 'near_top',

    // Viewport Pan/Zoom
    zoom: 1.0,
    panX: 0,
    panY: 0,
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,

    // Analysis State
    analysisTaskId: null,
    analysisPollInterval: null,
    analysisResult: null
  };

  // Canonical calibration keys in strict order
  const CALIBRATION_KEYS = [
    'near_top', 'near_base', 'far_top', 'far_base',
    'pitch_left_near', 'pitch_right_near', 'pitch_left_far', 'pitch_right_far'
  ];

  // DOM Elements Cache
  const elements = {
    // Top Bar
    cameraStatusPill: document.getElementById('cameraStatusPill'),
    cameraStatusText: document.getElementById('cameraStatusText'),
    systemStatusPill: document.getElementById('systemStatusPill'),
    systemStatusText: document.getElementById('systemStatusText'),
    settingsBtn: document.getElementById('settingsBtn'),
    settingsModal: document.getElementById('settingsModal'),
    closeSettingsBtn: document.getElementById('closeSettingsBtn'),
    saveSettingsBtn: document.getElementById('saveSettingsBtn'),
    settingStumpsConf: document.getElementById('settingStumpsConf'),

    // Mode Selector
    modeCardLive: document.getElementById('modeCardLive'),
    modeCardVideo: document.getElementById('modeCardVideo'),
    liveModeView: document.getElementById('liveModeView'),
    videoModeView: document.getElementById('videoModeView'),

    // Live Viewport
    cameraDeviceSelect: document.getElementById('cameraDeviceSelect'),
    startCameraBtn: document.getElementById('startCameraBtn'),
    stopCameraBtn: document.getElementById('stopCameraBtn'),
    liveVideo: document.getElementById('liveVideo'),
    liveCanvas: document.getElementById('liveCanvas'),
    liveStumpsBadge: document.getElementById('liveStumpsBadge'),
    liveStumpsText: document.getElementById('liveStumpsText'),
    liveDrsBadge: document.getElementById('liveDrsBadge'),
    liveDrsText: document.getElementById('liveDrsText'),
    liveStabilityBadge: document.getElementById('liveStabilityBadge'),
    liveStabilityText: document.getElementById('liveStabilityText'),

    // Live Dashboard
    liveSpeedNumber: document.getElementById('liveSpeedNumber'),
    liveMeasurementBadge: document.getElementById('liveMeasurementBadge'),
    liveC1C2Speed: document.getElementById('liveC1C2Speed'),
    liveUncertainty: document.getElementById('liveUncertainty'),
    liveCalibPill: document.getElementById('liveCalibPill'),
    liveNearWicketVal: document.getElementById('liveNearWicketVal'),
    liveFarWicketVal: document.getElementById('liveFarWicketVal'),
    liveDrsStatusVal: document.getElementById('liveDrsStatusVal'),
    livePitchStatusVal: document.getElementById('livePitchStatusVal'),
    liveStabilityVal: document.getElementById('liveStabilityVal'),
    liveReprojVal: document.getElementById('liveReprojVal'),
    liveAutoCalibBtn: document.getElementById('liveAutoCalibBtn'),
    liveConfirmCalibBtn: document.getElementById('liveConfirmCalibBtn'),
    liveResetCalibBtn: document.getElementById('liveResetCalibBtn'),

    // Video Viewport & Upload
    uploadPanel: document.getElementById('uploadPanel'),
    videoFileInput: document.getElementById('videoFileInput'),
    browseVideoBtn: document.getElementById('browseVideoBtn'),
    sampleChipsList: document.getElementById('sampleChipsList'),
    videoWorkspace: document.getElementById('videoWorkspace'),
    videoCanvas: document.getElementById('videoCanvas'),
    calibrationFrameImg: document.getElementById('calibrationFrameImg'),
    videoStumpsBadge: document.getElementById('videoStumpsBadge'),
    videoStumpsText: document.getElementById('videoStumpsText'),
    videoDrsBadge: document.getElementById('videoDrsBadge'),
    videoDrsText: document.getElementById('videoDrsText'),

    // Playback & Scrubber
    stepBackBtn: document.getElementById('stepBackBtn'),
    playPauseBtn: document.getElementById('playPauseBtn'),
    stepForwardBtn: document.getElementById('stepForwardBtn'),
    frameSlider: document.getElementById('frameSlider'),
    timeReadout: document.getElementById('timeReadout'),
    zoomInBtn: document.getElementById('zoomInBtn'),
    zoomOutBtn: document.getElementById('zoomOutBtn'),
    resetViewBtn: document.getElementById('resetViewBtn'),
    autoSelectFrameBtn: document.getElementById('autoSelectFrameBtn'),

    // Video Telemetry
    videoCalibStatusPill: document.getElementById('videoCalibStatusPill'),
    videoMetaVal: document.getElementById('videoMetaVal'),
    videoStumpCountVal: document.getElementById('videoStumpCountVal'),
    videoDrsVal: document.getElementById('videoDrsVal'),
    videoDrsLengthVal: document.getElementById('videoDrsLengthVal'),
    videoReprojVal: document.getElementById('videoReprojVal'),
    videoPitchCornersVal: document.getElementById('videoPitchCornersVal'),
    videoAutoCalibBtn: document.getElementById('videoAutoCalibBtn'),
    toggleCorrectionBtn: document.getElementById('toggleCorrectionBtn'),
    confirmVideoCalibBtn: document.getElementById('confirmVideoCalibBtn'),
    startAnalysisBtn: document.getElementById('startAnalysisBtn'),

    // Video Meta Details
    metaFilename: document.getElementById('metaFilename'),
    metaResolution: document.getElementById('metaResolution'),
    metaFps: document.getElementById('metaFps'),
    metaDuration: document.getElementById('metaDuration'),

    // Analysis Progress & Results
    analysisProgressCard: document.getElementById('analysisProgressCard'),
    progressStageText: document.getElementById('progressStageText'),
    progressPctText: document.getElementById('progressPctText'),
    progressFill: document.getElementById('progressFill'),
    analysisResultsDashboard: document.getElementById('analysisResultsDashboard'),
    resultsMeasurementBadge: document.getElementById('resultsMeasurementBadge'),
    resultReleaseSpeed: document.getElementById('resultReleaseSpeed'),
    resultUncertainty: document.getElementById('resultUncertainty'),
    resultC1C2Speed: document.getElementById('resultC1C2Speed'),
    resultQcScore: document.getElementById('resultQcScore'),
    resultStatusVal: document.getElementById('resultStatusVal'),
    resultRejectionVal: document.getElementById('resultRejectionVal'),

    // Result Media & Downloads
    finalVideoPlayer: document.getElementById('finalVideoPlayer'),
    diagVideoPlayer: document.getElementById('diagVideoPlayer'),
    jsonReportText: document.getElementById('jsonReportText'),
    dlFinalVideo: document.getElementById('dlFinalVideo'),
    dlDiagVideo: document.getElementById('dlDiagVideo'),
    dlJsonReport: document.getElementById('dlJsonReport'),
    dlExcelReport: document.getElementById('dlExcelReport')
  };

  // ============================================================================
  // Initialization & System Status
  // ============================================================================
  async function initApp() {
    setupEventListeners();
    await checkSystemStatus();
    await loadSampleVideos();
    await populateCameraDevices();
  }

  async function checkSystemStatus() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        state.systemOnline = true;
        elements.systemStatusPill.className = 'status-pill online';
        elements.systemStatusText.textContent = data.device.device_name || 'System Online';
        if (data.active_calibration_present) {
          state.activeCalibration = data;
        }
      }
    } catch (e) {
      elements.systemStatusPill.className = 'status-pill';
      elements.systemStatusText.textContent = 'Server Offline';
    }
  }

  async function loadSampleVideos() {
    try {
      const res = await fetch('/api/sample_videos');
      if (res.ok) {
        const data = await res.json();
        if (data.sample_videos && data.sample_videos.length > 0) {
          elements.sampleChipsList.innerHTML = '';
          data.sample_videos.forEach(sv => {
            const btn = document.createElement('button');
            btn.className = 'sample-chip';
            btn.innerHTML = `⚡ ${sv.name} (${(sv.size_bytes / 1024 / 1024).toFixed(1)} MB)`;
            btn.addEventListener('click', () => loadVideoByPath(sv.path));
            elements.sampleChipsList.appendChild(btn);
          });
        }
      }
    } catch (e) {
      console.warn('Could not load sample videos:', e);
    }
  }

  async function populateCameraDevices() {
    try {
      const res = await fetch('/api/camera/list');
      if (res.ok) {
        const data = await res.json();
        if (data.cameras && data.cameras.length > 0) {
          elements.cameraDeviceSelect.innerHTML = '';
          data.cameras.forEach(cam => {
            const opt = document.createElement('option');
            opt.value = cam.device_index;
            opt.textContent = `${cam.name} (${cam.resolution} @ ${cam.fps} FPS)`;
            elements.cameraDeviceSelect.appendChild(opt);
          });
        }
      }
    } catch (e) {
      console.warn('Could not enumerate cameras:', e);
    }
  }

  // ============================================================================
  // Event Listeners
  // ============================================================================
  function setupEventListeners() {
    // Mode Switching
    elements.modeCardLive.addEventListener('click', () => switchMode('live'));
    elements.modeCardVideo.addEventListener('click', () => switchMode('video'));

    // Settings Modal
    elements.settingsBtn.addEventListener('click', () => elements.settingsModal.classList.add('active'));
    elements.closeSettingsBtn.addEventListener('click', () => elements.settingsModal.classList.remove('active'));
    elements.saveSettingsBtn.addEventListener('click', () => elements.settingsModal.classList.remove('active'));

    // Live Camera Controls
    elements.startCameraBtn.addEventListener('click', startLiveCamera);
    elements.stopCameraBtn.addEventListener('click', stopLiveCamera);
    elements.liveAutoCalibBtn.addEventListener('click', triggerLiveAutoCalibrate);
    elements.liveConfirmCalibBtn.addEventListener('click', confirmLiveCalibration);
    elements.liveResetCalibBtn.addEventListener('click', resetLiveCalibration);

    // Video Upload
    elements.browseVideoBtn.addEventListener('click', () => elements.videoFileInput.click());
    elements.videoFileInput.addEventListener('change', handleVideoFileSelect);
    setupDragAndDrop();

    // Video Playback & Scrubber
    elements.frameSlider.addEventListener('input', handleSliderChange);
    elements.stepBackBtn.addEventListener('click', () => stepFrame(-1));
    elements.stepForwardBtn.addEventListener('click', () => stepFrame(1));
    elements.playPauseBtn.addEventListener('click', togglePlayPause);
    elements.autoSelectFrameBtn.addEventListener('click', autoSelectBestCalibrationFrame);

    // Viewport Zoom & Pan
    elements.zoomInBtn.addEventListener('click', () => applyZoom(1.2));
    elements.zoomOutBtn.addEventListener('click', () => applyZoom(0.8));
    elements.resetViewBtn.addEventListener('click', resetZoomAndPan);
    setupCanvasInteractivity(elements.videoCanvas);

    // Video Calibration Actions
    elements.videoAutoCalibBtn.addEventListener('click', triggerVideoAutoCalibrate);
    elements.toggleCorrectionBtn.addEventListener('click', toggleManualCorrectionMode);
    elements.confirmVideoCalibBtn.addEventListener('click', confirmVideoCalibration);
    elements.startAnalysisBtn.addEventListener('click', startFullVideoAnalysis);

    // Tab Navigation
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        const targetId = btn.getAttribute('data-tab');
        const targetContent = document.getElementById(targetId);
        if (targetContent) targetContent.classList.add('active');
      });
    });
  }

  function switchMode(newMode) {
    if (state.mode === newMode) return;
    state.mode = newMode;

    if (newMode === 'live') {
      elements.modeCardLive.classList.add('active');
      elements.modeCardVideo.classList.remove('active');
      elements.liveModeView.classList.add('active');
      elements.videoModeView.classList.remove('active');
      if (state.isPlaying) togglePlayPause();
    } else {
      elements.modeCardVideo.classList.add('active');
      elements.modeCardLive.classList.remove('active');
      elements.videoModeView.classList.add('active');
      elements.liveModeView.classList.remove('active');
      if (state.cameraRunning) stopLiveCamera();
    }
  }

  // ============================================================================
  // MODE 1: LIVE CAMERA IMPLEMENTATION
  // ============================================================================
  async function startLiveCamera() {
    try {
      const devIdx = parseInt(elements.cameraDeviceSelect.value || '0', 10);
      elements.startCameraBtn.disabled = true;
      elements.startCameraBtn.textContent = 'Connecting...';

      // Open camera via browser getUserMedia for high frame rate, or server endpoint
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 } },
        audio: false
      });

      state.cameraStream = stream;
      elements.liveVideo.srcObject = stream;
      await elements.liveVideo.play();

      state.cameraRunning = true;
      elements.startCameraBtn.style.display = 'none';
      elements.stopCameraBtn.style.display = 'inline-flex';
      elements.cameraStatusPill.className = 'status-pill online';
      elements.cameraStatusText.textContent = `Camera Online (${elements.liveVideo.videoWidth || 1280}x${elements.liveVideo.videoHeight || 720})`;

      // Connect WebSocket for high-frequency telemetry
      connectLiveWebSocket();

      // Start processing loop
      startLiveFrameLoop();
    } catch (err) {
      console.warn('getUserMedia failed, connecting via backend capture:', err);
      // Fallback: connect backend capture device
      const res = await fetch('/api/camera/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ device_index: parseInt(elements.cameraDeviceSelect.value, 10), width: 1280, height: 720 })
      });
      if (res.ok) {
        state.cameraRunning = true;
        elements.startCameraBtn.style.display = 'none';
        elements.stopCameraBtn.style.display = 'inline-flex';
        elements.cameraStatusPill.className = 'status-pill online';
        elements.cameraStatusText.textContent = 'Backend Camera Active';
        connectLiveWebSocket();
        startBackendLivePoll();
      } else {
        alert('Could not start camera. Please verify device connection and permissions.');
      }
    } finally {
      elements.startCameraBtn.disabled = false;
      elements.startCameraBtn.textContent = '▶ Start Camera';
    }
  }

  function stopLiveCamera() {
    state.cameraRunning = false;
    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach(track => track.stop());
      state.cameraStream = null;
    }
    if (state.cameraSocket) {
      state.cameraSocket.close();
      state.cameraSocket = null;
    }
    if (state.liveInterval) {
      clearInterval(state.liveInterval);
      state.liveInterval = null;
    }

    elements.startCameraBtn.style.display = 'inline-flex';
    elements.stopCameraBtn.style.display = 'none';
    elements.cameraStatusPill.className = 'status-pill';
    elements.cameraStatusText.textContent = 'Camera Offline';
    fetch('/api/camera/disconnect', { method: 'POST' }).catch(() => {});

    // Clear canvas
    const ctx = elements.liveCanvas.getContext('2d');
    ctx.clearRect(0, 0, elements.liveCanvas.width, elements.liveCanvas.height);
  }

  function connectLiveWebSocket() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${proto}//${window.location.host}/ws/live`;
    try {
      const ws = new WebSocket(wsUrl);
      ws.onopen = () => console.log('Live WebSocket connected.');
      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === 'telemetry' || msg.type === 'detection') {
            handleLiveTelemetry(msg.data);
          }
        } catch (e) {}
      };
      ws.onerror = (e) => console.warn('WebSocket error:', e);
      state.cameraSocket = ws;
    } catch (e) {
      console.warn('Could not establish WebSocket:', e);
    }
  }

  function startLiveFrameLoop() {
    const offscreenCanvas = document.createElement('canvas');
    const offCtx = offscreenCanvas.getContext('2d');

    state.liveInterval = setInterval(() => {
      if (!state.cameraRunning || !elements.liveVideo.videoWidth) return;

      const vw = elements.liveVideo.videoWidth;
      const vh = elements.liveVideo.videoHeight;

      elements.liveCanvas.width = vw;
      elements.liveCanvas.height = vh;
      offscreenCanvas.width = 640;
      offscreenCanvas.height = Math.round(640 * (vh / vw));

      // Draw current video frame
      offCtx.drawImage(elements.liveVideo, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
      const base64Data = offscreenCanvas.toDataURL('image/jpeg', 0.65);

      if (state.cameraSocket && state.cameraSocket.readyState === WebSocket.OPEN) {
        state.cameraSocket.send(JSON.stringify({
          type: 'frame',
          image: base64Data,
          confidence: parseFloat(elements.settingStumpsConf.value || '0.20')
        }));
      } else {
        // Fallback REST call
        fetch('/api/live/process_frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_base64: base64Data, confidence_threshold: 0.20 })
        })
        .then(r => r.json())
        .then(data => handleLiveTelemetry(data))
        .catch(() => {});
      }
    }, 120); // ~8-10 FPS live inference sample
  }

  function startBackendLivePoll() {
    state.liveInterval = setInterval(async () => {
      if (!state.cameraRunning) return;
      try {
        const res = await fetch('/api/camera/status');
        if (res.ok) {
          const data = await res.json();
          // Update live telemetry
        }
      } catch (e) {}
    }, 300);
  }

  function handleLiveTelemetry(data) {
    state.liveLatestTelemetry = data;
    const det = data.detection || data;

    // Render Canvas Overlay
    const ctx = elements.liveCanvas.getContext('2d');
    ctx.clearRect(0, 0, elements.liveCanvas.width, elements.liveCanvas.height);

    // Draw Video Frame to Canvas
    if (elements.liveVideo && elements.liveVideo.videoWidth) {
      ctx.drawImage(elements.liveVideo, 0, 0, elements.liveCanvas.width, elements.liveCanvas.height);
    }

    // Draw DRS Geometry & Overlays
    renderDrsOverlays(ctx, det, elements.liveCanvas.width, elements.liveCanvas.height);

    // Update Telemetry Badges & Sidebar
    const hasNear = !!det.near_wicket;
    const hasFar = !!det.far_wicket;
    const hasDrs = !!(det.drs_line && det.drs_line.is_valid);

    if (hasNear && hasFar) {
      elements.liveStumpsBadge.className = 'hud-badge success';
      elements.liveStumpsText.textContent = 'Both Wickets Found';
      elements.liveNearWicketVal.textContent = `Found (${(det.near_wicket.confidence * 100).toFixed(0)}%)`;
      elements.liveFarWicketVal.textContent = `Found (${(det.far_wicket.confidence * 100).toFixed(0)}%)`;
    } else if (hasNear || hasFar) {
      elements.liveStumpsBadge.className = 'hud-badge warning';
      elements.liveStumpsText.textContent = hasNear ? 'Near Wicket Only' : 'Far Wicket Only';
      elements.liveNearWicketVal.textContent = hasNear ? `Found (${(det.near_wicket.confidence * 100).toFixed(0)}%)` : 'Searching';
      elements.liveFarWicketVal.textContent = hasFar ? `Found (${(det.far_wicket.confidence * 100).toFixed(0)}%)` : 'Searching';
    } else {
      elements.liveStumpsBadge.className = 'hud-badge danger';
      elements.liveStumpsText.textContent = 'Searching Stumps';
      elements.liveNearWicketVal.textContent = 'Searching';
      elements.liveFarWicketVal.textContent = 'Searching';
    }

    if (hasDrs) {
      elements.liveDrsBadge.style.display = 'inline-flex';
      elements.liveDrsText.textContent = `20.12m Line (${det.drs_line.pixel_length.toFixed(0)}px)`;
      elements.liveDrsStatusVal.textContent = `Generated (${det.drs_line.pixel_length.toFixed(0)}px, ${det.drs_line.angle_deg.toFixed(1)}°)`;
      elements.liveDrsStatusVal.className = 'telemetry-val success';
      elements.liveConfirmCalibBtn.disabled = false;
    } else {
      elements.liveDrsBadge.style.display = 'none';
      elements.liveDrsStatusVal.textContent = 'Not Generated';
      elements.liveDrsStatusVal.className = 'telemetry-val';
      elements.liveConfirmCalibBtn.disabled = true;
    }

    // Pitch status
    if (det.pitch_bbox) {
      elements.livePitchStatusVal.textContent = 'Detected';
      elements.livePitchStatusVal.className = 'telemetry-val success';
    } else {
      elements.livePitchStatusVal.textContent = 'Model Inactive';
      elements.livePitchStatusVal.className = 'telemetry-val';
    }

    // Camera stability
    const isStable = data.camera_status ? data.camera_status.is_stable : true;
    if (isStable) {
      elements.liveStabilityBadge.className = 'hud-badge success';
      elements.liveStabilityText.textContent = 'Stationary';
      elements.liveStabilityVal.textContent = 'Stable';
      elements.liveStabilityVal.className = 'telemetry-val success';
    } else {
      elements.liveStabilityBadge.className = 'hud-badge danger';
      elements.liveStabilityText.textContent = `Drift: ${data.camera_status.drift_px}px`;
      elements.liveStabilityVal.textContent = 'Vibration / Movement';
      elements.liveStabilityVal.className = 'telemetry-val danger';
    }

    // Speed Dashboard Update
    if (data.speed_dashboard) {
      const spd = data.speed_dashboard;
      if (spd.release_speed_kmh) {
        elements.liveSpeedNumber.textContent = spd.release_speed_kmh.toFixed(1);
        elements.liveMeasurementBadge.className = 'measurement-pill measured';
        elements.liveMeasurementBadge.textContent = 'MEASURED';
      }
      if (spd.c1_c2_speed_kmh) {
        elements.liveC1C2Speed.textContent = `${spd.c1_c2_speed_kmh.toFixed(1)} km/h`;
      }
      if (spd.uncertainty_kmh) {
        elements.liveUncertainty.textContent = `±${spd.uncertainty_kmh.toFixed(1)} km/h`;
      }
    }
  }

  function triggerLiveAutoCalibrate() {
    if (!state.liveLatestTelemetry) {
      alert('Awaiting camera frame detection before calibration.');
      return;
    }
    const det = state.liveLatestTelemetry.detection || state.liveLatestTelemetry;
    if (!det.near_wicket || !det.far_wicket) {
      alert('Both wickets must be clearly visible in camera preview to calibrate.');
      return;
    }

    fetch('/api/calibrate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_points: {
          near_base: det.near_wicket.base_center,
          near_top: det.near_wicket.top_center,
          far_base: det.far_wicket.base_center,
          far_top: det.far_wicket.top_center
        },
        image_width: elements.liveCanvas.width || 1280,
        image_height: elements.liveCanvas.height || 720
      })
    })
    .then(r => r.json())
    .then(res => {
      if (res.success) {
        state.liveCalibrated = true;
        elements.liveCalibPill.className = 'status-pill online';
        elements.liveCalibPill.textContent = 'Calibrated';
        elements.liveReprojVal.textContent = `${res.reprojection_error_px.toFixed(2)} px`;
        elements.liveReprojVal.className = 'telemetry-val success';
        alert(`Camera Calibrated Successfully!\nReprojection RMSE: ${res.reprojection_error_px.toFixed(2)}px\nLongitudinal distance: 20.12m locked.`);
      } else {
        alert(`Calibration insufficient: ${res.message || 'High reprojection error'}`);
      }
    })
    .catch(e => alert(`Calibration error: ${e}`));
  }

  function confirmLiveCalibration() {
    if (!state.liveLatestTelemetry) return;
    const det = state.liveLatestTelemetry.detection || state.liveLatestTelemetry;
    fetch('/api/calibration/confirm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(det)
    })
    .then(() => {
      elements.liveCalibPill.className = 'status-pill online';
      elements.liveCalibPill.textContent = 'Locked';
      alert('Live camera calibration locked for active bowling session.');
    });
  }

  function resetLiveCalibration() {
    state.liveCalibrated = false;
    elements.liveCalibPill.className = 'status-pill';
    elements.liveCalibPill.textContent = 'Pending';
    elements.liveReprojVal.textContent = '-- px';
    elements.liveReprojVal.className = 'telemetry-val';
    elements.liveSpeedNumber.textContent = '--.-';
    elements.liveMeasurementBadge.className = 'measurement-pill not-measured';
    elements.liveMeasurementBadge.textContent = 'NOT_MEASURED';
    fetch('/api/calibration/reset', { method: 'POST' });
  }

  // ============================================================================
  // MODE 2: INPUT VIDEO IMPLEMENTATION
  // ============================================================================
  function setupDragAndDrop() {
    const p = elements.uploadPanel;
    ['dragenter', 'dragover'].forEach(ev => {
      p.addEventListener(ev, (e) => {
        e.preventDefault();
        p.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach(ev => {
      p.addEventListener(ev, (e) => {
        e.preventDefault();
        p.classList.remove('dragover');
      });
    });
    p.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        uploadVideoFile(e.dataTransfer.files[0]);
      }
    });
  }

  function handleVideoFileSelect(e) {
    if (e.target.files && e.target.files[0]) {
      uploadVideoFile(e.target.files[0]);
    }
  }

  async function uploadVideoFile(file) {
    if (file.size > 500 * 1024 * 1024) {
      alert('File size exceeds the 500 MB limit.');
      return;
    }

    elements.browseVideoBtn.disabled = true;
    elements.browseVideoBtn.textContent = 'Uploading...';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/video/upload', {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        const data = await res.json();
        await setupLoadedVideo(data.file_path, data.metadata);
      } else {
        alert('Upload failed. Please check file format.');
      }
    } catch (e) {
      alert(`Upload error: ${e}`);
    } finally {
      elements.browseVideoBtn.disabled = false;
      elements.browseVideoBtn.textContent = '📁 Browse Computer';
    }
  }

  async function loadVideoByPath(videoPath) {
    try {
      const res = await fetch(`/api/video/metadata?video_path=${encodeURIComponent(videoPath)}`);
      if (res.ok) {
        const meta = await res.json();
        await setupLoadedVideo(videoPath, meta);
      } else {
        alert(`Video could not be opened: ${videoPath}`);
      }
    } catch (e) {
      alert(`Error loading video: ${e}`);
    }
  }

  async function setupLoadedVideo(videoPath, metadata) {
    state.currentVideoPath = videoPath;
    state.currentVideoHash = metadata.file_hash;
    state.currentVideoMeta = metadata;
    state.currentFrameIndex = 0;

    // Display Metadata
    elements.metaFilename.textContent = metadata.file_name;
    elements.metaResolution.textContent = `${metadata.width} x ${metadata.height}`;
    elements.metaFps.textContent = `${metadata.fps.toFixed(2)} FPS`;
    elements.metaDuration.textContent = `${metadata.duration_seconds.toFixed(2)}s (${metadata.total_frames} frames)`;
    elements.videoMetaVal.textContent = `${metadata.width}x${metadata.height} @ ${metadata.fps.toFixed(0)}fps`;

    // Configure Scrubber
    elements.frameSlider.max = Math.max(0, metadata.total_frames - 1);
    elements.frameSlider.value = 0;
    updateTimeReadout(0, metadata.total_frames, metadata.fps);

    // Reveal Workspace & Hide Results Dashboard until analysis is run
    elements.videoWorkspace.style.display = 'grid';
    elements.analysisResultsDashboard.style.display = 'none';
    elements.analysisProgressCard.style.display = 'none';

    // Auto-select best calibration frame
    await autoSelectBestCalibrationFrame();
  }

  async function autoSelectBestCalibrationFrame() {
    if (!state.currentVideoPath) return;
    try {
      const res = await fetch(`/api/video/best_calibration_frame?video_path=${encodeURIComponent(state.currentVideoPath)}`);
      if (res.ok) {
        const data = await res.json();
        const bestIdx = data.selected_frame_index || 0;
        state.calibrationFrameIndex = bestIdx;
        state.currentFrameIndex = bestIdx;
        elements.frameSlider.value = bestIdx;
        await loadAndDetectFrame(bestIdx);
      } else {
        await loadAndDetectFrame(0);
      }
    } catch (e) {
      await loadAndDetectFrame(0);
    }
  }

  async function loadAndDetectFrame(frameIndex) {
    if (!state.currentVideoPath) return;

    // 1. Fetch Frame Image
    const imgUrl = `/api/video/frame_image?video_path=${encodeURIComponent(state.currentVideoPath)}&frame_index=${frameIndex}`;
    const img = new Image();
    img.src = imgUrl;
    await img.decode();

    state.currentFrameImg = img;
    elements.videoCanvas.width = img.naturalWidth || state.currentVideoMeta.width;
    elements.videoCanvas.height = img.naturalHeight || state.currentVideoMeta.height;

    // 2. Run Stump Detection on Frame
    const formData = new FormData();
    formData.append('video_path', state.currentVideoPath);
    formData.append('frame_index', frameIndex);
    formData.append('confidence', elements.settingStumpsConf.value || '0.20');

    try {
      const detRes = await fetch('/api/video/detect_frame', {
        method: 'POST',
        body: formData
      });
      if (detRes.ok) {
        const det = await detRes.json();
        state.activeDetection = det;
        renderVideoFrameWithOverlays();
        updateVideoTelemetry(det);
      }
    } catch (e) {
      renderVideoFrameWithOverlays();
    }

    updateTimeReadout(frameIndex, state.currentVideoMeta.total_frames, state.currentVideoMeta.fps);
  }

  function renderVideoFrameWithOverlays() {
    const canvas = elements.videoCanvas;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    // Apply Zoom & Pan
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.zoom, state.zoom);

    // Draw background video frame
    if (state.currentFrameImg) {
      ctx.drawImage(state.currentFrameImg, 0, 0, canvas.width, canvas.height);
    }

    // Draw DRS Geometry & Landmarks
    if (state.activeDetection) {
      renderDrsOverlays(ctx, state.activeDetection, canvas.width, canvas.height);
    }

    // Draw Manual Landmark Correction Handles if active
    if (state.isManualCorrectionActive) {
      renderManualCorrectionHandles(ctx);
    }

    ctx.restore();
  }

  function updateVideoTelemetry(det) {
    const hasNear = !!det.near_wicket;
    const hasFar = !!det.far_wicket;
    const hasDrs = !!(det.drs_line && det.drs_line.is_valid);

    const stumpCount = (hasNear ? 1 : 0) + (hasFar ? 1 : 0);
    elements.videoStumpCountVal.textContent = `${stumpCount} / 2`;

    if (hasNear && hasFar) {
      elements.videoStumpsBadge.className = 'hud-badge success';
      elements.videoStumpsText.textContent = 'Both Stumps Found';
    } else if (hasNear || hasFar) {
      elements.videoStumpsBadge.className = 'hud-badge warning';
      elements.videoStumpsText.textContent = hasNear ? 'Near Stumps Only' : 'Far Stumps Only';
    } else {
      elements.videoStumpsBadge.className = 'hud-badge danger';
      elements.videoStumpsText.textContent = 'Searching Stumps';
    }

    if (hasDrs) {
      elements.videoDrsBadge.style.display = 'inline-flex';
      elements.videoDrsText.textContent = `20.12m Reference Line`;
      elements.videoDrsVal.textContent = 'Valid (20.12m)';
      elements.videoDrsVal.className = 'telemetry-val success';
      elements.videoDrsLengthVal.textContent = `${det.drs_line.pixel_length.toFixed(1)} px (${det.drs_line.angle_deg.toFixed(1)}°)`;
      elements.confirmVideoCalibBtn.disabled = false;
      elements.startAnalysisBtn.disabled = false;
    } else {
      elements.videoDrsBadge.style.display = 'none';
      elements.videoDrsVal.textContent = 'Insufficient';
      elements.videoDrsVal.className = 'telemetry-val danger';
      elements.videoDrsLengthVal.textContent = '-- px';
      elements.confirmVideoCalibBtn.disabled = true;
    }

    if (det.pitch_bbox) {
      elements.videoPitchCornersVal.textContent = 'Boundaries Detected';
      elements.videoPitchCornersVal.className = 'telemetry-val success';
    } else {
      elements.videoPitchCornersVal.textContent = 'Unavailable (Stumps Primary)';
      elements.videoPitchCornersVal.className = 'telemetry-val';
    }
  }

  // ============================================================================
  // Canvas DRS Geometry Renderer (Shared between Live and Video Modes)
  // ============================================================================
  function renderDrsOverlays(ctx, det, width, height) {
    if (!det) return;

    // 1. Draw Pitch Boundary if available
    if (det.pitch_bbox) {
      const [px1, py1, px2, py2] = det.pitch_bbox;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 6]);
      ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);
      ctx.setLineDash([]);
    }

    // 2. Draw Stump Bounding Boxes
    [det.near_wicket, det.far_wicket].forEach(w => {
      if (!w) return;
      const isNear = w.stump_type === 'near';
      const [x1, y1, x2, y2] = w.bbox;
      const boxColor = isNear ? '#10b981' : '#3b82f6';

      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Label
      ctx.fillStyle = boxColor;
      ctx.font = 'bold 12px Inter, sans-serif';
      const label = `${isNear ? 'NEAR WICKET' : 'FAR WICKET'} (${(w.confidence * 100).toFixed(0)}%)`;
      ctx.fillText(label, x1, y1 - 6);

      // Stump Top Landmark (Cyan circle)
      ctx.fillStyle = '#06b6d4';
      ctx.beginPath();
      ctx.arc(w.top_center[0], w.top_center[1], 4, 0, 2 * Math.PI);
      ctx.fill();

      // Stump Base Landmark (Yellow circle)
      ctx.fillStyle = '#facc15';
      ctx.beginPath();
      ctx.arc(w.base_center[0], w.base_center[1], 4, 0, 2 * Math.PI);
      ctx.fill();

      // Base Centre A / B (Red ring)
      ctx.strokeStyle = '#ef4444';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(w.base_center[0], w.base_center[1], 8, 0, 2 * Math.PI);
      ctx.stroke();
    });

    // 3. Draw Automatically Generated 20.12m DRS Line
    if (det.drs_line && det.drs_line.near_point && det.drs_line.far_point) {
      const p1 = det.drs_line.near_point;
      const p2 = det.drs_line.far_point;

      // Glow effect
      ctx.shadowColor = '#06b6d4';
      ctx.shadowBlur = 10;
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.stroke();

      // Reset shadow
      ctx.shadowBlur = 0;

      // Inner bright core
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.stroke();

      // Midpoint distance tag
      const mid = det.drs_line.midpoint || [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2];
      ctx.fillStyle = 'rgba(10, 25, 47, 0.9)';
      ctx.fillRect(mid[0] - 55, mid[1] - 12, 110, 24);
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 1;
      ctx.strokeRect(mid[0] - 55, mid[1] - 12, 110, 24);

      ctx.fillStyle = '#06b6d4';
      ctx.font = 'bold 11px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('20.12m DRS LINE', mid[0], mid[1]);
      ctx.textAlign = 'start';
      ctx.textBaseline = 'alphabetic';
    }
  }

  // ============================================================================
  // Manual Landmark Fallback Mode
  // ============================================================================
  function toggleManualCorrectionMode() {
    state.isManualCorrectionActive = !state.isManualCorrectionActive;
    if (state.isManualCorrectionActive) {
      elements.toggleCorrectionBtn.textContent = '✓ Exit Correction Mode';
      elements.toggleCorrectionBtn.classList.add('btn-primary');

      // Populate initial points from active detection if available
      if (state.activeDetection) {
        const d = state.activeDetection;
        if (d.near_wicket) {
          state.manualPoints.near_top = [...d.near_wicket.top_center];
          state.manualPoints.near_base = [...d.near_wicket.base_center];
        }
        if (d.far_wicket) {
          state.manualPoints.far_top = [...d.far_wicket.top_center];
          state.manualPoints.far_base = [...d.far_wicket.base_center];
        }
      }
    } else {
      elements.toggleCorrectionBtn.textContent = '✏ Manual Landmark Correction';
      elements.toggleCorrectionBtn.classList.remove('btn-primary');
    }
    renderVideoFrameWithOverlays();
  }

  function renderManualCorrectionHandles(ctx) {
    CALIBRATION_KEYS.forEach(key => {
      const pt = state.manualPoints[key];
      if (!pt) return;
      const isSelected = state.activePointKey === key;

      ctx.fillStyle = isSelected ? '#f43f5e' : '#3b82f6';
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], isSelected ? 7 : 5, 0, 2 * Math.PI);
      ctx.fill();

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 10px Inter, sans-serif';
      ctx.fillText(key, pt[0] + 8, pt[1] + 4);
    });
  }

  // ============================================================================
  // Video Calibration & Background Analysis
  // ============================================================================
  async function triggerVideoAutoCalibrate() {
    if (!state.activeDetection || !state.currentFrameImg) {
      alert('Please load a video frame with visible wickets first.');
      return;
    }

    const det = state.activeDetection;
    const points = state.isManualCorrectionActive ? state.manualPoints : {
      near_base: det.near_wicket ? det.near_wicket.base_center : null,
      near_top: det.near_wicket ? det.near_wicket.top_center : null,
      far_base: det.far_wicket ? det.far_wicket.base_center : null,
      far_top: det.far_wicket ? det.far_wicket.top_center : null
    };

    try {
      const res = await fetch('/api/calibrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_points: points,
          image_width: elements.videoCanvas.width,
          image_height: elements.videoCanvas.height,
          video_hash: state.currentVideoHash,
          video_path: state.currentVideoPath
        })
      });

      if (res.ok) {
        const calib = await res.json();
        if (calib.success) {
          state.activeCalibration = calib;
          elements.videoCalibStatusPill.className = 'status-pill online';
          elements.videoCalibStatusPill.textContent = 'Calibrated';
          elements.videoReprojVal.textContent = `${calib.reprojection_error_px.toFixed(2)} px`;
          elements.videoReprojVal.className = 'telemetry-val success';
          elements.confirmVideoCalibBtn.disabled = false;
          elements.startAnalysisBtn.disabled = false;
          alert(`Auto Calibration Successful!\nReprojection RMSE: ${calib.reprojection_error_px.toFixed(2)}px\nLongitudinal DRS axis: 20.12m locked.`);
        } else {
          alert(`Calibration rejected: ${calib.message}`);
        }
      }
    } catch (e) {
      alert(`Calibration error: ${e}`);
    }
  }

  async function confirmVideoCalibration() {
    if (!state.activeCalibration) return;
    try {
      const res = await fetch('/api/calibration/confirm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state.activeCalibration)
      });
      if (res.ok) {
        elements.videoCalibStatusPill.className = 'status-pill online';
        elements.videoCalibStatusPill.textContent = 'Locked (20.12m)';
        elements.startAnalysisBtn.disabled = false;
        alert('Calibration confirmed and locked to video hash!');
      }
    } catch (e) {
      alert(`Error locking calibration: ${e}`);
    }
  }

  async function startFullVideoAnalysis() {
    if (!state.currentVideoPath) {
      alert('Please upload or select a video first.');
      return;
    }

    elements.startAnalysisBtn.disabled = true;
    elements.analysisProgressCard.style.display = 'flex';
    elements.analysisResultsDashboard.style.display = 'none';

    try {
      const res = await fetch('/api/video/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_path: state.currentVideoPath,
          custom_calibration: state.activeCalibration
        })
      });

      if (res.ok) {
        const data = await res.json();
        state.analysisTaskId = data.task_id;
        pollAnalysisProgress(data.task_id);
      } else {
        alert('Failed to launch analysis task.');
        elements.startAnalysisBtn.disabled = false;
      }
    } catch (e) {
      alert(`Error starting analysis: ${e}`);
      elements.startAnalysisBtn.disabled = false;
    }
  }

  function pollAnalysisProgress(taskId) {
    if (state.analysisPollInterval) clearInterval(state.analysisPollInterval);

    state.analysisPollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/video/analysis_status/${taskId}`);
        if (!res.ok) return;

        const data = await res.json();
        elements.progressPctText.textContent = `${data.progress_pct.toFixed(0)}%`;
        elements.progressStageText.textContent = data.stage;
        elements.progressFill.style.width = `${data.progress_pct}%`;

        if (data.status === 'completed') {
          clearInterval(state.analysisPollInterval);
          state.analysisPollInterval = null;
          await fetchAnalysisResults(taskId);
        } else if (data.status === 'failed') {
          clearInterval(state.analysisPollInterval);
          state.analysisPollInterval = null;
          elements.startAnalysisBtn.disabled = false;
          alert(`Analysis Error: ${data.error || 'Pipeline execution failed.'}`);
        }
      } catch (e) {
        console.warn('Error polling progress:', e);
      }
    }, 400);
  }

  async function fetchAnalysisResults(taskId) {
    try {
      const res = await fetch(`/api/video/analysis_result/${taskId}`);
      if (!res.ok) return;

      const data = await res.json();
      state.analysisResult = data;

      // Populate Results Display
      elements.analysisProgressCard.style.display = 'none';
      elements.analysisResultsDashboard.style.display = 'flex';
      elements.startAnalysisBtn.disabled = false;

      const isMeasured = data.status === 'MEASURED';
      elements.resultsMeasurementBadge.className = `measurement-pill ${isMeasured ? 'measured' : 'not-measured'}`;
      elements.resultsMeasurementBadge.textContent = data.status;

      elements.resultReleaseSpeed.textContent = data.release_speed_kmh ? data.release_speed_kmh.toFixed(1) : '--.-';
      elements.resultUncertainty.textContent = data.uncertainty_kmh ? `±${data.uncertainty_kmh.toFixed(1)} km/h uncertainty` : 'Uncertainty unconstrained';
      elements.resultC1C2Speed.textContent = data.c1_c2_speed_kmh ? `${data.c1_c2_speed_kmh.toFixed(1)} km/h` : '--.- km/h';
      elements.resultQcScore.textContent = `${data.qc_score || 0} / 100`;
      elements.resultStatusVal.textContent = data.status;
      elements.resultRejectionVal.textContent = data.rejection_reason || 'None';

      // Attach Video Streams
      if (data.final_video) {
        elements.finalVideoPlayer.src = `/api/video/stream?video_path=${encodeURIComponent(data.final_video)}`;
        elements.dlFinalVideo.href = `/api/video/download/final/${taskId}`;
      }
      if (data.diagnostic_video) {
        elements.diagVideoPlayer.src = `/api/video/stream?video_path=${encodeURIComponent(data.diagnostic_video)}`;
        elements.dlDiagVideo.href = `/api/video/download/diagnostic/${taskId}`;
      }

      // JSON and Excel downloads
      elements.dlJsonReport.href = `/api/video/download/json/${taskId}`;
      elements.dlExcelReport.href = `/api/video/download/excel/${taskId}`;

      // Load JSON Report text
      if (data.json_report) {
        fetch(elements.dlJsonReport.href)
          .then(r => r.json())
          .then(j => {
            elements.jsonReportText.textContent = JSON.stringify(j, null, 2);
          })
          .catch(() => {});
      }

    } catch (e) {
      console.error('Error fetching analysis results:', e);
    }
  }

  // ============================================================================
  // Video Scrubber & Playback Helpers
  // ============================================================================
  function handleSliderChange(e) {
    const idx = parseInt(e.target.value, 10);
    state.currentFrameIndex = idx;
    loadAndDetectFrame(idx);
  }

  function stepFrame(delta) {
    if (!state.currentVideoMeta) return;
    const nextIdx = Math.max(0, Math.min(state.currentVideoMeta.total_frames - 1, state.currentFrameIndex + delta));
    state.currentFrameIndex = nextIdx;
    elements.frameSlider.value = nextIdx;
    loadAndDetectFrame(nextIdx);
  }

  function togglePlayPause() {
    state.isPlaying = !state.isPlaying;
    elements.playPauseBtn.textContent = state.isPlaying ? '⏸ Pause' : '▶ Play';

    if (state.isPlaying) {
      state.playInterval = setInterval(() => {
        if (!state.currentVideoMeta) return;
        if (state.currentFrameIndex >= state.currentVideoMeta.total_frames - 1) {
          state.currentFrameIndex = 0;
        } else {
          state.currentFrameIndex += 1;
        }
        elements.frameSlider.value = state.currentFrameIndex;
        loadAndDetectFrame(state.currentFrameIndex);
      }, 100);
    } else {
      if (state.playInterval) clearInterval(state.playInterval);
      state.playInterval = null;
    }
  }

  function updateTimeReadout(frameIdx, totalFrames, fps) {
    const curSec = fps > 0 ? (frameIdx / fps) : 0;
    const m = Math.floor(curSec / 60);
    const s = Math.floor(curSec % 60);
    const ms = Math.floor((curSec % 1) * 10);
    elements.timeReadout.textContent = `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${ms} [${frameIdx}/${totalFrames}]`;
  }

  // ============================================================================
  // Canvas Pan & Zoom
  // ============================================================================
  function applyZoom(factor) {
    state.zoom = Math.max(0.5, Math.min(5.0, state.zoom * factor));
    renderVideoFrameWithOverlays();
  }

  function resetZoomAndPan() {
    state.zoom = 1.0;
    state.panX = 0;
    state.panY = 0;
    renderVideoFrameWithOverlays();
  }

  function setupCanvasInteractivity(canvas) {
    canvas.addEventListener('mousedown', (e) => {
      const rect = canvas.getBoundingClientRect();
      const clickX = (e.clientX - rect.left - state.panX) / state.zoom;
      const clickY = (e.clientY - rect.top - state.panY) / state.zoom;

      // In manual correction mode, check if clicking near a handle
      if (state.isManualCorrectionActive) {
        let clickedKey = null;
        CALIBRATION_KEYS.forEach(key => {
          const pt = state.manualPoints[key];
          if (pt && Math.hypot(pt[0] - clickX, pt[1] - clickY) < 14) {
            clickedKey = key;
          }
        });
        if (clickedKey) {
          state.activePointKey = clickedKey;
          renderVideoFrameWithOverlays();
          return;
        } else if (state.activePointKey) {
          // Set point coordinate
          state.manualPoints[state.activePointKey] = [Math.round(clickX), Math.round(clickY)];
          renderVideoFrameWithOverlays();
          return;
        }
      }

      state.isDragging = true;
      state.dragStartX = e.clientX - state.panX;
      state.dragStartY = e.clientY - state.panY;
    });

    window.addEventListener('mousemove', (e) => {
      if (!state.isDragging) return;
      state.panX = e.clientX - state.dragStartX;
      state.panY = e.clientY - state.dragStartY;
      renderVideoFrameWithOverlays();
    });

    window.addEventListener('mouseup', () => {
      state.isDragging = false;
    });
  }

  // Bootstrap
  window.addEventListener('DOMContentLoaded', initApp);

})();
