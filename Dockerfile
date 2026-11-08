# ==============================================================================
# Cricket Ball Speed Analyzer - Production Dockerfile
# Optimized for DigitalOcean (Droplet & App Platform)
# ==============================================================================

FROM python:3.11-slim-bookworm

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000 \
    HOST=0.0.0.0 \
    PYTHONPATH=/app:/app/cricket_ball_speed:/app/backend

# Install essential system dependencies (OpenCV headless libraries, FFmpeg, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first for efficient layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY --chown=appuser:appuser . .

# Ensure storage directories exist with correct permissions
RUN mkdir -p /app/storage/uploads \
             /app/storage/results \
             /app/storage/calibration \
             /app/storage/logs \
             /app/storage/tasks \
             /app/cricket_ball_speed/storage/temp \
    && chown -R appuser:appuser /app/storage /app/cricket_ball_speed/storage

# Switch to non-root user
USER appuser

# Healthcheck for container orchestration and DigitalOcean readiness probes
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE 8000

# Start server using production entrypoint
CMD ["python", "main.py", "server", "--host", "0.0.0.0", "--port", "8000"]
