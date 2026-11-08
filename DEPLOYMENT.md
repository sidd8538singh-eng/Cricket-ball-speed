# Production Deployment Guide: DigitalOcean

Complete senior-engineer guide for deploying the **Cricket Ball Speed Analyzer & DRS Vision Platform** to DigitalOcean.

---

## Table of Contents
1. [Recommended Architecture & Droplet Sizing](#1-recommended-architecture--droplet-sizing)
2. [Option A: DigitalOcean Droplet via Docker Compose (Recommended)](#2-option-a-digitalocean-droplet-via-docker-compose-recommended)
3. [Option B: DigitalOcean Droplet Native Service (Systemd + Nginx)](#3-option-b-digitalocean-droplet-native-service-systemd--nginx)
4. [Option C: DigitalOcean App Platform (PaaS)](#4-option-c-digitalocean-app-platform-paas)
5. [Custom Domain & Let's Encrypt Free SSL (HTTPS/WSS)](#5-custom-domain--lets-encrypt-free-ssl-httpswss)
6. [Nginx Reverse Proxy & 500MB Video Upload Configuration](#6-nginx-reverse-proxy--500mb-video-upload-configuration)
7. [Healthcheck, Logging & Monitoring](#7-healthcheck-logging--monitoring)
8. [Maintenance, Backups & Upgrades](#8-maintenance-backups--upgrades)

---

## 1. Recommended Architecture & Droplet Sizing

### Architecture
```
[User Browser / Live Camera]
           │
           ▼  HTTPS (443) / WSS (WebSocket)
    [Nginx Reverse Proxy] (500MB upload limit, gzip, SSL termination)
           │
           ▼  HTTP (8000) / WS (/ws/live)
 [FastAPI + Uvicorn Application] (Docker container or native Systemd)
     ├── LiveCameraService (WebSocket telemetry)
     ├── VideoService (Multipart streaming & asynchronous processing)
     └── CricketSpeedPipeline (3D Kalman & Trajectory Fitting)
           │
     [Persistent Storage Mount] (/storage: uploads, results, calibrations)
```

### DigitalOcean Droplet Sizing Recommendations
- **Minimum (Testing / Light Practice):**
  - **Droplet Type:** Basic Regular or Premium Intel/AMD
  - **Specs:** 2 vCPU, 4 GB RAM, 80 GB NVMe SSD (\$24–\$28/mo)
  - Suitable for CPU inference with standard 720p/1080p recorded video deliveries.
- **Recommended (Production Net Sessions & Multiple Deliveries):**
  - **Droplet Type:** General Purpose Dedicated CPU or Memory-Optimized
  - **Specs:** 4 vCPU, 8 GB RAM, 160 GB SSD (\$48–\$60/mo)
  - Handles concurrent uploads, high-framerate multi-frame decoding, and live WebSocket streams.
- **High-Speed Real-time net/broadcast:**
  - DigitalOcean GPU Droplet (1x NVIDIA H100 or A10G) if real-time 120 FPS continuous inference is required.

---

## 2. Option A: DigitalOcean Droplet via Docker Compose (Recommended)

Docker Compose provides an isolated, reproducible environment where all computer vision, FFmpeg, and Python libraries are containerized with zero host dependency conflicts.

### Step 1: Create Droplet
1. Log in to [cloud.digitalocean.com](https://cloud.digitalocean.com).
2. Click **Create** > **Droplets**.
3. Choose **Ubuntu 22.04 LTS (x64)** or **Ubuntu 24.04 LTS (x64)**.
4. Select **4 GB RAM / 2 vCPU** (minimum) or **8 GB RAM / 4 vCPU** (recommended).
5. Choose your datacenter region (e.g. Bangalore `BLR1` for India, Frankfurt `FRA1` for Europe, NYC for US).
6. Select **SSH Key** authentication.
7. Click **Create Droplet**.

### Step 2: SSH into your Droplet
```bash
ssh root@<YOUR_DROPLET_IP>
```

### Step 3: Clone Repository
```bash
cd /opt
git clone https://github.com/sidd8538singh-eng/Cricket-ball-speed.git
cd Cricket-ball-speed
```

### Step 4: Run Automated Deployment Script
We have provided an automated deployment script that configures Docker, firewall, permissions, and launches the container:
```bash
chmod +x scripts/deploy_digitalocean.sh
sudo bash scripts/deploy_digitalocean.sh
```

### Step 5: Verify Deployment
Check running container:
```bash
docker compose ps
```
Check health endpoint:
```bash
curl http://localhost:8000/health
```
Output:
```json
{"status":"healthy","service":"cricket-ball-speed","version":"2.0.0"}
```
You can now open `http://<YOUR_DROPLET_IP>:8000` in your web browser!

---

## 3. Option B: DigitalOcean Droplet Native Service (Systemd + Nginx)

If you prefer running directly on Ubuntu without Docker:

### Step 1: Install System Packages
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv ffmpeg libgl1 libglib2.0-0 nginx curl git
```

### Step 2: Clone and Setup Virtual Environment
```bash
sudo mkdir -p /var/www
cd /var/www
sudo git clone https://github.com/sidd8538singh-eng/Cricket-ball-speed.git cricket-ball-speed
cd cricket-ball-speed

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Setup Systemd Service
```bash
sudo cp scripts/systemd/cricket-speed.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable cricket-speed
sudo systemctl start cricket-speed
sudo systemctl status cricket-speed
```

---

## 4. Option C: DigitalOcean App Platform (PaaS)

If you want a fully managed platform without maintaining Linux servers:
1. Go to **Apps** in DigitalOcean console > Click **Create App**.
2. Connect your GitHub account and select repository `sidd8538singh-eng/Cricket-ball-speed`.
3. DigitalOcean will automatically detect the **`Dockerfile`**.
4. Configure resource size: **Pro (2 vCPU, 4GB RAM minimum)**.
5. In **Environment Variables**, set:
   - `PORT`: `8000`
   - `HOST`: `0.0.0.0`
   - `CORS_ORIGINS`: `*`
6. In **Health Checks**, set HTTP path to `/health`.
7. Click **Deploy**.

---

## 5. Custom Domain & Let's Encrypt Free SSL (HTTPS/WSS)

Modern browsers require **HTTPS** and **WSS** (Secure WebSockets) to access webcams via `navigator.mediaDevices.getUserMedia()`. Setting up SSL is essential.

### Step 1: Point DNS Record to Droplet IP
In your domain registrar (GoDaddy, Namecheap, Cloudflare, etc.):
- Add an **A Record**:
  - Name: `cricket` (or `@` for root domain)
  - Value: `<YOUR_DROPLET_IP>`
  - TTL: 300 seconds

### Step 2: Configure Nginx
Copy the pre-configured Nginx site file:
```bash
sudo cp /opt/Cricket-ball-speed/nginx/cricket-speed.conf /etc/nginx/sites-available/cricket-speed
```
Edit the file to put your real domain name:
```bash
sudo nano /etc/nginx/sites-available/cricket-speed
```
Change `server_name _;` to:
```nginx
server_name cricket.yourdomain.com;
```
Enable the site and test Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/cricket-speed /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

### Step 3: Obtain Free SSL Certificate with Certbot
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d cricket.yourdomain.com
```
Certbot will automatically install the certificate, configure HTTPS on port 443, and set up automatic 90-day renewal.

Now visit `https://cricket.yourdomain.com` in your browser!

---

## 6. Nginx Reverse Proxy & 500MB Video Upload Configuration

The included Nginx configuration handles large video uploads and live WebSocket streaming:

```nginx
# 1. 500MB Video Ingestion
client_max_body_size 500M;
client_body_buffer_size 128k;

# 2. WebSocket Support for Live Camera (/ws/live)
proxy_http_version 1.1;
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";

# 3. 10-Minute Timeout for Trajectory Processing
proxy_connect_timeout 60s;
proxy_send_timeout 600s;
proxy_read_timeout 600s;
```

---

## 7. Healthcheck, Logging & Monitoring

### Healthcheck URL
- `GET /health` or `GET /api/health`
- Returns: `{"status": "healthy", "service": "cricket-ball-speed", "version": "2.0.0"}`

### Live System Logs
If running with Docker Compose:
```bash
# Follow logs in real-time
docker compose logs -f

# Follow last 100 lines
docker compose logs --tail=100 -f
```
If running with Native Systemd:
```bash
sudo journalctl -u cricket-speed -f
```

### Application Result Logs
Processed delivery logs are kept in:
- `storage/logs/`
- Output videos & JSON reports: `storage/results/`

---

## 8. Maintenance, Backups & Upgrades

### Updating to the Latest Code
To pull the latest updates from GitHub and redeploy:
```bash
cd /opt/Cricket-ball-speed
git pull origin main

# Rebuild and restart container
docker compose build
docker compose up -d
```

### Backing Up Deliveries and Calibrations
All delivery results, audit trails, and locked calibrations live in the `./storage/` folder.
To backup:
```bash
tar -czvf cricket_backup_$(date +%F).tar.gz storage/
```
You can transfer the backup archive using `scp`:
```bash
scp root@<YOUR_DROPLET_IP>:/opt/Cricket-ball-speed/cricket_backup_*.tar.gz ~/Downloads/
```
