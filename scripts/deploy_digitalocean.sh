#!/bin/bash
# ==============================================================================
# One-Click Production Deployment Script for DigitalOcean Droplet (Ubuntu 22.04 / 24.04)
# Usage: sudo bash scripts/deploy_digitalocean.sh
# ==============================================================================

set -e

echo "======================================================================"
echo "DEPLOYING CRICKET BALL SPEED ANALYZER TO DIGITALOCEAN"
echo "======================================================================"

# 1. Update system packages
echo "[1/6] Updating system packages..."
apt-get update && apt-get upgrade -y
apt-get install -y curl git ufw ca-certificates gnupg lsb-release

# 2. Install Docker if not already installed
if ! command -v docker &> /dev/null; then
    echo "[2/6] Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    systemctl enable docker
    systemctl start docker
else
    echo "[2/6] Docker is already installed."
fi

# 3. Install Docker Compose plugin if missing
if ! docker compose version &> /dev/null; then
    echo "[3/6] Installing Docker Compose plugin..."
    apt-get install -y docker-compose-plugin
else
    echo "[3/6] Docker Compose plugin is already installed."
fi

# 4. Configure UFW Firewall
echo "[4/6] Configuring Firewall (UFW)..."
ufw allow 22/tcp || true
ufw allow 80/tcp || true
ufw allow 443/tcp || true
ufw allow 8000/tcp || true
ufw --force enable || true

# 5. Create storage directories with permissions
echo "[5/6] Creating persistent storage directories..."
mkdir -p storage/uploads storage/results storage/calibration storage/logs storage/tasks cricket_ball_speed/storage/temp
chmod -R 775 storage cricket_ball_speed/storage

# 6. Build and launch container
echo "[6/6] Building and starting Docker container..."
docker compose down || true
docker compose build --pull
docker compose up -d

echo "======================================================================"
echo "DEPLOYMENT COMPLETE!"
echo "Server is running at: http://$(curl -s ifconfig.me):8000"
echo "Healthcheck:         http://$(curl -s ifconfig.me):8000/health"
echo "API Docs:            http://$(curl -s ifconfig.me):8000/docs"
echo "======================================================================"
echo "Check container status: docker compose ps"
echo "View live logs:         docker compose logs -f"
echo "======================================================================"
