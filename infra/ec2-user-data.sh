#!/bin/bash
# EC2 User Data script for bootstrapping the stock-trend-mvp application
# This script runs once when the EC2 instance is first launched

set -e

echo "Starting EC2 bootstrap for stock-trend-mvp..."

# Update system
apt-get update -y

# Install dependencies
apt-get install -y python3 python3-pip python3-venv git nginx

# Create application directory
APP_DIR="/home/ubuntu/stock-trend-mvp"
mkdir -p $APP_DIR
cd $APP_DIR

# Clone repository (update with your repo URL)
# git clone https://github.com/Shohruz7/Stock-Model.git .

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create log directory
mkdir -p /var/log/stock-trend
chown ubuntu:ubuntu /var/log/stock-trend

# Copy systemd service file
cp infra/streamlit.service /etc/systemd/system/streamlit-app.service

# Configure nginx (if nginx config exists)
if [ -f "infra/nginx.conf" ]; then
    cp infra/nginx.conf /etc/nginx/sites-available/stock-trend
    ln -sf /etc/nginx/sites-available/stock-trend /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default  # Remove default site
    nginx -t && systemctl restart nginx
fi

# Enable and start service
systemctl daemon-reload
systemctl enable streamlit-app
systemctl start streamlit-app

# Wait a moment for service to start
sleep 5

# Check service status
systemctl status streamlit-app --no-pager

echo "Bootstrap complete!"
echo "Application should be available at http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8501"

