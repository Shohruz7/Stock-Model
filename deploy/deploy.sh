#!/bin/bash
# Deployment script for updating the application on EC2

set -e

EC2_IP=$1

if [ -z "$EC2_IP" ]; then
    echo "Usage: ./deploy/deploy.sh EC2_IP"
    exit 1
fi

echo "Deploying to EC2 instance at $EC2_IP..."

ssh ubuntu@$EC2_IP << 'EOF'
    cd ~/stock-trend-mvp
    git pull
    source venv/bin/activate
    pip install -r requirements.txt
    sudo systemctl restart streamlit-app
    echo "Deployment complete!"
EOF

echo "Done!"

