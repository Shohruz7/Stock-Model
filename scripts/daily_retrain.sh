#!/bin/bash
# Daily retraining script to be run as a cron job
# Updates data, retrains model, and uploads to S3

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="/var/log/stock-trend"
LOG_FILE="$LOG_DIR/retrain.log"

mkdir -p $LOG_DIR

cd $PROJECT_DIR
source venv/bin/activate

echo "$(date): Starting daily retrain..." >> $LOG_FILE

# Set S3 bucket (should be set as environment variable or in config)
S3_BUCKET=${S3_BUCKET:-"your-bucket-name"}

# Calculate date range (last month to today)
END_DATE=$(date +%Y-%m-%d)
START_DATE=$(date -d "1 month ago" +%Y-%m-%d 2>/dev/null || date -v-1m +%Y-%m-%d)

# Download data for tickers in tickers.txt (if exists)
if [ -f "tickers.txt" ]; then
    python3 src/data_fetch.py \
        --tickers-file tickers.txt \
        --start $START_DATE \
        --end $END_DATE \
        --out data/ \
        --s3-bucket $S3_BUCKET \
        --s3-prefix raw/ 2>&1 | tee -a $LOG_FILE
fi

# Train model (implementation will be added)
# python3 src/train.py --data data/ --out models/ --s3-bucket $S3_BUCKET 2>&1 | tee -a $LOG_FILE

echo "$(date): Daily retrain complete." >> $LOG_FILE

