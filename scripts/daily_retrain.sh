#!/bin/bash
# Daily retraining script to be run as a cron job
# Updates data, retrains model, and uploads to S3

# Don't use set -e so we can continue processing other tickers if one fails
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="/var/log/stock-trend"
LOG_FILE="$LOG_DIR/retrain.log"
ERROR_LOG="$LOG_DIR/retrain.error.log"

mkdir -p $LOG_DIR

cd $PROJECT_DIR

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "$(date): ERROR - Virtual environment not found. Creating..." >> $ERROR_LOG
    python3 -m venv venv
fi

source venv/bin/activate

# Ensure dependencies are installed
pip install -q -r requirements.txt

echo "$(date): Starting daily retrain..." >> $LOG_FILE

# Set S3 bucket (should be set as environment variable or in config)
S3_BUCKET=${S3_BUCKET:-""}
S3_MODEL_KEY=${S3_MODEL_KEY:-"models/production.joblib"}

# Calculate date range (2 years of data for training)
END_DATE=$(date +%Y-%m-%d)
START_DATE=$(date -d "2 years ago" +%Y-%m-%d 2>/dev/null || date -v-2y +%Y-%m-%d)

# Download data for tickers in tickers.txt (if exists)
if [ -f "tickers.txt" ]; then
    echo "$(date): Downloading data for tickers..." >> $LOG_FILE
    
    # Download data for each ticker
    while IFS= read -r ticker || [ -n "$ticker" ]; do
        # Skip empty lines and comments
        [[ -z "$ticker" || "$ticker" =~ ^#.*$ ]] && continue
        
        echo "$(date): Downloading data for $ticker..." >> $LOG_FILE
        
        python3 src/data_fetch.py \
            --ticker "$ticker" \
            --start "$START_DATE" \
            --end "$END_DATE" \
            --out data/ \
            $([ -n "$S3_BUCKET" ] && echo "--s3-bucket $S3_BUCKET --s3-prefix raw/") \
            2>&1 | tee -a $LOG_FILE
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "$(date): Successfully downloaded data for $ticker" >> $LOG_FILE
            
            # Find the downloaded CSV file
            DATA_FILE=$(find data/ -name "${ticker}-*.csv" -type f | sort -r | head -1)
            
            if [ -n "$DATA_FILE" ] && [ -f "$DATA_FILE" ]; then
                echo "$(date): Training model for $ticker using $DATA_FILE..." >> $LOG_FILE
                
                # Train model
                python3 src/train.py \
                    --data "$DATA_FILE" \
                    --out models/ \
                    --n-estimators 300 \
                    --no-tune \
                    $([ -n "$S3_BUCKET" ] && echo "--s3-bucket $S3_BUCKET --s3-key models/${ticker}_production.joblib") \
                    2>&1 | tee -a $LOG_FILE
                
                if [ ${PIPESTATUS[0]} -eq 0 ]; then
                    echo "$(date): Successfully trained model for $ticker" >> $LOG_FILE
                    
                    # If this is the first ticker or a designated production ticker, upload as production model
                    if [ "$ticker" = "AAPL" ] || [ -z "$PRODUCTION_TICKER" ]; then
                        PRODUCTION_TICKER="$ticker"
                        MODEL_FILE=$(find models/ -name "${ticker}_model.joblib" -type f | sort -r | head -1)
                        
                        if [ -n "$MODEL_FILE" ] && [ -n "$S3_BUCKET" ]; then
                            echo "$(date): Uploading $ticker model as production model..." >> $LOG_FILE
                            python3 -c "
import sys
sys.path.insert(0, 'src')
from model_utils import upload_model_to_s3
import os
upload_model_to_s3('$MODEL_FILE', '$S3_MODEL_KEY', '$S3_BUCKET')
" 2>&1 | tee -a $LOG_FILE
                        fi
                    fi
                else
                    echo "$(date): ERROR - Failed to train model for $ticker" >> $ERROR_LOG
                fi
            else
                echo "$(date): ERROR - Data file not found for $ticker" >> $ERROR_LOG
            fi
        else
            echo "$(date): ERROR - Failed to download data for $ticker" >> $ERROR_LOG
        fi
        
    done < tickers.txt
else
    echo "$(date): WARNING - tickers.txt not found. Skipping retraining." >> $LOG_FILE
fi

echo "$(date): Daily retrain complete." >> $LOG_FILE

