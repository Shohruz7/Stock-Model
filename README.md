# Stock Trend Predictor

A model that predicts next-day stock price direction using yfinance data, scikit-learn models, and Streamlit UI, deployed on AWS EC2.

## Overview

- **Data Source**: yfinance (daily OHLCV)
- **Model**: RandomForestClassifier for binary classification (up/down)
- **Storage**: AWS S3 (raw/, processed/, models/, metadata/)
- **Deployment**: Single EC2 instance with systemd service
- **Retraining**: Daily cron job updates production model

## Repository Structure

```
stock-trend/
├── app/
│   ├── streamlit_app.py      # Streamlit UI
│   └── predict.py            # Prediction utilities
├── src/
│   ├── data_fetch.py         # yfinance downloader + S3 upload
│   ├── features.py           # Feature engineering
│   ├── train.py              # Model training script
│   ├── model_utils.py        # Model persistence utilities
│   └── eval.py               # Model evaluation
├── infra/
│   ├── ec2-user-data.sh      # EC2 bootstrap script
│   └── streamlit.service      # systemd unit file
├── deploy/
│   └── deploy.sh              # Deployment script
├── scripts/
│   └── daily_retrain.sh       # Daily retraining cron job
├── tests/                     # Test files
└── sample_data/               # Sample CSV for offline testing
```

## Local Development

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Fetching

```bash
# Download data for a single ticker
python src/data_fetch.py --ticker AAPL --start 2018-01-01 --end 2024-12-31 --out data/

# Download multiple tickers from file
python src/data_fetch.py --tickers-file tickers.txt --start 2018-01-01 --end 2024-12-31 --out data/
```

### Training

```bash
# Train model on processed data
python src/train.py --data data/AAPL-2018-01-01-2024-12-31.csv --out models/
```

### Running Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app will be available at `http://localhost:8501`

## AWS Deployment

### Prerequisites

- AWS S3 bucket created
- EC2 instance with IAM role having S3 read/write permissions
- SSH access to EC2 instance

### EC2 Bootstrap

The `infra/ec2-user-data.sh` script will:
- Install Python 3, pip, nginx, git
- Clone repository
- Create virtual environment
- Install dependencies
- Configure systemd service for Streamlit

### Deploy Updates

```bash
ssh ubuntu@EC2_IP 'cd ~/stock-trend-mvp && git pull && source venv/bin/activate && pip install -r requirements.txt && sudo systemctl restart streamlit-app'
```

Or use the deploy script:

```bash
./deploy/deploy.sh EC2_IP
```

### Daily Retraining

The `scripts/daily_retrain.sh` script runs as a cron job:
- Downloads latest data
- Retrains model
- Uploads updated model to S3

Add to crontab:
```bash
0 2 * * * /home/ubuntu/stock-trend-mvp/scripts/daily_retrain.sh >> /var/log/stock-trend/retrain.log 2>&1
```

## Environment Variables

- `AWS_DEFAULT_REGION`: AWS region (default: us-east-1)
- `S3_BUCKET`: S3 bucket name (required for S3 operations)
- `STREAMLIT_PORT`: Streamlit port (default: 8501)

## Model Artifact Structure

Saved model artifacts (joblib format) contain:
- `model`: Trained RandomForestClassifier
- `scaler`: StandardScaler for feature normalization
- `feature_cols`: List of feature column names
- `meta`: Metadata (ticker, training date, accuracy, etc.)

## Testing

```bash
# Run tests
pytest tests/
```

## Notes

- All timestamps stored in UTC
- Market hours mapped to US/Eastern
- No AWS credentials in code; uses instance role or environment variables
- Sample data included in `sample_data/` for offline testing

