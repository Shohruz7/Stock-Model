"""
Data fetching module for downloading stock data using yfinance and uploading to S3.

This module provides functions to:
- Download daily OHLCV data for tickers
- Save data to local CSV files
- Upload data to S3 buckets
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import boto3
import pandas as pd
import yfinance as yf


def download_daily(
    ticker: str,
    start: str,
    end: str,
    out_path: str,
    interval: str = "1d",
) -> str:
    """
    Download daily OHLCV data for a ticker and save to CSV.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        out_path: Directory path to save CSV file
        interval: Data interval (default: '1d' for daily)

    Returns:
        Path to the saved CSV file

    Raises:
        ValueError: If ticker data cannot be downloaded
    """
    # Create output directory if it doesn't exist
    Path(out_path).mkdir(parents=True, exist_ok=True)

    # Download data using yfinance
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, interval=interval)

        if df.empty:
            raise ValueError(f"No data available for ticker {ticker}")

        # Reset index to make Date a column
        df.reset_index(inplace=True)
        df.rename(columns={"Date": "date"}, inplace=True)

        # Ensure date column is datetime
        df["date"] = pd.to_datetime(df["date"])

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        # Generate filename: {ticker}-{start_date}-{end_date}.csv
        filename = f"{ticker}-{start}-{end}.csv"
        filepath = os.path.join(out_path, filename)

        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Downloaded {len(df)} rows for {ticker} -> {filepath}")

        return filepath

    except Exception as e:
        error_msg = str(e)
        if "No data found" in error_msg or "symbol may be delisted" in error_msg.lower():
            raise ValueError(
                f"❌ No data available for ticker '{ticker}'. "
                f"The ticker may be invalid, delisted, or the date range may be outside available data. "
                f"Please verify the ticker symbol and try a different date range."
            )
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            raise ValueError(
                f"❌ Rate limit exceeded. Too many requests to yfinance. "
                f"Please wait a few minutes and try again, or reduce the number of concurrent requests."
            )
        else:
            raise ValueError(
                f"❌ Error downloading data for {ticker}: {error_msg}. "
                f"Please check your internet connection and try again."
            )


def upload_to_s3(
    local_path: str,
    s3_key: str,
    bucket: str,
    region: Optional[str] = None,
) -> str:
    """
    Upload a file to S3 using boto3 (reads credentials from instance role or env).

    Args:
        local_path: Local file path to upload
        s3_key: S3 key (path) where file will be stored
        bucket: S3 bucket name
        region: AWS region (default: from env or us-east-1)

    Returns:
        S3 URI of uploaded file (s3://bucket/key)

    Raises:
        FileNotFoundError: If local file doesn't exist
        boto3.exceptions.Boto3Error: If S3 upload fails
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"❌ Local file not found: {local_path}\n"
            f"Please ensure the file exists and the path is correct."
        )

    # Initialize S3 client (uses default credentials chain: env vars, IAM role, etc.)
    s3_client = boto3.client("s3", region_name=region)

    try:
        # Upload file
        s3_client.upload_file(local_path, bucket, s3_key)
        s3_uri = f"s3://{bucket}/{s3_key}"
        print(f"Uploaded {local_path} -> {s3_uri}")
        return s3_uri

    except Exception as e:
        error_msg = str(e)
        if "AccessDenied" in error_msg or "403" in error_msg:
            raise Exception(
                f"❌ S3 Access Denied. Check that:\n"
                f"- IAM role has S3 write permissions\n"
                f"- Bucket name is correct: {bucket}\n"
                f"- Bucket policy allows uploads\n"
                f"Original error: {error_msg}"
            )
        elif "NoCredentialsError" in error_msg or "credentials" in error_msg.lower():
            raise Exception(
                f"❌ AWS credentials not found. Ensure:\n"
                f"- EC2 instance has IAM role attached\n"
                f"- Or AWS credentials are configured (aws configure)\n"
                f"Original error: {error_msg}"
            )
        else:
            raise Exception(
                f"❌ Error uploading to S3: {error_msg}\n"
                f"Bucket: {bucket}, Key: {s3_key}"
            )


def download_multiple_tickers(
    tickers: List[str],
    start: str,
    end: str,
    out_path: str,
    upload_to_s3_bucket: Optional[str] = None,
    s3_prefix: str = "raw/",
) -> List[str]:
    """
    Download data for multiple tickers and optionally upload to S3.

    Args:
        tickers: List of ticker symbols
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        out_path: Local directory to save CSV files
        upload_to_s3_bucket: Optional S3 bucket name for upload
        s3_prefix: S3 key prefix (default: 'raw/')

    Returns:
        List of local file paths for downloaded CSVs
    """
    downloaded_files = []

    for ticker in tickers:
        try:
            filepath = download_daily(ticker, start, end, out_path)

            # Optionally upload to S3
            if upload_to_s3_bucket:
                filename = os.path.basename(filepath)
                s3_key = f"{s3_prefix}{filename}"
                upload_to_s3(filepath, s3_key, upload_to_s3_bucket)

            downloaded_files.append(filepath)

        except Exception as e:
            print(f"Warning: Failed to download {ticker}: {e}")
            continue

    return downloaded_files


def main():
    """CLI entry point for data fetching."""
    parser = argparse.ArgumentParser(
        description="Download stock data using yfinance"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Single ticker symbol (e.g., AAPL)",
    )
    parser.add_argument(
        "--tickers-file",
        type=str,
        help="Path to file with one ticker per line",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/",
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        help="Optional S3 bucket name for upload",
    )
    parser.add_argument(
        "--s3-prefix",
        type=str,
        default="raw/",
        help="S3 key prefix (default: raw/)",
    )

    args = parser.parse_args()

    # Determine tickers to download
    tickers = []
    if args.ticker:
        tickers = [args.ticker]
    elif args.tickers_file:
        with open(args.tickers_file, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Must provide either --ticker or --tickers-file")

    # Download data
    download_multiple_tickers(
        tickers=tickers,
        start=args.start,
        end=args.end,
        out_path=args.out,
        upload_to_s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
    )


if __name__ == "__main__":
    main()

