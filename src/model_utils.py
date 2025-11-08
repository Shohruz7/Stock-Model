"""
Model persistence utilities for saving/loading model artifacts.

This module handles saving and loading model artifacts including:
- Trained model (RandomForestClassifier)
- Feature scaler (StandardScaler)
- Feature column names
- Metadata (ticker, training date, accuracy, etc.)
"""

import os
import joblib
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path

import boto3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def save_model_artifact(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    feature_cols: list,
    metadata: Dict[str, Any],
    filepath: str,
) -> str:
    """
    Save model artifact to local file using joblib.

    Args:
        model: Trained RandomForestClassifier
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
        metadata: Dictionary with metadata (ticker, accuracy, training_date, etc.)
        filepath: Local file path to save artifact

    Returns:
        Path to saved file

    Raises:
        ValueError: If required components are missing
    """
    # Validate inputs
    if model is None or scaler is None:
        raise ValueError("Model and scaler must be provided")

    if not feature_cols:
        raise ValueError("Feature columns list cannot be empty")

    # Create output directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Prepare artifact dictionary
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "meta": metadata,
    }

    # Save using joblib
    joblib.dump(artifact, filepath)
    print(f"Model artifact saved to {filepath}")

    return filepath


def load_model_artifact(filepath: str) -> Dict[str, Any]:
    """
    Load model artifact from local file.

    Args:
        filepath: Path to model artifact file

    Returns:
        Dictionary with keys: 'model', 'scaler', 'feature_cols', 'meta'

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If artifact is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model artifact not found: {filepath}")

    try:
        artifact = joblib.load(filepath)

        # Validate artifact structure
        required_keys = ["model", "scaler", "feature_cols", "meta"]
        missing_keys = [key for key in required_keys if key not in artifact]
        if missing_keys:
            raise ValueError(f"Invalid artifact: missing keys {missing_keys}")

        return artifact

    except Exception as e:
        raise ValueError(f"Error loading model artifact: {str(e)}")


def upload_model_to_s3(
    local_path: str,
    s3_key: str,
    bucket: str,
    region: Optional[str] = None,
) -> str:
    """
    Upload model artifact to S3.

    Args:
        local_path: Local file path to model artifact
        s3_key: S3 key (path) where artifact will be stored
        bucket: S3 bucket name
        region: AWS region (default: from env or us-east-1)

    Returns:
        S3 URI of uploaded file (s3://bucket/key)
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    # Initialize S3 client (uses default credentials chain)
    s3_client = boto3.client("s3", region_name=region)

    try:
        # Upload file
        s3_client.upload_file(local_path, bucket, s3_key)
        s3_uri = f"s3://{bucket}/{s3_key}"
        print(f"Model artifact uploaded to {s3_uri}")
        return s3_uri

    except Exception as e:
        raise Exception(f"Error uploading model to S3: {str(e)}")


def download_model_from_s3(
    s3_key: str,
    bucket: str,
    local_path: str,
    region: Optional[str] = None,
) -> str:
    """
    Download model artifact from S3 to local file.

    Args:
        s3_key: S3 key (path) of the model artifact
        bucket: S3 bucket name
        local_path: Local file path to save artifact
        region: AWS region (default: from env or us-east-1)

    Returns:
        Path to downloaded file
    """
    # Create output directory if it doesn't exist
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    # Initialize S3 client
    s3_client = boto3.client("s3", region_name=region)

    try:
        # Download file
        s3_client.download_file(bucket, s3_key, local_path)
        print(f"Model artifact downloaded from s3://{bucket}/{s3_key} to {local_path}")
        return local_path

    except Exception as e:
        raise Exception(f"Error downloading model from S3: {str(e)}")


def load_model_from_s3(
    s3_key: str,
    bucket: str,
    local_cache_path: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load model artifact directly from S3 (downloads to cache first).

    Args:
        s3_key: S3 key (path) of the model artifact
        bucket: S3 bucket name
        local_cache_path: Optional local path to cache downloaded file
        region: AWS region (default: from env or us-east-1)

    Returns:
        Dictionary with keys: 'model', 'scaler', 'feature_cols', 'meta'
    """
    # Use cache path or create temporary one
    if local_cache_path is None:
        local_cache_path = f"/tmp/model_{s3_key.replace('/', '_')}.joblib"

    # Download from S3
    download_model_from_s3(s3_key, bucket, local_cache_path, region)

    # Load artifact
    return load_model_artifact(local_cache_path)


def create_metadata(
    ticker: str,
    accuracy: float,
    train_size: int,
    test_size: int,
    feature_count: int,
    training_date: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create metadata dictionary for model artifact.

    Args:
        ticker: Stock ticker symbol
        accuracy: Model accuracy score
        train_size: Number of training samples
        test_size: Number of test samples
        feature_count: Number of features
        training_date: Optional training date (default: current UTC)
        additional_info: Optional additional metadata

    Returns:
        Metadata dictionary
    """
    if training_date is None:
        training_date = datetime.utcnow().isoformat()

    metadata = {
        "ticker": ticker,
        "accuracy": float(accuracy),
        "train_size": int(train_size),
        "test_size": int(test_size),
        "feature_count": int(feature_count),
        "training_date": training_date,
        "model_type": "RandomForestClassifier",
    }

    if additional_info:
        metadata.update(additional_info)

    return metadata
