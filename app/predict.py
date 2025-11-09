"""
Prediction utilities for loading models and making predictions.

This module provides functions to:
- Load models from local files or S3
- Compute features from latest data
- Make predictions with probabilities
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model_utils import load_model_artifact, load_model_from_s3
from features import compute_features


def load_model(
    model_path: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    s3_key: Optional[str] = None,
) -> Dict:
    """
    Load model artifact from local file or S3.

    Args:
        model_path: Local path to model artifact file
        s3_bucket: Optional S3 bucket name
        s3_key: Optional S3 key for model artifact

    Returns:
        Dictionary with keys: 'model', 'scaler', 'feature_cols', 'meta'

    Raises:
        ValueError: If neither model_path nor s3_bucket is provided
        FileNotFoundError: If model file doesn't exist
    """
    if model_path:
        artifact = load_model_artifact(model_path)
    elif s3_bucket and s3_key:
        artifact = load_model_from_s3(s3_key, s3_bucket)
    else:
        raise ValueError("Must provide either model_path or (s3_bucket, s3_key)")

    return artifact


def predict_next_day(
    df: pd.DataFrame,
    model_artifact: Dict,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Predict next day direction and probability from historical data.

    Args:
        df: DataFrame with OHLCV data (must have at least 20 rows for features)
        model_artifact: Model artifact dictionary from load_model()

    Returns:
        Tuple of:
        - prediction: 'Up' or 'Down'
        - probability: Probability of the predicted class
        - probabilities: Dictionary with {'Up': prob, 'Down': prob}

    Raises:
        ValueError: If data is insufficient or features don't match
    """
    # Compute features (this will use the last row for prediction)
    try:
        X, _ = compute_features(df)
    except Exception as e:
        raise ValueError(f"Error computing features: {str(e)}")

    if len(X) == 0:
        raise ValueError("Insufficient data to compute features")

    # Get the last row (most recent features)
    latest_features = X.iloc[[-1]].copy()

    # Verify feature columns match
    model_feature_cols = model_artifact["feature_cols"]
    missing_cols = set(model_feature_cols) - set(latest_features.columns)
    if missing_cols:
        # This shouldn't happen if compute_features is working correctly
        # But if it does, provide helpful error message
        raise ValueError(
            f"Missing feature columns: {missing_cols}. "
            f"Computed features: {list(latest_features.columns)}. "
            f"Expected {len(model_feature_cols)} features, got {len(latest_features.columns)}. "
            f"Please ensure you have at least 26 days of data."
        )

    # Select and order features to match model
    latest_features = latest_features[model_feature_cols]

    # Scale features
    scaler = model_artifact["scaler"]
    scaled_features = scaler.transform(latest_features)

    # Make prediction
    model = model_artifact["model"]
    prediction_proba = model.predict_proba(scaled_features)[0]
    prediction_class = model.predict(scaled_features)[0]

    # Map class to label (0 = Down, 1 = Up)
    probabilities = {
        "Down": float(prediction_proba[0]),
        "Up": float(prediction_proba[1]),
    }

    prediction = "Up" if prediction_class == 1 else "Down"
    probability = probabilities[prediction]

    return prediction, probability, probabilities


def get_model_info(model_artifact: Dict) -> Dict:
    """
    Extract model information from artifact.

    Args:
        model_artifact: Model artifact dictionary

    Returns:
        Dictionary with model metadata
    """
    meta = model_artifact.get("meta", {})
    return {
        "ticker": meta.get("ticker", "Unknown"),
        "accuracy": meta.get("accuracy", 0.0),
        "train_accuracy": meta.get("train_accuracy", 0.0),
        "training_date": meta.get("training_date", "Unknown"),
        "train_size": meta.get("train_size", 0),
        "test_size": meta.get("test_size", 0),
        "feature_count": meta.get("feature_count", 0),
        "model_type": meta.get("model_type", "Unknown"),
    }


def validate_data_for_prediction(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that DataFrame has sufficient data for prediction.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_cols = ["date", "Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"

    if len(df) < 26:
        return False, f"Insufficient data: need at least 26 rows for all features, got {len(df)}"

    # Check for NaN values in critical columns
    if df[["Close", "Volume"]].isna().any().any():
        return False, "Data contains NaN values in Close or Volume columns"

    return True, ""
