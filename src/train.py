"""
Model training script for stock trend prediction.

This script:
- Loads processed CSV data
- Computes features and target
- Splits data by time (80% train, 20% test)
- Scales features using StandardScaler
- Trains RandomForestClassifier
- Evaluates model accuracy
- Saves model artifact (local + optional S3)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from features import compute_features, get_feature_names
from model_utils import (
    save_model_artifact,
    upload_model_to_s3,
    create_metadata,
)


def train_model(
    data_path: str,
    output_dir: str = "models/",
    s3_bucket: Optional[str] = None,
    s3_key: Optional[str] = None,
    test_size: float = 0.2,
    n_estimators: int = 100,
    random_state: int = 42,
) -> dict:
    """
    Train RandomForestClassifier model on stock data.

    Args:
        data_path: Path to CSV file with OHLCV data
        output_dir: Directory to save model artifact
        s3_bucket: Optional S3 bucket name for upload
        s3_key: Optional S3 key for model artifact (default: models/production.joblib)
        test_size: Proportion of data to use for testing (default: 0.2)
        n_estimators: Number of trees in RandomForest (default: 100)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary with training results (accuracy, model path, etc.)

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data is insufficient or invalid
    """
    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Extract ticker from filename if not in data
    ticker = df.get("ticker", None)
    if ticker is None:
        # Try to extract from filename
        filename = os.path.basename(data_path)
        ticker = filename.split("-")[0] if "-" in filename else "UNKNOWN"

    print(f"Processing data for ticker: {ticker}")
    print(f"Data shape: {df.shape}")

    # Compute features
    print("Computing features...")
    X, y = compute_features(df)

    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # Time-based split (80% train, 20% test)
    split_idx = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print(f"Training RandomForestClassifier (n_estimators={n_estimators})...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    print("Evaluating model...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\nTrain Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Classification report
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["Down", "Up"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Down    Up")
    print(f"Actual Down   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"        Up    {cm[1,0]:4d}  {cm[1,1]:4d}")

    # Feature importance (top 10)
    feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

    # Create metadata
    metadata = create_metadata(
        ticker=ticker,
        accuracy=test_accuracy,
        train_size=len(X_train),
        test_size=len(X_test),
        feature_count=len(X.columns),
        additional_info={
            "train_accuracy": float(train_accuracy),
            "n_estimators": n_estimators,
            "random_state": random_state,
            "top_features": feature_importance.head(5)["feature"].tolist(),
        },
    )

    # Save model artifact
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model_filename = f"{ticker}_model.joblib"
    model_path = os.path.join(output_dir, model_filename)

    print(f"\nSaving model artifact to {model_path}...")
    save_model_artifact(
        model=model,
        scaler=scaler,
        feature_cols=X.columns.tolist(),
        metadata=metadata,
        filepath=model_path,
    )

    # Optionally upload to S3
    if s3_bucket:
        if s3_key is None:
            s3_key = "models/production.joblib"
        print(f"Uploading model to S3: s3://{s3_bucket}/{s3_key}...")
        upload_model_to_s3(model_path, s3_key, s3_bucket)

    results = {
        "model_path": model_path,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "ticker": ticker,
        "metadata": metadata,
    }

    return results


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train stock trend prediction model")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV file with OHLCV data",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="models/",
        help="Output directory for model artifact (default: models/)",
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        help="Optional S3 bucket name for model upload",
    )
    parser.add_argument(
        "--s3-key",
        type=str,
        default="models/production.joblib",
        help="S3 key for model artifact (default: models/production.joblib)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in RandomForest (default: 100)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Train model
    results = train_model(
        data_path=args.data,
        output_dir=args.out,
        s3_bucket=args.s3_bucket,
        s3_key=args.s3_key if args.s3_bucket else None,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )

    print(f"\n✅ Training complete!")
    print(f"Model saved to: {results['model_path']}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
