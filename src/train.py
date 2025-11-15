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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Try to import XGBoost, but don't fail if it's not available
XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False

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
    n_estimators: int = 200,
    random_state: int = 42,
    tune_hyperparameters: bool = True,
    model_type: str = "random_forest",
) -> dict:
    """
    Train a machine learning model on stock data.

    Args:
        data_path: Path to CSV file with OHLCV data
        output_dir: Directory to save model artifact
        s3_bucket: Optional S3 bucket name for upload
        s3_key: Optional S3 key for model artifact (default: models/production.joblib)
        test_size: Proportion of data to use for testing (default: 0.2)
        n_estimators: Number of trees/estimators (default: 200)
        random_state: Random seed for reproducibility (default: 42)
        tune_hyperparameters: Whether to tune hyperparameters (default: True)
        model_type: Type of model to train - 'random_forest', 'xgboost', or 'gradient_boosting' (default: 'random_forest')

    Returns:
        Dictionary with training results (accuracy, model path, etc.)

    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If data is insufficient or invalid
    """
    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"❌ Data file not found: {data_path}\n"
            f"Please ensure the file exists. You may need to download data first using:\n"
            f"  python src/data_fetch.py --ticker TICKER --start YYYY-MM-DD --end YYYY-MM-DD --out data/"
        )

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

    # Validate data quality
    if len(df) < 50:
        raise ValueError(
            f"❌ Insufficient data: Only {len(df)} rows available. "
            f"Need at least 50 rows for reliable model training. "
            f"Please download more historical data."
        )

    # Compute features
    print("Computing features...")
    try:
        X, y = compute_features(df)
    except Exception as e:
        raise ValueError(
            f"❌ Error computing features: {str(e)}\n"
            f"This may indicate data quality issues. Please check:\n"
            f"- Data has required columns: date, Open, High, Low, Close, Volume\n"
            f"- No excessive missing values\n"
            f"- At least 26 days of data available"
        )

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

    # Normalize model_type
    model_type = model_type.lower().strip()
    if model_type not in ["random_forest", "xgboost", "gradient_boosting"]:
        raise ValueError(
            f"Invalid model_type: {model_type}. Must be one of: 'random_forest', 'xgboost', 'gradient_boosting'"
        )

    # Train model with hyperparameter tuning
    if tune_hyperparameters:
        if model_type == "xgboost" and XGBOOST_AVAILABLE:
            print("Tuning XGBoost hyperparameters with RandomizedSearchCV...")
            param_grid = {
                "n_estimators": [200, 300, 500],
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5],
                "gamma": [0, 0.1, 0.2],
            }
            
            base_model = xgb.XGBClassifier(
                random_state=random_state,
                n_jobs=-1,
                eval_metric="logloss",
                use_label_encoder=False,
            )
            
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=30,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
                random_state=random_state,
                verbose=1,
            )
            
            search.fit(X_train_scaled, y_train)
            model = search.best_estimator_
            
            print(f"\nBest parameters: {search.best_params_}")
            print(f"Best CV score: {search.best_score_:.4f}")
        elif model_type == "gradient_boosting":
            print("Tuning GradientBoostingClassifier hyperparameters with RandomizedSearchCV...")
            param_grid = {
                "n_estimators": [200, 300, 500],
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 0.9, 1.0],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            }
            
            base_model = GradientBoostingClassifier(
                random_state=random_state,
                verbose=0,
            )
            
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=30,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
                random_state=random_state,
                verbose=1,
            )
            
            search.fit(X_train_scaled, y_train)
            model = search.best_estimator_
            
            print(f"\nBest parameters: {search.best_params_}")
            print(f"Best CV score: {search.best_score_:.4f}")
        else:  # random_forest
            print("Tuning RandomForest hyperparameters with RandomizedSearchCV...")
            param_grid = {
                "n_estimators": [300, 500, 700],
                "max_depth": [8, 10, 12, 15],
                "min_samples_split": [5, 10, 15],
                "min_samples_leaf": [2, 4, 6],
                "max_features": ["sqrt", "log2"],
                "class_weight": ["balanced"],
                "max_samples": [0.7, 0.8, 0.9],
            }
            
            base_model = RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,
                verbose=0,
            )
            
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=30,
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
                random_state=random_state,
                verbose=1,
            )
            
            search.fit(X_train_scaled, y_train)
            model = search.best_estimator_
            
            print(f"\nBest parameters: {search.best_params_}")
            print(f"Best CV score: {search.best_score_:.4f}")
    else:
        # No hyperparameter tuning - use default parameters
        if model_type == "xgboost" and XGBOOST_AVAILABLE:
            print(f"Training XGBoostClassifier (n_estimators={n_estimators})...")
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=3,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="logloss",
                use_label_encoder=False,
            )
        elif model_type == "gradient_boosting":
            print(f"Training GradientBoostingClassifier (n_estimators={n_estimators})...")
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=random_state,
                verbose=0,
            )
        else:  # random_forest
            print(f"Training RandomForestClassifier (n_estimators={n_estimators})...")
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
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

    # Optional: Run backtest on test set for additional metrics
    backtest_metrics = None
    try:
        from eval import backtest_model, calculate_returns, calculate_metrics
        
        # Create test data DataFrame for backtesting
        test_data = df.iloc[split_idx:].copy().reset_index(drop=True)
        if len(test_data) > 0:
            # Recompute features for test data (needed for backtesting)
            X_test_features, _ = compute_features(test_data)
            if len(X_test_features) > 0:
                # Get predictions for backtesting
                test_predictions = pd.Series(y_test_pred, index=X_test_features.index)
                test_prices = test_data["Close"].iloc[-len(test_predictions):].values
                test_prices_series = pd.Series(test_prices, index=X_test_features.index)
                
                # Calculate returns
                returns_df = calculate_returns(
                    test_predictions, test_prices_series, initial_capital=10000.0
                )
                backtest_metrics = calculate_metrics(returns_df)
    except Exception as e:
        print(f"Note: Could not compute backtest metrics: {str(e)}")

    # Create metadata
    additional_info = {
        "train_accuracy": float(train_accuracy),
        "model_type": model_type,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "top_features": feature_importance.head(5)["feature"].tolist(),
    }
    
    if backtest_metrics:
        additional_info["backtest_metrics"] = backtest_metrics
    
    metadata = create_metadata(
        ticker=ticker,
        accuracy=test_accuracy,
        train_size=len(X_train),
        test_size=len(X_test),
        feature_count=len(X.columns),
        additional_info=additional_info,
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
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Disable hyperparameter tuning (faster but less accurate)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost", "gradient_boosting"],
        help="Type of model to train (default: random_forest)",
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
        tune_hyperparameters=not args.no_tune,
        model_type=args.model_type,
    )

    print(f"\n✅ Training complete!")
    print(f"Model saved to: {results['model_path']}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
