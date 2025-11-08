"""
Feature engineering module for stock price prediction.

This module computes technical indicators and features from OHLCV data
for use in machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def compute_features(df: pd.DataFrame, rsi_period: int = 14) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute features and target variable from OHLCV DataFrame.

    Features computed:
    - Price percentage changes (1, 2, 3 days)
    - Simple Moving Averages (5, 10, 20 days)
    - RSI (14-day default)
    - Rolling standard deviation (10-day)
    - Volume z-score (10-day window)

    Target variable:
    - Next day direction: 1 if close_t+1 > close_t, else 0

    Args:
        df: DataFrame with columns: date, Open, High, Low, Close, Volume
        rsi_period: Period for RSI calculation (default: 14)

    Returns:
        Tuple of (X: DataFrame with features, y: Series with target labels)

    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ["date", "Open", "High", "Low", "Close", "Volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Make a copy to avoid modifying original
    df = df.copy()

    # Ensure date is datetime and sort
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure numeric columns are numeric
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with NaN in critical columns
    df = df.dropna(subset=["Close", "Volume"]).reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(f"Insufficient data: need at least 20 rows, got {len(df)}")

    # ===== FEATURES =====

    # 1. Price percentage changes
    df["pct_change_1"] = df["Close"].pct_change(1)
    df["pct_change_2"] = df["Close"].pct_change(2)
    df["pct_change_3"] = df["Close"].pct_change(3)

    # 2. Simple Moving Averages
    df["sma_5"] = df["Close"].rolling(window=5, min_periods=1).mean()
    df["sma_10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["sma_20"] = df["Close"].rolling(window=20, min_periods=1).mean()

    # Price relative to SMAs
    df["close_to_sma5"] = (df["Close"] - df["sma_5"]) / df["sma_5"]
    df["close_to_sma10"] = (df["Close"] - df["sma_10"]) / df["sma_10"]
    df["close_to_sma20"] = (df["Close"] - df["sma_20"]) / df["sma_20"]

    # 3. RSI (Relative Strength Index)
    df["rsi"] = compute_rsi(df["Close"], period=rsi_period)

    # 4. Rolling standard deviation (volatility)
    df["rolling_std_10"] = df["Close"].rolling(window=10, min_periods=1).std()

    # 5. Volume features
    df["volume_sma_10"] = df["Volume"].rolling(window=10, min_periods=1).mean()
    df["volume_zscore"] = (
        (df["Volume"] - df["volume_sma_10"]) / df["volume_sma_10"].replace(0, np.nan)
    ).fillna(0)

    # 6. High-Low range
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["hl_range_5"] = df["hl_range"].rolling(window=5, min_periods=1).mean()

    # 7. Price position in daily range
    df["price_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-10)

    # ===== TARGET VARIABLE =====
    # Next day direction: 1 if close_t+1 > close_t, else 0
    df["next_close"] = df["Close"].shift(-1)
    df["target"] = (df["next_close"] > df["Close"]).astype(int)

    # ===== SELECT FEATURE COLUMNS =====
    feature_cols = [
        "pct_change_1",
        "pct_change_2",
        "pct_change_3",
        "sma_5",
        "sma_10",
        "sma_20",
        "close_to_sma5",
        "close_to_sma10",
        "close_to_sma20",
        "rsi",
        "rolling_std_10",
        "volume_zscore",
        "hl_range",
        "hl_range_5",
        "price_position",
    ]

    # Extract features and target
    X = df[feature_cols].copy()
    y = df["target"].copy()

    # Remove rows where target is NaN (last row has no next day)
    valid_mask = ~y.isna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    # Fill any remaining NaN values with 0 (shouldn't happen after valid_mask)
    X = X.fillna(0)

    return X, y


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    Args:
        prices: Series of closing prices
        period: RSI period (default: 14)

    Returns:
        Series of RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()

    rs = gain / (loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi


def get_feature_names() -> list:
    """
    Get list of feature column names.

    Returns:
        List of feature names
    """
    return [
        "pct_change_1",
        "pct_change_2",
        "pct_change_3",
        "sma_5",
        "sma_10",
        "sma_20",
        "close_to_sma5",
        "close_to_sma10",
        "close_to_sma20",
        "rsi",
        "rolling_std_10",
        "volume_zscore",
        "hl_range",
        "hl_range_5",
        "price_position",
    ]

