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

    # Need at least 26 rows for MACD (EMA-26)
    if len(df) < 26:
        raise ValueError(f"Insufficient data: need at least 26 rows for all features, got {len(df)}")

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

    # 8. MACD (Moving Average Convergence Divergence)
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # 9. Bollinger Bands
    bb_period = 20
    bb_std = 2
    df["bb_middle"] = df["Close"].rolling(window=bb_period, min_periods=1).mean()
    bb_std_val = df["Close"].rolling(window=bb_period, min_periods=1).std()
    df["bb_upper"] = df["bb_middle"] + (bb_std_val * bb_std)
    df["bb_lower"] = df["bb_middle"] - (bb_std_val * bb_std)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

    # 10. Momentum indicators
    df["momentum_5"] = df["Close"].pct_change(5)
    df["momentum_10"] = df["Close"].pct_change(10)
    df["roc_5"] = ((df["Close"] - df["Close"].shift(5)) / df["Close"].shift(5)) * 100
    df["roc_10"] = ((df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)) * 100

    # 11. Exponential Moving Averages
    df["ema_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["close_to_ema5"] = (df["Close"] - df["ema_5"]) / df["ema_5"]
    df["close_to_ema10"] = (df["Close"] - df["ema_10"]) / df["ema_10"]
    df["close_to_ema20"] = (df["Close"] - df["ema_20"]) / df["ema_20"]

    # 12. Volume indicators
    df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(window=20, min_periods=1).mean()
    df["volume_price_trend"] = (df["Volume"] * df["Close"].pct_change()).rolling(window=5, min_periods=1).sum()

    # 13. Volatility features
    df["atr_14"] = compute_atr(df, period=14)
    df["volatility_5"] = df["Close"].pct_change().rolling(window=5, min_periods=1).std()
    df["volatility_10"] = df["Close"].pct_change().rolling(window=10, min_periods=1).std()
    df["volatility_20"] = df["Close"].pct_change().rolling(window=20, min_periods=1).std()

    # 14. Price patterns
    df["higher_high"] = (df["High"] > df["High"].shift(1)).astype(int)
    df["lower_low"] = (df["Low"] < df["Low"].shift(1)).astype(int)
    df["gap_up"] = ((df["Open"] > df["Close"].shift(1))).astype(int)
    df["gap_down"] = ((df["Open"] < df["Close"].shift(1))).astype(int)

    # 15. Trend strength
    df["adx"] = compute_adx(df, period=14)
    # Trend strength: 1 if price increased over window, -1 if decreased, 0 if same
    def calc_trend(x):
        if len(x) < 2:
            return 0
        if x.iloc[-1] > x.iloc[0]:
            return 1
        elif x.iloc[-1] < x.iloc[0]:
            return -1
        return 0
    
    df["trend_strength"] = df["Close"].rolling(window=10, min_periods=1).apply(calc_trend, raw=False)

    # ===== TARGET VARIABLE =====
    # Next day direction: 1 if close_t+1 > close_t, else 0
    # Use threshold-based target: predict if price will move by more than 0.5%
    # This creates a more meaningful prediction task than simple up/down
    df["next_close"] = df["Close"].shift(-1)
    df["next_pct_change"] = (df["next_close"] - df["Close"]) / df["Close"]
    
    # Target: 1 if price goes up by more than 0.5%, 0 otherwise
    threshold = 0.005  # 0.5% threshold
    df["target"] = (df["next_pct_change"] > threshold).astype(int)

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
        "macd",
        "macd_signal",
        "macd_histogram",
        "bb_width",
        "bb_position",
        "momentum_5",
        "momentum_10",
        "roc_5",
        "roc_10",
        "close_to_ema5",
        "close_to_ema10",
        "close_to_ema20",
        "volume_ratio",
        "volume_price_trend",
        "atr_14",
        "volatility_5",
        "volatility_10",
        "volatility_20",
        "higher_high",
        "lower_low",
        "gap_up",
        "gap_down",
        "adx",
        "trend_strength",
    ]

    # Extract features and target
    # Ensure all feature columns exist (create missing ones with 0)
    for col in feature_cols:
        if col not in df.columns:
            print(f"Warning: Feature {col} not found, filling with 0")
            df[col] = 0
    
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


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR).

    Args:
        df: DataFrame with High, Low, Close columns
        period: ATR period (default: 14)

    Returns:
        Series of ATR values
    """
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    
    return atr


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average Directional Index (ADX) - simplified version.

    Args:
        df: DataFrame with High, Low, Close columns
        period: ADX period (default: 14)

    Returns:
        Series of ADX values
    """
    # Simplified ADX calculation
    high_diff = df["High"].diff()
    low_diff = -df["Low"].diff()
    
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    atr = compute_atr(df, period)
    
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period, min_periods=1).mean()
    
    return adx.fillna(0)


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
        "macd",
        "macd_signal",
        "macd_histogram",
        "bb_width",
        "bb_position",
        "momentum_5",
        "momentum_10",
        "roc_5",
        "roc_10",
        "close_to_ema5",
        "close_to_ema10",
        "close_to_ema20",
        "volume_ratio",
        "volume_price_trend",
        "atr_14",
        "volatility_5",
        "volatility_10",
        "volatility_20",
        "higher_high",
        "lower_low",
        "gap_up",
        "gap_down",
        "adx",
        "trend_strength",
    ]

