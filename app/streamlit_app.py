"""
Streamlit app for stock trend prediction.

This app provides:
- Ticker input and data loading
- Historical price charts
- Feature visualization
- Next-day prediction with probabilities
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predict import (
    load_model,
    predict_next_day,
    get_model_info,
    validate_data_for_prediction,
)
from data_fetch import download_daily


# Page configuration
st.set_page_config(
    page_title="Stock Trend Predictor",
    page_icon="📈",
    layout="wide",
)

# Title
st.title("📈 Stock Trend Predictor")
st.markdown("Predict next-day stock price direction using machine learning")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Initialize session state for model
if "model_artifact" not in st.session_state:
    st.session_state.model_artifact = None
if "model_info" not in st.session_state:
    st.session_state.model_info = None

# Model loading options
st.sidebar.subheader("Model")
model_source = st.sidebar.radio(
    "Model Source",
    ["Local File", "S3"],
    help="Choose where to load the model from",
)

if model_source == "Local File":
    model_path = st.sidebar.text_input(
        "Model Path",
        value="models/AAPL_model.joblib",
        help="Path to local model artifact file",
    )
    if st.sidebar.button("Load Model"):
        try:
            if os.path.exists(model_path):
                st.session_state.model_artifact = load_model(model_path=model_path)
                st.session_state.model_info = get_model_info(st.session_state.model_artifact)
                st.sidebar.success(f"✅ Model loaded: {st.session_state.model_info['ticker']}")
            else:
                st.sidebar.error(f"❌ Model file not found: {model_path}")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")

else:  # S3
    s3_bucket = st.sidebar.text_input(
        "S3 Bucket",
        value=os.getenv("S3_BUCKET", ""),
        help="S3 bucket name",
    )
    s3_key = st.sidebar.text_input(
        "S3 Key",
        value="models/production.joblib",
        help="S3 key for model artifact",
    )
    if st.sidebar.button("Load Model from S3"):
        try:
            if s3_bucket:
                st.session_state.model_artifact = load_model(s3_bucket=s3_bucket, s3_key=s3_key)
                st.session_state.model_info = get_model_info(st.session_state.model_artifact)
                st.sidebar.success(f"✅ Model loaded: {st.session_state.model_info['ticker']}")
            else:
                st.sidebar.error("❌ Please provide S3 bucket name")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")

# Display model info if loaded
if st.session_state.model_info:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Information")
    st.sidebar.metric("Ticker", st.session_state.model_info["ticker"])
    st.sidebar.metric("Test Accuracy", f"{st.session_state.model_info['accuracy']:.2%}")
    st.sidebar.metric("Train Accuracy", f"{st.session_state.model_info['train_accuracy']:.2%}")
    st.sidebar.caption(f"Trained: {st.session_state.model_info['training_date']}")
    st.sidebar.caption(f"Features: {st.session_state.model_info['feature_count']}")

# Main content area
st.header("Stock Data & Prediction")

# Ticker input
col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input(
        "Stock Ticker",
        value="AAPL",
        placeholder="Enter ticker symbol (e.g., AAPL, MSFT, GOOGL)",
        help="Enter a valid stock ticker symbol",
    ).upper()

with col2:
    use_default_range = st.checkbox("Use default range", value=True)

# Date range selection
if use_default_range:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years default (ensures >26 days)
    col1, col2 = st.columns(2)
    with col1:
        start_date_input = st.date_input("Start Date", value=start_date, disabled=True)
    with col2:
        end_date_input = st.date_input("End Date", value=end_date, disabled=True)
else:
    col1, col2 = st.columns(2)
    with col1:
        start_date_input = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365 * 2),
        )
    with col2:
        end_date_input = st.date_input("End Date", value=datetime.now())

# Load data button
if st.button("📥 Load Data", type="primary"):
    if not ticker:
        st.error("Please enter a ticker symbol")
    else:
        with st.spinner(f"Downloading data for {ticker}..."):
            try:
                # Download data using yfinance
                start_str = start_date_input.strftime("%Y-%m-%d")
                end_str = end_date_input.strftime("%Y-%m-%d")

                stock = yf.Ticker(ticker)
                df = stock.history(start=start_str, end=end_str, interval="1d")

                if df.empty:
                    st.error(f"No data available for ticker {ticker}")
                else:
                    # Reset index and format
                    df.reset_index(inplace=True)
                    df.rename(columns={"Date": "date"}, inplace=True)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date").reset_index(drop=True)

                    # Store in session state
                    st.session_state["data"] = df
                    st.session_state["ticker"] = ticker
                    st.success(f"✅ Loaded {len(df)} days of data for {ticker}")

            except Exception as e:
                st.error(f"Error downloading data: {str(e)}")

# Display data if available
if "data" in st.session_state:
    df = st.session_state["data"]
    ticker = st.session_state.get("ticker", "UNKNOWN")

    st.markdown("---")

    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", len(df))
    with col2:
        latest_close = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2] if len(df) > 1 else latest_close
        change = latest_close - prev_close
        change_pct = (change / prev_close) * 100 if prev_close > 0 else 0
        st.metric("Latest Close", f"${latest_close:.2f}", f"{change_pct:+.2f}%")
    with col3:
        st.metric("Date Range", f"{df['date'].min().date()} to {df['date'].max().date()}")
    with col4:
        st.metric("Avg Volume", f"{df['Volume'].mean():,.0f}")

    # Price chart
    st.subheader("📊 Historical Price Chart")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["date"], df["Close"], linewidth=2, label="Close Price")
    ax.fill_between(df["date"], df["Low"], df["High"], alpha=0.3, label="Daily Range")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.set_title(f"{ticker} Stock Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Features table (sample)
    st.subheader("📋 Feature Sample (Last 5 Days)")
    try:
        from features import compute_features

        X, y = compute_features(df)
        if len(X) > 0:
            # Show last 5 rows
            feature_sample = X.tail(5).copy()
            feature_sample.index = df["date"].iloc[-len(feature_sample) :].values
            st.dataframe(feature_sample.style.format("{:.4f}"), use_container_width=True)
        else:
            st.warning("Unable to compute features (insufficient data)")
    except Exception as e:
        st.warning(f"Unable to compute features: {str(e)}")

    # Prediction section
    st.markdown("---")
    st.subheader("🔮 Next-Day Prediction")

    if st.session_state.model_artifact is None:
        st.warning("⚠️ Please load a model from the sidebar to make predictions")
    else:
        # Validate data
        is_valid, error_msg = validate_data_for_prediction(df)
        if not is_valid:
            st.error(f"❌ {error_msg}")
        else:
            if st.button("🔮 Predict Next Day", type="primary"):
                with st.spinner("Computing prediction..."):
                    try:
                        prediction, probability, probabilities = predict_next_day(
                            df, st.session_state.model_artifact
                        )

                        # Display prediction
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### Prediction Result")
                            if prediction == "Up":
                                st.success(f"📈 **{prediction}** ({probability:.2%} confidence)")
                            else:
                                st.error(f"📉 **{prediction}** ({probability:.2%} confidence)")

                        with col2:
                            st.markdown("### Probabilities")
                            prob_df = pd.DataFrame(
                                {
                                    "Direction": ["Up", "Down"],
                                    "Probability": [
                                        probabilities["Up"],
                                        probabilities["Down"],
                                    ],
                                }
                            )
                            st.bar_chart(prob_df.set_index("Direction"))

                        # Additional info
                        st.info(
                            f"Model trained on {st.session_state.model_info['ticker']} with "
                            f"{st.session_state.model_info['accuracy']:.2%} test accuracy. "
                            f"Based on {len(df)} days of historical data."
                        )

                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.caption(
    "Stock Trend Predictor MVP | Data provided by yfinance | "
    "Predictions are for informational purposes only"
)
