"""
Streamlit app for stock trend prediction.

This app provides:
- Ticker input and data loading
- Historical price charts
- Feature visualization
- Next-day prediction with probabilities
- Backtesting with performance metrics
- Model comparison
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict

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

# Initialize session state
if "model_artifact" not in st.session_state:
    st.session_state.model_artifact = None
if "model_info" not in st.session_state:
    st.session_state.model_info = None
if "models" not in st.session_state:
    st.session_state.models = []  # For model comparison
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []  # Store predictions with timestamps

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
    col1, col2 = st.sidebar.columns(2)
    with col1:
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
    with col2:
        if st.sidebar.button("Add to Compare"):
            try:
                if os.path.exists(model_path):
                    artifact = load_model(model_path=model_path)
                    info = get_model_info(artifact)
                    st.session_state.models.append({"artifact": artifact, "info": info, "path": model_path})
                    st.sidebar.success(f"✅ Added {info['ticker']} to comparison")
                else:
                    st.sidebar.error(f"❌ Model file not found: {model_path}")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")

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

# Clear comparison models button
if len(st.session_state.models) > 0:
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Comparison Models"):
        st.session_state.models = []
        st.sidebar.success("✅ Comparison models cleared")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Backtesting", "⚖️ Model Comparison", "📜 Prediction History"])

# ===== TAB 1: PREDICTION =====
with tab1:
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
        start_date = end_date - timedelta(days=365 * 2)  # 2 years default
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
                    start_str = start_date_input.strftime("%Y-%m-%d")
                    end_str = end_date_input.strftime("%Y-%m-%d")

                    stock = yf.Ticker(ticker)
                    df = stock.history(start=start_str, end=end_str, interval="1d")

                    if df.empty:
                        st.error(f"No data available for ticker {ticker}")
                    else:
                        df.reset_index(inplace=True)
                        df.rename(columns={"Date": "date"}, inplace=True)
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.sort_values("date").reset_index(drop=True)

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

                            st.info(
                                f"Model trained on {st.session_state.model_info['ticker']} with "
                                f"{st.session_state.model_info['accuracy']:.2%} test accuracy. "
                                f"Based on {len(df)} days of historical data."
                            )

                            # Save to prediction history
                            latest_price = df["Close"].iloc[-1]
                            prediction_record = {
                                "timestamp": datetime.now(),
                                "ticker": ticker,
                                "date": df["date"].iloc[-1],
                                "current_price": latest_price,
                                "prediction": prediction,
                                "probability": probability,
                                "prob_up": probabilities["Up"],
                                "prob_down": probabilities["Down"],
                                "model_ticker": st.session_state.model_info["ticker"],
                                "model_accuracy": st.session_state.model_info["accuracy"],
                            }
                            st.session_state.prediction_history.append(prediction_record)

                        except Exception as e:
                            error_msg = str(e)
                            if "Missing feature columns" in error_msg:
                                st.error(
                                    f"❌ **Feature Mismatch Error**\n\n"
                                    f"The model expects different features than what was computed. "
                                    f"This usually means:\n"
                                    f"- The model was trained with a different version of the code\n"
                                    f"- You need to retrain the model with the current feature set\n\n"
                                    f"**Technical details:** {error_msg}"
                                )
                            elif "Insufficient data" in error_msg:
                                st.error(
                                    f"❌ **Insufficient Data**\n\n"
                                    f"Need at least 26 days of data to compute all features. "
                                    f"Current data: {len(df)} days.\n\n"
                                    f"**Solution:** Select a longer date range (at least 1 month)."
                                )
                            elif "Error computing features" in error_msg:
                                st.error(
                                    f"❌ **Feature Computation Error**\n\n"
                                    f"Failed to compute features from the data. "
                                    f"This may indicate data quality issues.\n\n"
                                    f"**Technical details:** {error_msg}\n\n"
                                    f"**Solution:** Try downloading fresh data or check for missing values."
                                )
                            else:
                                st.error(
                                    f"❌ **Prediction Error**\n\n"
                                    f"An unexpected error occurred while making the prediction.\n\n"
                                    f"**Error:** {error_msg}\n\n"
                                    f"**Troubleshooting:**\n"
                                    f"- Ensure the model is properly loaded\n"
                                    f"- Verify data has all required columns (date, Open, High, Low, Close, Volume)\n"
                                    f"- Check that data has at least 26 days of history"
                                )

# ===== TAB 2: BACKTESTING =====
with tab2:
    st.header("📊 Model Backtesting")

    if st.session_state.model_artifact is None:
        st.warning("⚠️ Please load a model from the sidebar to run backtesting")
    else:
        # Backtest configuration
        col1, col2 = st.columns(2)
        with col1:
            backtest_start = st.date_input(
                "Backtest Start Date",
                value=datetime.now() - timedelta(days=365),
                help="Start date for backtesting period",
            )
        with col2:
            backtest_end = st.date_input(
                "Backtest End Date",
                value=datetime.now(),
                help="End date for backtesting period",
            )

        # Use loaded data or allow new data
        use_loaded_data = st.checkbox("Use loaded data from Prediction tab", value=True)

        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                try:
                    from eval import backtest_model, print_backtest_summary

                    # Get data
                    if use_loaded_data and "data" in st.session_state:
                        data = st.session_state["data"].copy()
                    else:
                        # Download data for backtesting
                        ticker = st.session_state.get("ticker", "AAPL")
                        start_str = backtest_start.strftime("%Y-%m-%d")
                        end_str = backtest_end.strftime("%Y-%m-%d")
                        stock = yf.Ticker(ticker)
                        df = stock.history(start=start_str, end=end_str, interval="1d")
                        if df.empty:
                            st.error(f"No data available for {ticker}")
                            st.stop()
                        df.reset_index(inplace=True)
                        df.rename(columns={"Date": "date"}, inplace=True)
                        df["date"] = pd.to_datetime(df["date"])
                        data = df.sort_values("date").reset_index(drop=True)

                    # Run backtest
                    results = backtest_model(
                        st.session_state.model_artifact,
                        data,
                        start_date=backtest_start.strftime("%Y-%m-%d"),
                        end_date=backtest_end.strftime("%Y-%m-%d"),
                    )

                    # Store results
                    st.session_state["backtest_results"] = results

                    # Display summary
                    st.success("✅ Backtest completed!")
                    st.markdown("---")

                    # Metrics in columns
                    metrics = results["metrics"]
                    class_metrics = results["classification_metrics"]

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Return", f"{metrics['total_return']:.2%}")
                        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    with col2:
                        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                        st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
                    with col3:
                        st.metric("Accuracy", f"{class_metrics['accuracy']:.2%}")
                        st.metric("Precision", f"{class_metrics['precision']:.2%}")
                    with col4:
                        st.metric("Recall", f"{class_metrics['recall']:.2%}")
                        st.metric("F1 Score", f"{class_metrics['f1_score']:.2f}")

                    # Visualizations
                    st.markdown("---")
                    st.subheader("📈 Equity Curve")

                    returns_df = results["returns_df"]

                    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

                    # Equity curve
                    ax1 = axes[0]
                    ax1.plot(returns_df.index, returns_df["capital"], label="Strategy", linewidth=2)
                    ax1.plot(
                        returns_df.index,
                        returns_df["price"] / returns_df["price"].iloc[0] * 10000,
                        label="Buy & Hold",
                        alpha=0.7,
                        linewidth=1,
                    )
                    ax1.set_title("Equity Curve", fontsize=14, fontweight="bold")
                    ax1.set_ylabel("Capital ($)")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)

                    # Drawdown
                    ax2 = axes[1]
                    cumulative = returns_df["cumulative_returns"]
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max * 100
                    ax2.fill_between(returns_df.index, drawdown, 0, alpha=0.3, color="red")
                    ax2.plot(returns_df.index, drawdown, color="red", linewidth=1)
                    ax2.set_title("Drawdown", fontsize=14, fontweight="bold")
                    ax2.set_ylabel("Drawdown (%)")
                    ax2.set_xlabel("Trading Days")
                    ax2.grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Confusion matrix
                    st.markdown("---")
                    st.subheader("📊 Confusion Matrix")
                    cm = np.array(results["confusion_matrix"])
                    cm_df = pd.DataFrame(
                        cm,
                        index=["Actual Down", "Actual Up"],
                        columns=["Predicted Down", "Predicted Up"],
                    )
                    st.dataframe(cm_df.style.background_gradient(cmap="Blues"), use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Backtest error: {str(e)}")
                    st.exception(e)

# ===== TAB 3: MODEL COMPARISON =====
with tab3:
    st.header("⚖️ Model Comparison")

    if len(st.session_state.models) == 0:
        st.info("💡 Load models and click 'Add to Compare' in the sidebar to compare them")
    else:
        st.success(f"Comparing {len(st.session_state.models)} models")

        # Comparison table
        comparison_data = []
        for i, model_data in enumerate(st.session_state.models):
            info = model_data["info"]
            comparison_data.append(
                {
                    "Model": f"Model {i+1} ({info['ticker']})",
                    "Ticker": info["ticker"],
                    "Accuracy": info["accuracy"],
                    "Train Accuracy": info["train_accuracy"],
                    "Features": info["feature_count"],
                    "Training Date": info["training_date"],
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df.style.format({"Accuracy": "{:.2%}", "Train Accuracy": "{:.2%}"}), use_container_width=True)

        # Run comparison backtest if data is available
        if "data" in st.session_state and st.button("📊 Compare Models on Current Data", type="primary"):
            with st.spinner("Running comparison..."):
                try:
                    from eval import compare_models

                    data = st.session_state["data"].copy()
                    artifacts = [m["artifact"] for m in st.session_state.models]
                    names = [f"{m['info']['ticker']} Model" for m in st.session_state.models]

                    comparison_results = compare_models(artifacts, names, data)

                    st.markdown("---")
                    st.subheader("📊 Comparison Results")

                    # Format the comparison dataframe
                    display_df = comparison_results.copy()
                    display_df["Accuracy"] = display_df["Accuracy"].apply(lambda x: f"{x:.2%}")
                    display_df["Total Return"] = display_df["Total Return"].apply(lambda x: f"{x:.2%}")
                    display_df["Max Drawdown"] = display_df["Max Drawdown"].apply(lambda x: f"{x:.2%}")
                    display_df["Win Rate"] = display_df["Win Rate"].apply(lambda x: f"{x:.2%}")
                    display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].apply(lambda x: f"{x:.2f}")
                    display_df["F1 Score"] = display_df["F1 Score"].apply(lambda x: f"{x:.2f}")

                    st.dataframe(display_df, use_container_width=True)

                    # Visualization
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                    # Accuracy comparison
                    ax1 = axes[0, 0]
                    ax1.bar(comparison_results["Model"], comparison_results["Accuracy"])
                    ax1.set_title("Accuracy Comparison", fontweight="bold")
                    ax1.set_ylabel("Accuracy")
                    ax1.tick_params(axis="x", rotation=45)

                    # Sharpe Ratio
                    ax2 = axes[0, 1]
                    ax2.bar(comparison_results["Model"], comparison_results["Sharpe Ratio"])
                    ax2.set_title("Sharpe Ratio Comparison", fontweight="bold")
                    ax2.set_ylabel("Sharpe Ratio")
                    ax2.tick_params(axis="x", rotation=45)

                    # Total Return
                    ax3 = axes[1, 0]
                    ax3.bar(comparison_results["Model"], comparison_results["Total Return"])
                    ax3.set_title("Total Return Comparison", fontweight="bold")
                    ax3.set_ylabel("Total Return")
                    ax3.tick_params(axis="x", rotation=45)

                    # Win Rate
                    ax4 = axes[1, 1]
                    ax4.bar(comparison_results["Model"], comparison_results["Win Rate"])
                    ax4.set_title("Win Rate Comparison", fontweight="bold")
                    ax4.set_ylabel("Win Rate")
                    ax4.tick_params(axis="x", rotation=45)

                    plt.tight_layout()
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"❌ Comparison error: {str(e)}")
                    st.exception(e)

# ===== TAB 4: PREDICTION HISTORY =====
with tab4:
    st.header("📜 Prediction History")

    if len(st.session_state.prediction_history) == 0:
        st.info("💡 No predictions made yet. Make predictions in the Prediction tab to see them here.")
    else:
        # Convert history to DataFrame
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Display summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            up_predictions = (history_df["prediction"] == "Up").sum()
            st.metric("Up Predictions", up_predictions)
        with col3:
            avg_confidence = history_df["probability"].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")

        st.markdown("---")

        # Display history table
        st.subheader("Prediction Records")
        
        # Format for display
        display_df = history_df.copy()
        display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")
        display_df["current_price"] = display_df["current_price"].apply(lambda x: f"${x:.2f}")
        display_df["probability"] = display_df["probability"].apply(lambda x: f"{x:.2%}")
        display_df["prob_up"] = display_df["prob_up"].apply(lambda x: f"{x:.2%}")
        display_df["prob_down"] = display_df["prob_down"].apply(lambda x: f"{x:.2%}")
        display_df["model_accuracy"] = display_df["model_accuracy"].apply(lambda x: f"{x:.2%}")

        # Rename columns for better display
        display_df = display_df.rename(columns={
            "timestamp": "Time",
            "ticker": "Ticker",
            "date": "Data Date",
            "current_price": "Price",
            "prediction": "Prediction",
            "probability": "Confidence",
            "prob_up": "P(Up)",
            "prob_down": "P(Down)",
            "model_ticker": "Model",
            "model_accuracy": "Model Acc",
        })

        # Select columns to show
        cols_to_show = ["Time", "Ticker", "Data Date", "Price", "Prediction", "Confidence", "P(Up)", "P(Down)", "Model", "Model Acc"]
        st.dataframe(display_df[cols_to_show], use_container_width=True, hide_index=True)

        # Export button
        if st.button("📥 Export to CSV"):
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.prediction_history = []
            st.success("✅ Prediction history cleared!")
            st.rerun()

# Footer
st.markdown("---")
st.caption(
    "Stock Trend Predictor MVP | Data provided by yfinance | "
    "Predictions are for informational purposes only"
)
