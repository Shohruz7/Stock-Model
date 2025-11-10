"""
Model evaluation utilities with backtesting framework.

This module provides:
- Backtesting with walk-forward validation
- Performance metrics (Sharpe ratio, max drawdown, win rate)
- Cumulative returns calculation
- Visualization tools
- Model comparison utilities
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import compute_features
from model_utils import load_model_artifact


def calculate_returns(
    predictions: pd.Series,
    actual_prices: pd.Series,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001,  # 0.1% transaction cost
) -> pd.DataFrame:
    """
    Calculate cumulative returns based on predictions.

    Args:
        predictions: Series of predictions (1 = buy, 0 = hold/sell)
        actual_prices: Series of actual closing prices
        initial_capital: Starting capital
        transaction_cost: Transaction cost as fraction (default: 0.1%)

    Returns:
        DataFrame with columns: date, position, returns, cumulative_returns, capital
    """
    if len(predictions) != len(actual_prices):
        raise ValueError("Predictions and prices must have same length")

    # Calculate daily returns
    price_returns = actual_prices.pct_change().fillna(0)

    # Apply predictions: 1 = long position, 0 = no position
    positions = predictions.astype(int)
    strategy_returns = positions.shift(1) * price_returns  # Shift to avoid lookahead

    # Apply transaction costs when position changes
    position_changes = positions.diff().abs()
    strategy_returns = strategy_returns - (position_changes * transaction_cost)

    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    capital = initial_capital * cumulative_returns

    results = pd.DataFrame(
        {
            "position": positions,
            "returns": strategy_returns,
            "cumulative_returns": cumulative_returns,
            "capital": capital,
            "price": actual_prices,
        }
    )

    return results


def calculate_metrics(returns_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate performance metrics from returns DataFrame.

    Args:
        returns_df: DataFrame from calculate_returns()

    Returns:
        Dictionary with performance metrics
    """
    returns = returns_df["returns"].dropna()

    if len(returns) == 0:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    # Total return
    total_return = returns_df["cumulative_returns"].iloc[-1] - 1.0

    # Annualized return (assuming 252 trading days)
    trading_days = len(returns)
    years = trading_days / 252.0
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(252)

    # Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0

    # Maximum drawdown
    cumulative = returns_df["cumulative_returns"]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    winning_trades = (returns > 0).sum()
    total_trades = (returns != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

    # Profit factor (gross profit / gross loss)
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "total_trades": int(total_trades),
        "winning_trades": int(winning_trades),
    }


def backtest_model(
    model_artifact: Dict,
    data: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001,
) -> Dict:
    """
    Backtest a model on historical data.

    Args:
        model_artifact: Model artifact dictionary
        data: DataFrame with OHLCV data
        start_date: Optional start date for backtesting (YYYY-MM-DD)
        end_date: Optional end date for backtesting (YYYY-MM-DD)
        initial_capital: Starting capital
        transaction_cost: Transaction cost as fraction

    Returns:
        Dictionary with backtest results including predictions, returns, and metrics
    """
    # Filter data by date range if provided
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], utc=True)
        if start_date:
            data = data[data["date"] >= pd.to_datetime(start_date, utc=True)]
        if end_date:
            data = data[data["date"] <= pd.to_datetime(end_date, utc=True)]
        data = data.sort_values("date").reset_index(drop=True)

    # Compute features
    X, y_true = compute_features(data)

    if len(X) == 0:
        raise ValueError("No data available for backtesting")

    # Get model and scaler
    model = model_artifact["model"]
    scaler = model_artifact["scaler"]
    feature_cols = model_artifact["feature_cols"]

    # Ensure feature columns match
    missing_cols = set(feature_cols) - set(X.columns)
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    # Select and scale features
    X_scaled = scaler.transform(X[feature_cols])

    # Make predictions
    predictions = model.predict(X_scaled)
    prediction_proba = model.predict_proba(X_scaled)[:, 1]  # Probability of "Up"

    # Get actual prices for returns calculation
    # Align with predictions (predictions are for next day, so shift prices)
    actual_prices = data["Close"].iloc[-len(predictions) :].values

    # Calculate returns
    predictions_series = pd.Series(predictions, index=X.index)
    prices_series = pd.Series(actual_prices, index=X.index)

    returns_df = calculate_returns(
        predictions_series, prices_series, initial_capital, transaction_cost
    )

    # Calculate metrics
    metrics = calculate_metrics(returns_df)

    # Classification metrics
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, prediction_proba)
    except ValueError:
        roc_auc = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, predictions)

    results = {
        "predictions": predictions,
        "prediction_proba": prediction_proba,
        "y_true": y_true.values,
        "returns_df": returns_df,
        "metrics": metrics,
        "classification_metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
        },
        "confusion_matrix": cm.tolist(),
        "dates": data["date"].iloc[-len(predictions) :].values if "date" in data.columns else None,
    }

    return results


def walk_forward_validation(
    model_artifact: Dict,
    data: pd.DataFrame,
    train_window: int = 252,  # 1 year
    test_window: int = 63,  # ~3 months
    step: int = 21,  # ~1 month step
) -> List[Dict]:
    """
    Perform walk-forward validation (rolling window backtesting).

    Args:
        model_artifact: Model artifact dictionary
        data: Full historical DataFrame
        train_window: Size of training window (days)
        test_window: Size of test window (days)
        step: Step size for rolling window (days)

    Returns:
        List of backtest results for each window
    """
    if "date" not in data.columns:
        raise ValueError("Data must have 'date' column for walk-forward validation")

    data = data.sort_values("date").reset_index(drop=True)
    results = []

    total_days = len(data)
    current_start = 0

    while current_start + train_window + test_window <= total_days:
        train_end = current_start + train_window
        test_end = train_end + test_window

        train_data = data.iloc[current_start:train_end].copy()
        test_data = data.iloc[train_end:test_end].copy()

        try:
            # Backtest on test window
            backtest_result = backtest_model(
                model_artifact,
                test_data,
                initial_capital=10000.0,
            )

            # Add window information
            backtest_result["window"] = {
                "train_start": data.iloc[current_start]["date"],
                "train_end": data.iloc[train_end - 1]["date"],
                "test_start": data.iloc[train_end]["date"],
                "test_end": data.iloc[test_end - 1]["date"],
            }

            results.append(backtest_result)

        except Exception as e:
            print(f"Error in window {current_start}-{test_end}: {str(e)}")
            continue

        current_start += step

    return results


def plot_backtest_results(
    backtest_results: Dict,
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Plot backtest results including equity curve and metrics.

    Args:
        backtest_results: Results dictionary from backtest_model()
        save_path: Optional path to save plot
        show_plot: Whether to display plot
    """
    returns_df = backtest_results["returns_df"]
    metrics = backtest_results["metrics"]
    classification_metrics = backtest_results["classification_metrics"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Equity curve
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

    # Plot 2: Drawdown
    ax2 = axes[1]
    cumulative = returns_df["cumulative_returns"]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max * 100
    ax2.fill_between(returns_df.index, drawdown, 0, alpha=0.3, color="red")
    ax2.plot(returns_df.index, drawdown, color="red", linewidth=1)
    ax2.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Returns distribution
    ax3 = axes[2]
    returns = returns_df["returns"].dropna()
    ax3.hist(returns, bins=50, alpha=0.7, edgecolor="black")
    ax3.axvline(0, color="red", linestyle="--", linewidth=1)
    ax3.set_title("Returns Distribution", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Daily Returns")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Add metrics text
    metrics_text = f"""
    Performance Metrics:
    Total Return: {metrics['total_return']:.2%}
    Annualized Return: {metrics['annualized_return']:.2%}
    Volatility: {metrics['volatility']:.2%}
    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
    Max Drawdown: {metrics['max_drawdown']:.2%}
    Win Rate: {metrics['win_rate']:.2%}
    
    Classification Metrics:
    Accuracy: {classification_metrics['accuracy']:.2%}
    Precision: {classification_metrics['precision']:.2%}
    Recall: {classification_metrics['recall']:.2%}
    F1 Score: {classification_metrics['f1_score']:.2f}
    ROC AUC: {classification_metrics['roc_auc']:.2f}
    """
    fig.text(0.02, 0.02, metrics_text, fontsize=9, verticalalignment="bottom")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def compare_models(
    model_artifacts: List[Dict],
    model_names: List[str],
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare multiple models on the same dataset.

    Args:
        model_artifacts: List of model artifact dictionaries
        model_names: List of model names
        data: Test DataFrame

    Returns:
        DataFrame with comparison metrics
    """
    if len(model_artifacts) != len(model_names):
        raise ValueError("Number of models and names must match")

    comparison_results = []

    for artifact, name in zip(model_artifacts, model_names):
        try:
            results = backtest_model(artifact, data)
            metrics = results["metrics"]
            class_metrics = results["classification_metrics"]

            comparison_results.append(
                {
                    "Model": name,
                    "Accuracy": class_metrics["accuracy"],
                    "Sharpe Ratio": metrics["sharpe_ratio"],
                    "Total Return": metrics["total_return"],
                    "Max Drawdown": metrics["max_drawdown"],
                    "Win Rate": metrics["win_rate"],
                    "F1 Score": class_metrics["f1_score"],
                }
            )
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
            continue

    return pd.DataFrame(comparison_results)


def print_backtest_summary(backtest_results: Dict) -> None:
    """
    Print a formatted summary of backtest results.

    Args:
        backtest_results: Results dictionary from backtest_model()
    """
    metrics = backtest_results["metrics"]
    class_metrics = backtest_results["classification_metrics"]
    cm = np.array(backtest_results["confusion_matrix"])

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)

    print("\n📊 Performance Metrics:")
    print(f"  Total Return:        {metrics['total_return']:>10.2%}")
    print(f"  Annualized Return:  {metrics['annualized_return']:>10.2%}")
    print(f"  Volatility:         {metrics['volatility']:>10.2%}")
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>10.2f}")
    print(f"  Max Drawdown:       {metrics['max_drawdown']:>10.2%}")
    print(f"  Win Rate:           {metrics['win_rate']:>10.2%}")
    print(f"  Profit Factor:      {metrics['profit_factor']:>10.2f}")
    print(f"  Total Trades:       {metrics['total_trades']:>10d}")

    print("\n🎯 Classification Metrics:")
    print(f"  Accuracy:           {class_metrics['accuracy']:>10.2%}")
    print(f"  Precision:          {class_metrics['precision']:>10.2%}")
    print(f"  Recall:             {class_metrics['recall']:>10.2%}")
    print(f"  F1 Score:           {class_metrics['f1_score']:>10.2f}")
    print(f"  ROC AUC:            {class_metrics['roc_auc']:>10.2f}")

    print("\n📈 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Down    Up")
    print(f"Actual Down   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"        Up    {cm[1,0]:4d}  {cm[1,1]:4d}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Backtest a stock prediction model")
    parser.add_argument("--model", type=str, required=True, help="Path to model artifact")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--output", type=str, help="Output directory for results")

    args = parser.parse_args()

    # Load model
    artifact = load_model_artifact(args.model)

    # Load data
    data = pd.read_csv(args.data)
    data["date"] = pd.to_datetime(data["date"], utc=True)

    # Run backtest
    results = backtest_model(artifact, data, args.start, args.end)

    # Print summary
    print_backtest_summary(results)

    # Generate plots if requested
    if args.plot:
        output_path = None
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            output_path = os.path.join(args.output, "backtest_results.png")
        plot_backtest_results(results, save_path=output_path, show_plot=True)
