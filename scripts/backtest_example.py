#!/usr/bin/env python3
"""
Example script demonstrating backtesting functionality.

Usage:
    python scripts/backtest_example.py --model models/AAPL_model.joblib --data data/AAPL-2020-01-01-2024-12-31.csv
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import pandas as pd
from eval import backtest_model, print_backtest_summary, plot_backtest_results
from model_utils import load_model_artifact

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest example")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--data", type=str, required=True, help="Path to data CSV")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--plot", action="store_true", help="Generate plots")

    args = parser.parse_args()

    # Load model and data
    print(f"Loading model from {args.model}...")
    artifact = load_model_artifact(args.model)

    print(f"Loading data from {args.data}...")
    data = pd.read_csv(args.data)
    data["date"] = pd.to_datetime(data["date"])

    # Run backtest
    print("\nRunning backtest...")
    results = backtest_model(
        artifact,
        data,
        start_date=args.start,
        end_date=args.end,
    )

    # Print summary
    print_backtest_summary(results)

    # Generate plots if requested
    if args.plot:
        plot_backtest_results(results, show_plot=True)



