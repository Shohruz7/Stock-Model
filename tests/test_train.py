"""
Tests for model training pipeline.

This is a smoke test to verify training works on sample data.
"""

import os
import tempfile
import shutil
import pandas as pd
import pytest
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train import train_model
from model_utils import load_model_artifact, create_metadata


def create_sample_data(n_days: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    
    # Create realistic price data with some trend
    base_price = 100
    prices = []
    for i in range(n_days):
        # Random walk with slight upward trend
        change = (i * 0.1) + (i % 10 - 5) * 0.5
        price = base_price + change
        prices.append(price)
    
    df = pd.DataFrame({
        "date": dates,
        "Open": [p * 0.99 for p in prices],
        "High": [p * 1.02 for p in prices],
        "Low": [p * 0.98 for p in prices],
        "Close": prices,
        "Volume": [1000000 + i * 1000 for i in range(n_days)],
    })
    
    return df


def test_train_model_smoke():
    """Smoke test: verify training completes without errors."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Create sample data
        df = create_sample_data(n_days=100)
        data_path = os.path.join(temp_dir, "test_data.csv")
        df.to_csv(data_path, index=False)
        
        # Train model
        output_dir = os.path.join(temp_dir, "models")
        results = train_model(
            data_path=data_path,
            output_dir=output_dir,
            test_size=0.2,
            n_estimators=10,  # Small for speed
            random_state=42,
        )
        
        # Verify results
        assert "model_path" in results
        assert "test_accuracy" in results
        assert os.path.exists(results["model_path"])
        assert 0 <= results["test_accuracy"] <= 1
        
        # Verify model can be loaded
        artifact = load_model_artifact(results["model_path"])
        assert "model" in artifact
        assert "scaler" in artifact
        assert "feature_cols" in artifact
        assert "meta" in artifact
        
        print(f"✅ Training smoke test passed!")
        print(f"   Model path: {results['model_path']}")
        print(f"   Test accuracy: {results['test_accuracy']:.4f}")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_create_metadata():
    """Test metadata creation."""
    metadata = create_metadata(
        ticker="AAPL",
        accuracy=0.75,
        train_size=800,
        test_size=200,
        feature_count=15,
    )
    
    assert metadata["ticker"] == "AAPL"
    assert metadata["accuracy"] == 0.75
    assert metadata["train_size"] == 800
    assert metadata["test_size"] == 200
    assert metadata["feature_count"] == 15
    assert "training_date" in metadata


if __name__ == "__main__":
    test_train_model_smoke()
    test_create_metadata()
    print("All tests passed!")



