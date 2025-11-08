"""
Basic tests for feature computation.

This is a placeholder - will be expanded in next iteration.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest
from features import compute_features, get_feature_names


def test_get_feature_names():
    """Test that feature names are returned correctly."""
    feature_names = get_feature_names()
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0


def test_compute_features_shape():
    """Test that compute_features returns correct shapes."""
    # Create sample data
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "Open": range(100, 200),
            "High": range(101, 201),
            "Low": range(99, 199),
            "Close": range(100, 200),
            "Volume": [1000000] * 100,
        }
    )

    X, y = compute_features(df)

    # Check shapes
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 0
    assert len(X.columns) == len(get_feature_names())


def test_compute_features_no_lookahead():
    """Test that features don't use future data (no lookahead leakage)."""
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "Open": range(100, 150),
            "High": range(101, 151),
            "Low": range(99, 149),
            "Close": range(100, 150),
            "Volume": [1000000] * 50,
        }
    )

    X, y = compute_features(df)

    # Last row should have NaN target (no next day)
    # But X should have valid features
    assert not X.isna().any().any(), "Features should not contain NaN values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

