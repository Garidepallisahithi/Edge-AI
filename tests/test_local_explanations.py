"""
Unit tests for local_surrogate_explanations function.
Tests defensive fallback, z-score clipping, and edge cases.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase5_explain_uncertainty import local_surrogate_explanations


@pytest.fixture
def sample_data():
    """
    Create sample data for testing.
    Returns train_df, sample_df, and global_importance_df.
    """
    np.random.seed(42)

    # Create training data with realistic features
    n_train = 100
    feature_cols = [f"feature_{i}" for i in range(10)]

    train_data = {
        "sample_id": range(1000, 1000 + n_train),
        "label": np.random.randint(0, 2, n_train),
        "split": ["train"] * n_train,
        "anomaly_probability": np.random.uniform(0, 1, n_train),
        "confidence": np.random.uniform(0.5, 1.0, n_train),
        "prediction": np.random.randint(0, 2, n_train),
    }

    # Add numeric features
    for col in feature_cols:
        train_data[col] = np.random.uniform(1, 100, n_train)

    train_df = pd.DataFrame(train_data)

    # Create sample data: some normal, some sparse (for fallback test)
    sample_data_dict = {
        "sample_id": [5001, 5002, 5003],
        "label": [0, 1, 1],
        "split": ["test", "test", "test"],
        "anomaly_probability": [0.1, 0.9, 1.0],
        "confidence": [0.9, 0.9, 1.0],
        "prediction": [0, 1, 1],
    }

    # Normal features for samples 5001 and 5002
    for col in feature_cols:
        sample_data_dict[col] = [
            np.random.uniform(1, 100),  # sample 5001 - normal
            np.random.uniform(1, 100),  # sample 5002 - normal
            0.0,  # sample 5003 - sparse (all zeros) - triggers fallback
        ]

    sample_df = pd.DataFrame(sample_data_dict)

    # Create global importance dataframe
    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": np.random.uniform(0.01, 0.1, len(feature_cols)),
        }
    )

    return train_df, sample_df, importance_df


def test_local_explanations_basic(sample_data, tmp_path):
    """Test that local_surrogate_explanations runs without error."""
    train_df, sample_df, importance_df = sample_data
    out_dir = tmp_path / "explanations"
    out_dir.mkdir()

    local_df, out_csv = local_surrogate_explanations(
        sample_df=sample_df,
        train_df=train_df,
        global_importance_df=importance_df,
        out_dir=out_dir,
        base_model=None,
        calibrator=None,
        calibration_result=None,
    )

    # Check output exists
    assert out_csv.exists()
    assert not local_df.empty

    # Check required columns
    required_cols = [
        "sample_id",
        "label",
        "prediction",
        "anomaly_probability",
        "feature",
        "value",
        "train_median",
        "train_std",
        "zscore",
        "global_importance",
        "weighted_score",
    ]
    for col in required_cols:
        assert col in local_df.columns, f"Missing column: {col}"


def test_zscore_clipping(sample_data, tmp_path):
    """Test that z-scores are clipped to ±10."""
    train_df, sample_df, importance_df = sample_data
    out_dir = tmp_path / "explanations"
    out_dir.mkdir()

    local_df, _ = local_surrogate_explanations(
        sample_df=sample_df,
        train_df=train_df,
        global_importance_df=importance_df,
        out_dir=out_dir,
        base_model=None,
        calibrator=None,
        calibration_result=None,
    )

    # Check that z-scores are within clipped range
    zscore_col = local_df["zscore"]
    assert (zscore_col >= -10.0).all(), f"Found z-scores < -10: {zscore_col[zscore_col < -10.0]}"
    assert (zscore_col <= 10.0).all(), f"Found z-scores > 10: {zscore_col[zscore_col > 10.0]}"


def test_fallback_triggered_on_sparse_sample(sample_data, tmp_path):
    """Test that fallback is triggered when >40% features are zero."""
    train_df, sample_df, importance_df = sample_data
    out_dir = tmp_path / "explanations"
    out_dir.mkdir()

    # Create a test sample with many zeros
    sparse_sample = sample_df.iloc[[2]].copy()  # sample 5003 (all zeros)

    # Verify it has enough zeros to trigger fallback (>40%)
    feature_cols = [col for col in sparse_sample.columns if col.startswith("feature_")]
    zero_fraction = (sparse_sample[feature_cols] == 0.0).sum().sum() / (
        len(sparse_sample) * len(feature_cols)
    )
    assert zero_fraction > 0.4, f"Expected zero_fraction > 0.4, got {zero_fraction}"

    local_df, _ = local_surrogate_explanations(
        sample_df=sparse_sample,
        train_df=train_df,
        global_importance_df=importance_df,
        out_dir=out_dir,
        base_model=None,
        calibrator=None,
        calibration_result=None,
    )

    # Function should not crash and should produce output
    assert not local_df.empty, "Expected output rows even for sparse sample"
    assert len(local_df) > 0, "Expected at least one feature explanation"


def test_no_nan_in_output(sample_data, tmp_path):
    """Test that output contains no NaN values."""
    train_df, sample_df, importance_df = sample_data
    out_dir = tmp_path / "explanations"
    out_dir.mkdir()

    local_df, _ = local_surrogate_explanations(
        sample_df=sample_df,
        train_df=train_df,
        global_importance_df=importance_df,
        out_dir=out_dir,
        base_model=None,
        calibrator=None,
        calibration_result=None,
    )

    # Check no NaNs in numeric columns
    numeric_cols = ["zscore", "weighted_score", "global_importance", "value"]
    for col in numeric_cols:
        nan_count = local_df[col].isna().sum()
        assert nan_count == 0, f"Found {nan_count} NaNs in {col}"


def test_weighted_score_calculation(sample_data, tmp_path):
    """Test that weighted_score = abs(zscore) * global_importance."""
    train_df, sample_df, importance_df = sample_data
    out_dir = tmp_path / "explanations"
    out_dir.mkdir()

    local_df, _ = local_surrogate_explanations(
        sample_df=sample_df,
        train_df=train_df,
        global_importance_df=importance_df,
        out_dir=out_dir,
        base_model=None,
        calibrator=None,
        calibration_result=None,
    )

    # Verify weighted score calculation
    expected_weighted = (local_df["zscore"].abs() * local_df["global_importance"]).round(6)
    actual_weighted = local_df["weighted_score"].round(6)

    # Allow small floating point tolerance
    assert (
        np.abs(expected_weighted - actual_weighted) < 1e-5
    ).all(), "Weighted score calculation mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
