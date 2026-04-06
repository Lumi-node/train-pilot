"""
Tests for test data setup infrastructure.

Verifies that test data files (X_train.npy, y_train.npy) exist at the correct
path, have the correct shapes, and have the correct data types.

This test module can run standalone and does not depend on any other modules.
"""

import os
import pytest
import numpy as np


class TestDataSetup:
    """Tests for test data setup files and integrity."""

    @pytest.fixture(autouse=True)
    def setup_paths(self):
        """Set up paths for test data files."""
        self.base_path = "test_data_setup"
        self.x_train_path = os.path.join(self.base_path, "X_train.npy")
        self.y_train_path = os.path.join(self.base_path, "y_train.npy")

    def test_x_train_file_exists(self):
        """Test 1: Verify X_train.npy file exists."""
        assert os.path.exists(self.x_train_path), (
            f"X_train.npy not found at {self.x_train_path}"
        )
        assert os.path.isfile(self.x_train_path), (
            f"{self.x_train_path} exists but is not a file"
        )

    def test_y_train_file_exists(self):
        """Test 1: Verify y_train.npy file exists."""
        assert os.path.exists(self.y_train_path), (
            f"y_train.npy not found at {self.y_train_path}"
        )
        assert os.path.isfile(self.y_train_path), (
            f"{self.y_train_path} exists but is not a file"
        )

    def test_x_train_shape(self):
        """Test 2: Verify X_train.npy loads with correct shape (100, 28, 28)."""
        X_train = np.load(self.x_train_path)
        assert X_train.shape == (100, 28, 28), (
            f"X_train shape {X_train.shape} does not match expected (100, 28, 28)"
        )

    def test_y_train_shape(self):
        """Test 2: Verify y_train.npy loads with correct shape (100,)."""
        y_train = np.load(self.y_train_path)
        assert y_train.shape == (100,), (
            f"y_train shape {y_train.shape} does not match expected (100,)"
        )

    def test_x_train_dtype(self):
        """Test 3: Verify X_train.npy has dtype float32."""
        X_train = np.load(self.x_train_path)
        assert X_train.dtype == np.float32, (
            f"X_train dtype {X_train.dtype} does not match expected float32"
        )

    def test_y_train_dtype(self):
        """Test 3: Verify y_train.npy has dtype int64."""
        y_train = np.load(self.y_train_path)
        assert y_train.dtype == np.int64, (
            f"y_train dtype {y_train.dtype} does not match expected int64"
        )

    def test_x_train_contains_numeric_data(self):
        """Test: Verify X_train contains valid numeric data (no NaNs or Infs)."""
        X_train = np.load(self.x_train_path)
        assert not np.any(np.isnan(X_train)), (
            "X_train contains NaN values"
        )
        assert not np.any(np.isinf(X_train)), (
            "X_train contains infinite values"
        )

    def test_y_train_valid_labels(self):
        """Test: Verify y_train contains valid label values in range [0, 9]."""
        y_train = np.load(self.y_train_path)
        assert np.all(y_train >= 0) and np.all(y_train <= 9), (
            f"y_train contains values outside [0, 9], min={y_train.min()}, max={y_train.max()}"
        )

    def test_x_train_reasonable_values(self):
        """Test: Verify X_train values are within reasonable range for normalized data."""
        X_train = np.load(self.x_train_path)
        # For standard normal data, ~99.7% of values should be within [-3, 3]
        # We allow a wider range to accommodate random variation
        assert np.all(X_train >= -10) and np.all(X_train <= 10), (
            f"X_train values appear unreasonable, min={X_train.min()}, max={X_train.max()}"
        )

    def test_files_can_be_loaded_multiple_times(self):
        """Test: Verify files can be loaded multiple times without corruption."""
        # Load X_train twice and verify they're identical
        X_train_1 = np.load(self.x_train_path)
        X_train_2 = np.load(self.x_train_path)
        assert np.array_equal(X_train_1, X_train_2), (
            "X_train data differs on second load"
        )

        # Load y_train twice and verify they're identical
        y_train_1 = np.load(self.y_train_path)
        y_train_2 = np.load(self.y_train_path)
        assert np.array_equal(y_train_1, y_train_2), (
            "y_train data differs on second load"
        )

    def test_file_sizes_are_reasonable(self):
        """Test: Verify file sizes are reasonable for the expected data."""
        x_size = os.path.getsize(self.x_train_path)
        y_size = os.path.getsize(self.y_train_path)

        # X_train: 100 * 28 * 28 * 4 bytes (float32) + numpy overhead ≈ 312KB
        # y_train: 100 * 8 bytes (int64) + numpy overhead ≈ 1KB
        assert x_size > 300000 and x_size < 320000, (
            f"X_train file size {x_size} seems unreasonable"
        )
        assert y_size > 500 and y_size < 2000, (
            f"y_train file size {y_size} seems unreasonable"
        )


def test_acceptance_criteria_ac16():
    """Standalone test for AC16: Test data setup creates required files."""
    base_path = "test_data_setup"
    x_path = os.path.join(base_path, "X_train.npy")
    y_path = os.path.join(base_path, "y_train.npy")

    # Verify files exist
    assert os.path.exists(x_path), f"X_train.npy not found at {x_path}"
    assert os.path.exists(y_path), f"y_train.npy not found at {y_path}"

    # Load and verify shapes
    X = np.load(x_path)
    y = np.load(y_path)

    assert X.shape == (100, 28, 28), f"X shape {X.shape} != (100, 28, 28)"
    assert X.dtype == np.float32, f"X dtype {X.dtype} != float32"
    assert y.shape == (100,), f"y shape {y.shape} != (100,)"
    assert y.dtype == np.int64, f"y dtype {y.dtype} != int64"
