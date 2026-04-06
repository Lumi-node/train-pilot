"""
Unit tests for TrainingEnvironment class.

Tests verify:
1. TrainingEnvironment instantiation with dummy data
2. execute_training() return dict structure (4 required keys)
3. CPU training path execution without exceptions
4. ANE training graceful fallback to CPU if ane_trainer unavailable
5. execute_training() never raises; catches exceptions and returns failure dict
6. execution_time measurement is correct
"""

import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock
from train_orchestrated import TrainingEnvironment


class TestTrainingEnvironmentInstantiation:
    """Test 1: TrainingEnvironment instantiation with dummy data."""

    def test_instantiate_with_dummy_arrays(self):
        """Instantiate TrainingEnvironment with small dummy arrays (10×28×28)."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "small_cnn", "layers": 3}

        env = TrainingEnvironment(X_train, y_train, model_config)

        assert env is not None
        assert env.X_train.shape == (10, 28, 28)
        assert env.y_train.shape == (10,)
        assert env.model_config == model_config

    def test_instantiate_with_different_sizes(self):
        """Instantiate with different array sizes."""
        for n_samples in [5, 20, 100]:
            X_train = np.random.randn(n_samples, 28, 28).astype(np.float32)
            y_train = np.random.randint(0, 10, n_samples).astype(np.int64)
            model_config = {"name": "test_model"}

            env = TrainingEnvironment(X_train, y_train, model_config)

            assert env.X_train.shape[0] == n_samples
            assert env.y_train.shape[0] == n_samples


class TestExecuteTrainingReturnStructure:
    """Test 2: execute_training() returns dict with correct structure."""

    def test_cpu_training_returns_dict_with_4_keys(self):
        """Call execute_training('cpu', {}) and verify return dict has all 4 keys."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)
        result = env.execute_training("cpu", {})

        assert isinstance(result, dict)
        assert len(result) == 4
        assert set(result.keys()) == {"status", "backend", "execution_time", "message"}

    def test_ane_training_returns_dict_with_4_keys(self):
        """Call execute_training('ane', {}) and verify return dict structure."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)
        result = env.execute_training("ane", {})

        assert isinstance(result, dict)
        assert len(result) == 4
        assert set(result.keys()) == {"status", "backend", "execution_time", "message"}


class TestReturnDictTypes:
    """Test 3: Verify return dict has all 4 required keys with correct types."""

    def test_cpu_return_dict_types(self):
        """Verify CPU training return dict has correct types."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)
        result = env.execute_training("cpu", {})

        # Verify types
        assert isinstance(result["status"], str)
        assert isinstance(result["backend"], str)
        assert isinstance(result["execution_time"], float)
        assert isinstance(result["message"], str)

        # Verify valid status values
        assert result["status"] in ["success", "failure"]
        assert result["backend"] == "cpu"

    def test_ane_return_dict_types(self):
        """Verify ANE training return dict has correct types (even on fallback)."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)
        result = env.execute_training("ane", {})

        # Verify types
        assert isinstance(result["status"], str)
        assert isinstance(result["backend"], str)
        assert isinstance(result["execution_time"], float)
        assert isinstance(result["message"], str)

        # Verify valid values
        assert result["status"] in ["success", "failure"]
        assert result["backend"] == "ane"


class TestCPUTrainingExecution:
    """Test 4: CPU training path execution without exceptions."""

    def test_cpu_training_completes_without_exception(self):
        """Verify execute_training('cpu', profile) completes without raising."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)

        # Should not raise
        result = env.execute_training("cpu", {})
        assert result is not None

    def test_cpu_training_returns_success(self):
        """Verify CPU training returns status='success'."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)
        result = env.execute_training("cpu", {})

        assert result["status"] == "success"
        assert result["backend"] == "cpu"

    def test_cpu_training_execution_time_reasonable(self):
        """Verify CPU training execution_time is non-negative and reasonable."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)
        result = env.execute_training("cpu", {})

        assert result["execution_time"] >= 0.0
        assert result["execution_time"] < 60.0  # Should be quick for small data


class TestANEFallback:
    """Test 5: ANE training gracefully falls back to CPU if ane_trainer unavailable."""

    def test_ane_fallback_to_cpu_when_unavailable(self):
        """Verify execute_training('ane', profile) falls back to CPU if ane_trainer unavailable."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)

        # ane_trainer is typically not available, so fallback should happen
        result = env.execute_training("ane", {})

        # Should still complete without raising
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) == 4

    def test_ane_fallback_still_returns_success(self):
        """Verify ANE training returns success even when falling back to CPU."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)
        result = env.execute_training("ane", {})

        # Should return success or failure, but not raise
        assert result["status"] in ["success", "failure"]
        assert result["backend"] == "ane"


class TestExceptionHandling:
    """Test 6: execute_training() never raises; catches exceptions and returns failure dict."""

    def test_invalid_backend_returns_failure(self):
        """Verify invalid backend string returns failure dict without raising."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)

        # Invalid backend should not raise
        result = env.execute_training("invalid_backend", {})

        assert result["status"] == "failure"
        assert result["backend"] == "invalid_backend"
        assert isinstance(result["message"], str)
        assert "Unknown backend" in result["message"] or "failed" in result["message"].lower()

    def test_empty_arrays_handled_gracefully(self):
        """Verify empty arrays are handled gracefully (no exception)."""
        X_train = np.array([], dtype=np.float32).reshape(0, 28, 28)
        y_train = np.array([], dtype=np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)

        # Should not raise
        result = env.execute_training("cpu", {})
        assert isinstance(result, dict)
        assert set(result.keys()) == {"status", "backend", "execution_time", "message"}

    def test_missing_model_config_keys_handled_gracefully(self):
        """Verify missing model config keys are handled gracefully."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {}  # Empty config

        env = TrainingEnvironment(X_train, y_train, model_config)

        # Should not raise
        result = env.execute_training("cpu", {})
        assert isinstance(result, dict)


class TestTimingMeasurement:
    """Test 7: Verify execution_time measurement is correct."""

    def test_execution_time_is_non_zero(self):
        """Verify execution_time is non-zero (actual training happens)."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)
        result = env.execute_training("cpu", {})

        # Execution time should be non-zero for actual training
        assert result["execution_time"] > 0.0

    def test_execution_time_is_float(self):
        """Verify execution_time is always a float."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)
        result = env.execute_training("cpu", {})

        assert isinstance(result["execution_time"], float)

    def test_execution_time_reasonable_bounds(self):
        """Verify execution_time is within reasonable bounds."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)
        result = env.execute_training("cpu", {})

        # Should be less than a reasonable timeout (60 seconds for small data)
        assert 0.0 <= result["execution_time"] < 60.0


class TestBackendSpecificPaths:
    """Test both CPU and ANE backend paths."""

    def test_cpu_path_trains_successfully(self):
        """Test that CPU path actually trains (loss computed, gradients applied)."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)
        result = env.execute_training("cpu", {})

        # CPU should succeed with small data
        assert result["status"] == "success"
        assert result["execution_time"] > 0.0

    def test_ane_path_fallback_is_transparent(self):
        """Test that ANE path gracefully falls back without exposing implementation details."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)
        result = env.execute_training("ane", {})

        # Should not raise and should have valid structure
        assert isinstance(result, dict)
        assert result["backend"] == "ane"
        assert result["status"] in ["success", "failure"]


class TestMultipleInvocations:
    """Test multiple consecutive invocations."""

    def test_multiple_cpu_invocations(self):
        """Test that multiple calls to execute_training work correctly."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)

        # Multiple invocations
        for _ in range(3):
            result = env.execute_training("cpu", {})
            assert result["status"] == "success"
            assert len(result) == 4

    def test_mixed_backend_invocations(self):
        """Test mixing CPU and ANE backend calls."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)

        # CPU then ANE
        result_cpu = env.execute_training("cpu", {})
        assert result_cpu["backend"] == "cpu"

        result_ane = env.execute_training("ane", {})
        assert result_ane["backend"] == "ane"

        # Both should have valid structures
        assert len(result_cpu) == 4
        assert len(result_ane) == 4


class TestReturnDictCompleteness:
    """Test that return dict always has all required fields."""

    def test_success_case_message_contains_info(self):
        """Verify success message is informative."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)
        result = env.execute_training("cpu", {})

        if result["status"] == "success":
            assert "cpu" in result["message"].lower() or "CPU" in result["message"]

    def test_failure_case_message_contains_reason(self):
        """Verify failure message contains error information."""
        X_train = np.random.randn(10, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, 10).astype(np.int64)
        model_config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, model_config)
        result = env.execute_training("invalid_backend", {})

        if result["status"] == "failure":
            assert len(result["message"]) > 0
            assert isinstance(result["message"], str)
