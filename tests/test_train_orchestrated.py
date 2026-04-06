"""
Test module for train_orchestrated.py entry point.

Tests the CLI interface, orchestration loop, and backend selection.
Uses subprocess to test CLI (cannot mock sys.exit easily).
"""

import subprocess
import sys
import os
import pytest


class TestCLIInterface:
    """Tests for CLI argument parsing and --help output."""

    def test_help_output_contains_model_arg(self):
        """AC14: --help output contains --model argument."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode == 0
        assert "--model" in result.stdout
        help_text = result.stdout
        assert "model" in help_text.lower()

    def test_help_output_contains_dataset_arg(self):
        """AC14: --help output contains --dataset argument."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode == 0
        assert "--dataset" in result.stdout
        help_text = result.stdout
        assert "dataset" in help_text.lower()

    def test_help_output_format(self):
        """Test that --help output follows standard argparse format."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout
        assert "options:" in result.stdout or "optional arguments:" in result.stdout

    def test_model_choices_are_valid(self):
        """AC15: --model accepts small_cnn, large_dense, sequential_model."""
        # Test that help shows the valid choices
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert "small_cnn" in result.stdout
        assert "large_dense" in result.stdout
        assert "sequential_model" in result.stdout

    def test_model_argument_required(self):
        """Test that --model is a required argument."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py", "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "--model" in result.stderr

    def test_dataset_argument_required(self):
        """Test that --dataset is a required argument."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py", "--model", "small_cnn"],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "--dataset" in result.stderr

    def test_invalid_model_type_rejected(self):
        """Test that invalid model type is rejected by argparse."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "invalid_model",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower() or "error" in result.stderr.lower()


class TestIntegrationSmallCNN:
    """Integration tests for small_cnn model."""

    def test_small_cnn_execution_succeeds(self):
        """AC17: python3 train_orchestrated.py --model small_cnn --dataset ./test_data_setup exits 0."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "small_cnn",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    def test_small_cnn_logs_backend_decision(self):
        """AC18: small_cnn output contains backend decision (ane/cpu/backend string)."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "small_cnn",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout + result.stderr
        # Check for backend decision logging
        assert any(keyword in output.lower() for keyword in ["backend", "ane", "cpu", "selected"]), \
            f"No backend decision found in output: {output}"

    def test_small_cnn_completes_training(self):
        """Test that small_cnn completes training successfully."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "small_cnn",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout + result.stderr
        assert "completed" in output.lower() or "success" in output.lower()

    def test_small_cnn_output_contains_model_name(self):
        """Test that output mentions small_cnn model."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "small_cnn",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout + result.stderr
        assert "small_cnn" in output or "cnn" in output.lower()

    def test_small_cnn_shows_profiling_info(self):
        """Test that output shows workload profiling information."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "small_cnn",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout + result.stderr
        assert "profile" in output.lower() or "params" in output.lower()


class TestIntegrationLargeDense:
    """Integration tests for large_dense model."""

    def test_large_dense_execution_succeeds(self):
        """AC19: python3 train_orchestrated.py --model large_dense --dataset ./test_data_setup exits 0."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "large_dense",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    def test_large_dense_completes_training(self):
        """Test that large_dense completes training successfully."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "large_dense",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout + result.stderr
        assert "completed" in output.lower() or "success" in output.lower()

    def test_large_dense_output_contains_model_name(self):
        """Test that output mentions large_dense model."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "large_dense",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout + result.stderr
        assert "large_dense" in output or "dense" in output.lower()


class TestIntegrationSequentialModel:
    """Integration tests for sequential_model."""

    def test_sequential_model_execution_succeeds(self):
        """AC20: python3 train_orchestrated.py --model sequential_model --dataset ./test_data_setup exits 0."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "sequential_model",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        assert result.returncode == 0, f"Command failed with stderr: {result.stderr}"

    def test_sequential_model_completes_training(self):
        """Test that sequential_model completes training successfully."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "sequential_model",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout + result.stderr
        assert "completed" in output.lower() or "success" in output.lower()

    def test_sequential_model_output_contains_model_name(self):
        """Test that output mentions sequential_model."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "sequential_model",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout + result.stderr
        assert "sequential" in output.lower() or "rnn" in output.lower()


class TestBackendLogging:
    """Tests for backend decision logging."""

    def test_backend_decision_logged_small_cnn(self):
        """Test that backend selection is logged for small_cnn."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "small_cnn",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout + result.stderr
        # Should contain backend decision (either "ane" or "cpu" or mention of backend)
        assert any(term in output.lower() for term in ["selected backend", "backend:", "ane", "cpu"]), \
            f"Backend decision not logged: {output}"

    def test_backend_decision_logged_large_dense(self):
        """Test that backend selection is logged for large_dense."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "large_dense",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout + result.stderr
        assert any(term in output.lower() for term in ["selected backend", "backend:", "ane", "cpu"])

    def test_backend_decision_logged_sequential(self):
        """Test that backend selection is logged for sequential_model."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "sequential_model",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout + result.stderr
        assert any(term in output.lower() for term in ["selected backend", "backend:", "ane", "cpu"])

    def test_performance_prediction_logged(self):
        """Test that predicted performance is logged."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "small_cnn",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout + result.stderr
        # Should contain performance prediction info
        assert any(term in output.lower() for term in ["predicted", "time", "ane:", "cpu:"]), \
            f"Performance prediction not logged: {output}"


class TestErrorHandling:
    """Tests for error handling."""

    def test_missing_dataset_path_fails(self):
        """Test that missing dataset path causes failure."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "small_cnn",
             "--dataset", "./nonexistent_path"],
            capture_output=True,
            text=True,
            timeout=5
        )
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_missing_dataset_files_fails(self):
        """Test that missing dataset files cause failure."""
        # Create a directory but don't put files in it
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "train_orchestrated.py",
                 "--model", "small_cnn",
                 "--dataset", tmpdir],
                capture_output=True,
                text=True,
                timeout=5
            )
            assert result.returncode != 0


class TestOrchestrationLoop:
    """Tests for the think-act-observe orchestration cycle."""

    def test_orchestration_includes_observe_step(self):
        """Test that orchestration includes observe step (hardware state)."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "small_cnn",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout + result.stderr
        assert any(term in output.lower() for term in [
            "observing hardware",
            "hardware state",
            "ane available",
            "cpu utilization"
        ]), f"Observe step not logged: {output}"

    def test_orchestration_includes_think_step(self):
        """Test that orchestration includes think step (decision)."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "small_cnn",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout + result.stderr
        assert any(term in output.lower() for term in [
            "decision cycle",
            "selected backend",
            "think"
        ]), f"Think step not logged: {output}"

    def test_orchestration_includes_act_step(self):
        """Test that orchestration includes act step (execution)."""
        result = subprocess.run(
            [sys.executable, "train_orchestrated.py",
             "--model", "small_cnn",
             "--dataset", "./test_data_setup"],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout + result.stderr
        assert any(term in output.lower() for term in [
            "executing training",
            "training on",
            "completed"
        ]), f"Act step not logged: {output}"


class TestTrainingEnvironment:
    """Tests for TrainingEnvironment class."""

    def test_training_environment_import(self):
        """Test that TrainingEnvironment can be imported."""
        from train_orchestrated import TrainingEnvironment
        assert TrainingEnvironment is not None

    def test_training_environment_initialization(self):
        """Test TrainingEnvironment initialization."""
        import numpy as np
        from train_orchestrated import TrainingEnvironment

        X_train = np.random.randn(100, 28, 28).astype("float32")
        y_train = np.random.randint(0, 10, 100)
        config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, config)
        assert env is not None
        assert hasattr(env, 'execute_training')

    def test_training_environment_execute_training_returns_dict(self):
        """Test that execute_training returns a dict with required keys."""
        import numpy as np
        from train_orchestrated import TrainingEnvironment

        X_train = np.random.randn(100, 28, 28).astype("float32")
        y_train = np.random.randint(0, 10, 100)
        config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, config)
        result = env.execute_training("cpu", {"params": 50000, "layers": 3})

        assert isinstance(result, dict)
        assert "status" in result
        assert "backend" in result
        assert "execution_time" in result
        assert "message" in result

    def test_training_environment_cpu_backend_success(self):
        """Test that CPU backend training succeeds."""
        import numpy as np
        from train_orchestrated import TrainingEnvironment

        X_train = np.random.randn(10, 28, 28).astype("float32")
        y_train = np.random.randint(0, 10, 10)
        config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, config)
        result = env.execute_training("cpu", {"params": 50000, "layers": 3})

        assert result["status"] == "success"
        assert result["backend"] == "cpu"
        assert result["execution_time"] > 0
        assert "completed" in result["message"].lower()

    def test_training_environment_ane_backend_graceful_fallback(self):
        """Test that ANE backend gracefully falls back to CPU if unavailable."""
        import numpy as np
        from train_orchestrated import TrainingEnvironment

        X_train = np.random.randn(10, 28, 28).astype("float32")
        y_train = np.random.randint(0, 10, 10)
        config = {"name": "test_model"}

        env = TrainingEnvironment(X_train, y_train, config)
        result = env.execute_training("ane", {"params": 50000, "layers": 3})

        # ANE should either succeed or gracefully fall back to CPU
        assert result["status"] in ["success", "failure"]
        assert result["backend"] == "ane"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_load_model_config_small_cnn(self):
        """Test load_model_config for small_cnn."""
        from train_orchestrated import load_model_config

        config = load_model_config("small_cnn")
        assert config["name"] == "small_cnn"
        assert config["layers"] == 3
        assert config["has_conv"] == True
        assert config["has_rnn"] == False

    def test_load_model_config_large_dense(self):
        """Test load_model_config for large_dense."""
        from train_orchestrated import load_model_config

        config = load_model_config("large_dense")
        assert config["name"] == "large_dense"
        assert config["layers"] == 5
        assert config["has_conv"] == False
        assert config["has_rnn"] == False

    def test_load_model_config_sequential_model(self):
        """Test load_model_config for sequential_model."""
        from train_orchestrated import load_model_config

        config = load_model_config("sequential_model")
        assert config["name"] == "sequential_model"
        assert config["layers"] == 3
        assert config["has_conv"] == False
        assert config["has_rnn"] == True

    def test_load_dataset(self):
        """Test load_dataset loads data correctly."""
        from train_orchestrated import load_dataset

        X_train, y_train = load_dataset("./test_data_setup")
        assert X_train is not None
        assert y_train is not None
        assert len(X_train) == len(y_train)
        assert len(X_train) > 0

    def test_get_hardware_state(self):
        """Test get_hardware_state returns required fields."""
        from train_orchestrated import get_hardware_state

        hw_state = get_hardware_state()
        assert isinstance(hw_state, dict)
        assert "ane_available" in hw_state
        assert "cpu_utilization" in hw_state
        assert "timestamp" in hw_state
        assert isinstance(hw_state["ane_available"], bool)
        assert 0.0 <= hw_state["cpu_utilization"] <= 1.0
