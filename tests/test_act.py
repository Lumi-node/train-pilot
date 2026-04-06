"""
Unit tests for HardwareOrchestrator.act() method.

Tests the act() method which executes training on a selected backend via
environment interface. Covers AC2 requirement: act() method exists and is
callable with correct argument count (2 args: action, environment).
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from hardware_orchestrator import HardwareOrchestrator


class TestActMethodExists:
    """Test AC2: act() method exists and is callable with correct signature."""

    def test_act_method_exists(self):
        """Verify act() method exists on HardwareOrchestrator."""
        agent = HardwareOrchestrator()
        assert hasattr(agent, "act"), "HardwareOrchestrator must have act() method"

    def test_act_is_callable(self):
        """Verify act() is callable."""
        agent = HardwareOrchestrator()
        assert callable(agent.act), "act() must be callable"

    def test_act_accepts_two_arguments(self):
        """Verify act() accepts exactly 2 arguments (action, environment)."""
        import inspect

        agent = HardwareOrchestrator()
        sig = inspect.signature(agent.act)
        # Count parameters excluding self
        params = list(sig.parameters.keys())
        assert len(params) == 2, f"act() must have 2 parameters, got {len(params)}: {params}"

    def test_act_parameter_names(self):
        """Verify act() parameter names are 'action' and 'environment'."""
        import inspect

        agent = HardwareOrchestrator()
        sig = inspect.signature(agent.act)
        params = list(sig.parameters.keys())
        assert params[0] == "action", f"First parameter should be 'action', got {params[0]}"
        assert params[1] == "environment", f"Second parameter should be 'environment', got {params[1]}"


class TestActSuccessfulExecution:
    """Test successful execution: environment returns success dict."""

    def test_act_calls_environment_execute_training(self):
        """Verify act() calls environment.execute_training()."""
        agent = HardwareOrchestrator()

        # Create mock environment
        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "cpu",
                "execution_time": 10.5,
                "message": "Training completed successfully",
            }
        )

        # Setup observation
        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        # Call act
        result = agent.act("cpu", mock_env)

        # Verify environment.execute_training was called
        mock_env.execute_training.assert_called_once()

    def test_act_returns_success_dict_structure(self):
        """Verify act() returns dict with required keys on success."""
        agent = HardwareOrchestrator()

        # Create mock environment that returns success
        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "cpu",
                "execution_time": 10.5,
                "message": "Training completed successfully",
            }
        )

        # Setup observation
        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("cpu", mock_env)

        # Verify all required keys present
        assert isinstance(result, dict), "act() must return dict"
        required_keys = {"status", "backend", "execution_time", "message"}
        actual_keys = set(result.keys())
        assert actual_keys == required_keys, f"Missing keys: {required_keys - actual_keys}"

    def test_act_returns_correct_types_on_success(self):
        """Verify act() returns correct types in result dict."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "cpu",
                "execution_time": 10.5,
                "message": "Training completed successfully",
            }
        )

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("cpu", mock_env)

        assert isinstance(result["status"], str), f"status must be str, got {type(result['status'])}"
        assert isinstance(result["backend"], str), f"backend must be str, got {type(result['backend'])}"
        assert isinstance(result["execution_time"], (int, float)), f"execution_time must be numeric, got {type(result['execution_time'])}"
        assert isinstance(result["message"], str), f"message must be str, got {type(result['message'])}"

    def test_act_with_ane_backend(self):
        """Test act() with 'ane' backend."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "ane",
                "execution_time": 5.2,
                "message": "Training completed on ANE",
            }
        )

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.2, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("ane", mock_env)

        assert result["status"] == "success"
        assert result["backend"] == "ane"
        assert result["execution_time"] > 0

    def test_act_with_cpu_backend(self):
        """Test act() with 'cpu' backend."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "cpu",
                "execution_time": 12.3,
                "message": "Training completed on CPU",
            }
        )

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.5, "timestamp": 0},
            {"params": 250000, "layers": 3, "has_conv": False, "has_rnn": True, "depth": 12, "avg_layer_size": 5000},
        )

        result = agent.act("cpu", mock_env)

        assert result["status"] == "success"
        assert result["backend"] == "cpu"


class TestActFailureHandling:
    """Test failure handling: environment raises exception or returns failure dict."""

    def test_act_catches_environment_exception(self):
        """Verify act() catches exceptions from environment.execute_training()."""
        agent = HardwareOrchestrator()

        # Create mock environment that raises exception
        mock_env = Mock()
        mock_env.execute_training = Mock(side_effect=RuntimeError("Training failed"))

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        # Should not raise; should return failure dict
        result = agent.act("cpu", mock_env)

        assert isinstance(result, dict)
        assert result["status"] == "failure"

    def test_act_returns_failure_dict_on_exception(self):
        """Verify act() returns proper failure dict when exception occurs."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(side_effect=RuntimeError("Backend error"))

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("cpu", mock_env)

        assert result["status"] == "failure"
        assert result["backend"] == "cpu"
        assert isinstance(result["execution_time"], (int, float))
        assert isinstance(result["message"], str)

    def test_act_failure_message_includes_error(self):
        """Verify failure message includes the exception information."""
        agent = HardwareOrchestrator()

        error_msg = "Custom training error"
        mock_env = Mock()
        mock_env.execute_training = Mock(side_effect=RuntimeError(error_msg))

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("cpu", mock_env)

        assert error_msg in result["message"]

    def test_act_failure_execution_time_zero(self):
        """Verify failure dict has execution_time=0.0 on exception."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(side_effect=ValueError("Invalid value"))

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("cpu", mock_env)

        assert result["execution_time"] == 0.0

    def test_act_handles_attribute_error(self):
        """Verify act() handles missing execute_training method gracefully."""
        agent = HardwareOrchestrator()

        # Create mock without execute_training method
        mock_env = Mock(spec=[])  # Empty spec = no attributes

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("cpu", mock_env)

        assert result["status"] == "failure"
        assert result["backend"] == "cpu"

    def test_act_handles_type_error(self):
        """Verify act() handles TypeError from environment."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(side_effect=TypeError("Invalid argument type"))

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("cpu", mock_env)

        assert result["status"] == "failure"


class TestActReturnDictStructure:
    """Test return dict structure and types in all scenarios."""

    def test_return_dict_has_exactly_four_keys(self):
        """Verify returned dict always has exactly 4 keys."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "cpu",
                "execution_time": 10.0,
                "message": "Done",
            }
        )

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("cpu", mock_env)
        assert len(result) == 4, f"Result dict must have exactly 4 keys, got {len(result)}"

    def test_return_dict_status_is_string(self):
        """Verify status key is always a string."""
        agent = HardwareOrchestrator()

        # Success case
        mock_env_success = Mock()
        mock_env_success.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "cpu",
                "execution_time": 10.0,
                "message": "Done",
            }
        )

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("cpu", mock_env_success)
        assert isinstance(result["status"], str)

        # Failure case
        mock_env_failure = Mock()
        mock_env_failure.execute_training = Mock(side_effect=Exception("Error"))

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("cpu", mock_env_failure)
        assert isinstance(result["status"], str)

    def test_return_dict_backend_is_string(self):
        """Verify backend key is always a string."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "ane",
                "execution_time": 5.0,
                "message": "Done",
            }
        )

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.2, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("ane", mock_env)
        assert isinstance(result["backend"], str)

    def test_return_dict_execution_time_is_numeric(self):
        """Verify execution_time is always numeric (int or float)."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "cpu",
                "execution_time": 15.7,
                "message": "Completed",
            }
        )

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("cpu", mock_env)
        assert isinstance(result["execution_time"], (int, float))

    def test_return_dict_message_is_string(self):
        """Verify message key is always a string."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "cpu",
                "execution_time": 12.0,
                "message": "Training finished",
            }
        )

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("cpu", mock_env)
        assert isinstance(result["message"], str)


class TestActEdgeCases:
    """Test edge cases: unknown backend, missing environment method, etc."""

    def test_act_with_unknown_backend_string(self):
        """Test act() with unknown backend string."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "failure",
                "backend": "unknown",
                "execution_time": 0.0,
                "message": "Unknown backend",
            }
        )

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("unknown_backend", mock_env)

        # Should still return valid dict structure
        assert isinstance(result, dict)
        assert len(result) == 4

    def test_act_without_prior_observation(self):
        """Test act() called without prior observe() call."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "cpu",
                "execution_time": 10.0,
                "message": "Done",
            }
        )

        # No observe() call before act()
        result = agent.act("cpu", mock_env)

        # Should still work and pass empty workload_profile
        assert isinstance(result, dict)
        mock_env.execute_training.assert_called_once()

    def test_act_with_none_observation(self):
        """Test act() when _last_observation is None."""
        agent = HardwareOrchestrator()
        agent._last_observation = None

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "cpu",
                "execution_time": 10.0,
                "message": "Done",
            }
        )

        result = agent.act("cpu", mock_env)

        assert isinstance(result, dict)
        # execute_training should be called with empty workload_profile
        call_args = mock_env.execute_training.call_args
        assert call_args is not None

    def test_act_environment_returns_none(self):
        """Test act() when environment returns None (error case)."""
        agent = HardwareOrchestrator()

        # Environment returns None instead of dict
        mock_env = Mock()
        mock_env.execute_training = Mock(return_value=None)

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("cpu", mock_env)

        # Should return None from environment
        assert result is None

    def test_act_environment_returns_incomplete_dict(self):
        """Test act() when environment returns dict with missing keys."""
        agent = HardwareOrchestrator()

        # Environment returns incomplete dict
        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "cpu",
                # Missing execution_time and message
            }
        )

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("cpu", mock_env)

        # act() just delegates, so it returns what environment returns
        assert result == {"status": "success", "backend": "cpu"}

    def test_act_with_environment_method_exception_on_call(self):
        """Test act() when environment.execute_training raises during call."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(side_effect=IOError("File not found"))

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("cpu", mock_env)

        assert result["status"] == "failure"
        assert "File not found" in result["message"]

    def test_act_maintains_backend_in_result(self):
        """Test that act() preserves the action parameter in result backend field."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "ane",
                "execution_time": 5.0,
                "message": "Success on ANE",
            }
        )

        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.2, "timestamp": 0},
            {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000},
        )

        result = agent.act("ane", mock_env)

        # Result should preserve the backend from environment
        assert result["backend"] == "ane"

    def test_act_passes_correct_arguments_to_environment(self):
        """Test act() passes action and workload_profile to environment.execute_training()."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "cpu",
                "execution_time": 10.0,
                "message": "Done",
            }
        )

        profile = {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000}
        agent.observe(
            {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0},
            profile,
        )

        agent.act("cpu", mock_env)

        # Verify execute_training was called with correct arguments
        mock_env.execute_training.assert_called_once()
        call_kwargs = mock_env.execute_training.call_args[1]
        assert call_kwargs["backend"] == "cpu"
        assert call_kwargs["workload_profile"] == profile


class TestActIntegrationScenarios:
    """Test act() in realistic integration scenarios."""

    def test_act_after_think_decision(self):
        """Test act() called after think() makes a decision."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "ane",
                "execution_time": 5.0,
                "message": "Training on ANE",
            }
        )

        # Setup observation
        hw_state = {"ane_available": True, "cpu_utilization": 0.2, "timestamp": 0}
        profile = {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000}

        agent.observe(hw_state, profile)

        # Make decision
        backend = agent.think()

        # Execute decision
        result = agent.act(backend, mock_env)

        assert result["status"] == "success"
        assert result["backend"] == backend

    def test_act_complete_workflow_conv_model(self):
        """Test complete workflow: profile → observe → think → act for conv model."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "ane",
                "execution_time": 6.5,
                "message": "Training completed on ANE",
            }
        )

        # Profile workload
        model_config = {"name": "small_cnn", "layers": 3, "has_conv": True}
        profile = agent.profile_workload(model_config, 60000)

        # Observe hardware state
        hw_state = {"ane_available": True, "cpu_utilization": 0.2, "timestamp": 0}
        agent.observe(hw_state, profile)

        # Make decision
        backend = agent.think()

        # Execute
        result = agent.act(backend, mock_env)

        assert result["status"] == "success"

    def test_act_complete_workflow_rnn_model(self):
        """Test complete workflow for RNN model (should use CPU)."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "cpu",
                "execution_time": 12.3,
                "message": "Training completed on CPU",
            }
        )

        # Profile workload
        model_config = {"name": "sequential_model", "layers": 3, "has_rnn": True}
        profile = agent.profile_workload(model_config, 60000)

        # Observe hardware state
        hw_state = {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0}
        agent.observe(hw_state, profile)

        # Make decision (should be CPU for RNN)
        backend = agent.think()

        # Execute
        result = agent.act(backend, mock_env)

        assert result["status"] == "success"
        assert backend == "cpu"
