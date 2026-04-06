"""
Integration tests for HardwareOrchestrator complete think-act-observe cycle.

Tests verify:
1. Profile all three models successfully
2. Observe hardware state, think backend decision, act training execution
3. Routing decisions match expectations (AC21)
4. Speedup calculation and threshold (AC22)
5. End-to-end orchestration works without errors
"""

import pytest
from unittest.mock import Mock
from hardware_orchestrator import HardwareOrchestrator


class TestProfileAllModels:
    """Test 1: Profile all three models successfully."""

    def test_profile_small_cnn(self):
        """Profile small_cnn model."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "small_cnn"}, 60000)

        assert isinstance(profile, dict)
        assert profile["has_conv"] is True
        assert profile["has_rnn"] is False
        assert profile["layers"] == 3
        assert profile["params"] == 50000

    def test_profile_sequential_model(self):
        """Profile sequential_model (RNN)."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "sequential_model"}, 60000)

        assert isinstance(profile, dict)
        assert profile["has_conv"] is False
        assert profile["has_rnn"] is True
        assert profile["layers"] == 3
        assert profile["params"] == 250000

    def test_profile_large_dense(self):
        """Profile large_dense model."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "large_dense"}, 60000)

        assert isinstance(profile, dict)
        assert profile["has_conv"] is False
        assert profile["has_rnn"] is False
        assert profile["layers"] == 5
        assert profile["params"] == 1000000

    def test_all_profiles_have_required_keys(self):
        """Verify all profiles have required keys."""
        agent = HardwareOrchestrator()
        required_keys = {"params", "layers", "has_conv", "has_rnn", "depth", "avg_layer_size"}

        for model_name in ["small_cnn", "sequential_model", "large_dense"]:
            profile = agent.profile_workload({"name": model_name}, 60000)
            assert set(profile.keys()) == required_keys, f"Missing keys for {model_name}"


class TestThinkActObserveCycle:
    """Test 2: Complete observe -> think -> act cycle for each model."""

    def test_cycle_small_cnn(self):
        """Test complete cycle for small_cnn."""
        agent = HardwareOrchestrator()

        # Mock environment
        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "ane",
                "execution_time": 5.0,
                "message": "Training completed on ANE",
            }
        )

        # Profile
        profile = agent.profile_workload({"name": "small_cnn"}, 60000)

        # Observe
        hw_state = {"ane_available": True, "cpu_utilization": 0.2, "timestamp": 0}
        agent.observe(hw_state, profile)

        # Think
        backend = agent.think()
        assert backend in ["ane", "cpu"]

        # Act
        result = agent.act(backend, mock_env)
        assert result["status"] == "success"
        assert "backend" in result
        assert "execution_time" in result
        assert "message" in result
        assert isinstance(result["execution_time"], (int, float))

    def test_cycle_sequential_model(self):
        """Test complete cycle for sequential_model (RNN)."""
        agent = HardwareOrchestrator()

        # Mock environment
        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "cpu",
                "execution_time": 12.5,
                "message": "Training completed on CPU",
            }
        )

        # Profile
        profile = agent.profile_workload({"name": "sequential_model"}, 60000)

        # Observe
        hw_state = {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0}
        agent.observe(hw_state, profile)

        # Think
        backend = agent.think()
        assert backend in ["ane", "cpu"]

        # Act
        result = agent.act(backend, mock_env)
        assert result["status"] == "success"

    def test_cycle_large_dense(self):
        """Test complete cycle for large_dense."""
        agent = HardwareOrchestrator()

        # Mock environment
        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "cpu",
                "execution_time": 15.0,
                "message": "Training completed",
            }
        )

        # Profile
        profile = agent.profile_workload({"name": "large_dense"}, 60000)

        # Observe
        hw_state = {"ane_available": True, "cpu_utilization": 0.5, "timestamp": 0}
        agent.observe(hw_state, profile)

        # Think
        backend = agent.think()
        assert backend in ["ane", "cpu"]

        # Act
        result = agent.act(backend, mock_env)
        assert result["status"] == "success"


class TestRoutingDecisionsAC21:
    """Test 3: AC21 - Routing decisions match expected patterns."""

    def test_small_cnn_routes_to_ane(self):
        """Verify small_cnn (conv-heavy) routes to ANE with low CPU utilization."""
        agent = HardwareOrchestrator()

        # Profile small_cnn
        profile = agent.profile_workload({"name": "small_cnn"}, 60000)
        assert profile["has_conv"] is True
        assert profile["has_rnn"] is False

        # Test with ANE available and low CPU utilization
        hw_state = {"ane_available": True, "cpu_utilization": 0.2, "timestamp": 0}
        backend, confidence = agent.select_backend(profile, hw_state)

        # Should route to ANE
        assert backend == "ane", f"small_cnn should route to ANE, got {backend}"
        assert confidence >= 0.7, f"Confidence should be reasonable, got {confidence}"

    def test_sequential_model_routes_to_cpu(self):
        """Verify sequential_model (RNN) routes to CPU."""
        agent = HardwareOrchestrator()

        # Profile sequential_model
        profile = agent.profile_workload({"name": "sequential_model"}, 60000)
        assert profile["has_rnn"] is True

        # Test with various CPU utilizations
        hw_state = {"ane_available": True, "cpu_utilization": 0.1, "timestamp": 0}
        backend, confidence = agent.select_backend(profile, hw_state)

        # Should always route to CPU for RNN
        assert backend == "cpu", f"sequential_model should route to CPU, got {backend}"
        assert confidence >= 0.9, f"Confidence should be high, got {confidence}"

    def test_large_dense_flexible_routing(self):
        """Verify large_dense routes flexibly based on CPU utilization."""
        agent = HardwareOrchestrator()

        # Profile large_dense
        profile = agent.profile_workload({"name": "large_dense"}, 60000)
        assert profile["has_conv"] is False
        assert profile["has_rnn"] is False

        # Test with low CPU utilization (should route to ANE if implemented)
        hw_state_low = {"ane_available": True, "cpu_utilization": 0.2, "timestamp": 0}
        backend_low, _ = agent.select_backend(profile, hw_state_low)
        assert backend_low in ["ane", "cpu"], f"Backend must be 'ane' or 'cpu', got {backend_low}"

        # Test with high CPU utilization (should route to CPU)
        hw_state_high = {"ane_available": True, "cpu_utilization": 0.6, "timestamp": 0}
        backend_high, _ = agent.select_backend(profile, hw_state_high)
        assert backend_high == "cpu", f"With high CPU util, should route to CPU, got {backend_high}"

    def test_all_models_make_decisions(self):
        """Verify all three models get routing decisions."""
        agent = HardwareOrchestrator()
        hw_state = {"ane_available": True, "cpu_utilization": 0.2, "timestamp": 0}

        for model_name in ["small_cnn", "sequential_model", "large_dense"]:
            profile = agent.profile_workload({"name": model_name}, 60000)
            backend, confidence = agent.select_backend(profile, hw_state)

            assert backend in ["ane", "cpu"], f"{model_name}: backend must be 'ane' or 'cpu'"
            assert 0.0 <= confidence <= 1.0, f"{model_name}: confidence out of range"


class TestSpeedupCalculationAC22:
    """Test 4: AC22 - Speedup calculation and threshold."""

    def test_speedup_formula_for_small_cnn(self):
        """Calculate speedup for small_cnn where ANE is selected."""
        agent = HardwareOrchestrator()

        # Profile small_cnn
        profile = agent.profile_workload({"name": "small_cnn"}, 60000)

        # Get performance predictions
        ane_time = agent.predict_performance(profile, "ane")
        cpu_time = agent.predict_performance(profile, "cpu")

        # Calculate speedup
        speedup_pct = ((cpu_time / ane_time) - 1.0) * 100

        # For conv models, ANE should be faster
        assert ane_time < cpu_time, f"ANE should be faster for conv model: {ane_time} vs {cpu_time}"
        assert speedup_pct > 0, f"Speedup should be positive for conv model: {speedup_pct}%"
        assert speedup_pct >= 20.0, f"small_cnn speedup should be >= 20%: {speedup_pct}%"

    def test_speedup_formula_for_sequential_model(self):
        """Calculate speedup for sequential_model where CPU is selected (no speedup counted)."""
        agent = HardwareOrchestrator()

        # Profile sequential_model
        profile = agent.profile_workload({"name": "sequential_model"}, 60000)

        # Get performance predictions
        ane_time = agent.predict_performance(profile, "ane")
        cpu_time = agent.predict_performance(profile, "cpu")

        # CPU should be faster for RNN
        assert cpu_time < ane_time, f"CPU should be faster for RNN: {cpu_time} vs {ane_time}"

        # Since CPU is selected, speedup calculation would be negative or not counted
        speedup_pct = ((cpu_time / ane_time) - 1.0) * 100
        assert speedup_pct < 0, f"Speedup should be negative when CPU faster: {speedup_pct}%"

    def test_speedup_threshold_met_for_at_least_two_models(self):
        """Verify speedup calculation and count models achieving >= 20% speedup.

        This test validates the speedup calculation mechanism and counts models
        where the agent selects ANE and achieves >=20% speedup.

        Per the architecture:
        - small_cnn (conv): ANE 150%+ speedup (always selected with low CPU util)
        - sequential_model (RNN): CPU always selected (no speedup counted)
        - large_dense (dense): ANE slightly slower (-5%), but agent may select it with low CPU util

        NOTE: Actual achievement of >=2 models with >=20% speedup is subject to
        the hardware state and heuristics. This test documents the speedup calculation.
        """
        agent = HardwareOrchestrator()

        speedups = {}
        model_names = ["small_cnn", "large_dense", "sequential_model"]

        # Use consistent low CPU utilization per AC22 specification
        hw_state = {"ane_available": True, "cpu_utilization": 0.2, "timestamp": 0}

        for model_name in model_names:
            profile = agent.profile_workload({"name": model_name}, 60000)

            # Get backend decision
            selected_backend, _ = agent.select_backend(profile, hw_state)

            # Get performance predictions
            ane_time = agent.predict_performance(profile, "ane")
            cpu_time = agent.predict_performance(profile, "cpu")

            # Calculate speedup using exact AC22 formula: (cpu_time / ane_time - 1) * 100
            speedup_pct = ((cpu_time / ane_time) - 1.0) * 100

            # Record entry per AC22: only if ANE was selected, store the speedup
            if selected_backend == "ane":
                speedups[model_name] = speedup_pct
            else:
                speedups[model_name] = 0.0

        # Count models with >= 20% speedup
        models_with_speedup = sum(1 for v in speedups.values() if v >= 20.0)

        # Verify speedup calculation worked and at least small_cnn achieved threshold
        assert speedups["small_cnn"] >= 20.0, (
            f"small_cnn (conv model) should achieve >=20% speedup with ANE. "
            f"Got {speedups['small_cnn']}%"
        )

        # Log speedup values for review
        # AC22 requires >=2 models with >=20% speedup; document actual count
        assert isinstance(models_with_speedup, int), "Speedup count should be integer"
        assert models_with_speedup >= 2, (
            f"AC22 requires at least 2 models with >=20% speedup. "
            f"Got {models_with_speedup}: {speedups}"
        )

    def test_speedup_calculation_explicit_formula(self):
        """Test speedup calculation uses exact formula: (cpu_time / ane_time - 1) * 100."""
        agent = HardwareOrchestrator()

        profile = agent.profile_workload({"name": "small_cnn"}, 60000)
        ane_time = agent.predict_performance(profile, "ane")
        cpu_time = agent.predict_performance(profile, "cpu")

        # Explicit formula from AC22
        speedup_pct = ((cpu_time / ane_time) - 1.0) * 100

        # Verify formula correctness
        expected_speedup = ((cpu_time / ane_time) - 1.0) * 100
        assert speedup_pct == expected_speedup, f"Formula mismatch: {speedup_pct} vs {expected_speedup}"

        # For conv model, should be positive
        assert speedup_pct > 0, f"Conv model speedup should be positive: {speedup_pct}%"

    def test_speedup_for_models_where_ane_selected(self):
        """Only count speedup for models where ANE is selected."""
        agent = HardwareOrchestrator()
        hw_state = {"ane_available": True, "cpu_utilization": 0.2, "timestamp": 0}

        speedups_with_ane_selected = {}

        for model_name in ["small_cnn", "sequential_model", "large_dense"]:
            profile = agent.profile_workload({"name": model_name}, 60000)
            backend, _ = agent.select_backend(profile, hw_state)

            ane_time = agent.predict_performance(profile, "ane")
            cpu_time = agent.predict_performance(profile, "cpu")

            # Only count if ANE was actually selected
            if backend == "ane":
                speedup_pct = ((cpu_time / ane_time) - 1.0) * 100
                speedups_with_ane_selected[model_name] = speedup_pct

        # Verify small_cnn is in there with positive speedup
        assert "small_cnn" in speedups_with_ane_selected, "small_cnn should have ANE selected"
        assert speedups_with_ane_selected["small_cnn"] >= 20.0, (
            f"small_cnn should have >= 20% speedup: {speedups_with_ane_selected['small_cnn']}%"
        )

    def test_ac22_strict_requirement_at_least_two_models(self):
        """AC22 Strict: At least 2 of 3 models must achieve >=20% speedup.

        This test strictly validates AC22 as specified in the PRD.
        """
        agent = HardwareOrchestrator()
        hw_state = {"ane_available": True, "cpu_utilization": 0.2, "timestamp": 0}

        speedups = {}

        for model_name in ["small_cnn", "large_dense", "sequential_model"]:
            profile = agent.profile_workload({"name": model_name}, 60000)
            backend, _ = agent.select_backend(profile, hw_state)

            ane_time = agent.predict_performance(profile, "ane")
            cpu_time = agent.predict_performance(profile, "cpu")

            # Only count speedup if ANE was selected
            if backend == "ane":
                speedup_pct = ((cpu_time / ane_time) - 1.0) * 100
                speedups[model_name] = speedup_pct
            else:
                speedups[model_name] = 0.0

        # Count how many achieve >= 20% speedup
        models_with_speedup = sum(1 for v in speedups.values() if v >= 20.0)

        # PRD AC22 requirement: at least 2 of 3
        assert models_with_speedup >= 2, (
            f"AC22 requires at least 2 models with >=20% speedup. "
            f"Got {models_with_speedup}: {speedups}"
        )


class TestEdgeCases:
    """Test edge cases: ANE unavailable, high CPU utilization, etc."""

    def test_observe_without_prior_state(self):
        """Test think() without prior observe() returns safe default."""
        agent = HardwareOrchestrator()

        # Call think() without observe()
        backend = agent.think()

        # Should return safe default
        assert backend == "cpu", f"think() without observe() should default to CPU, got {backend}"

    def test_routing_with_ane_unavailable(self):
        """Test routing when ANE is unavailable."""
        agent = HardwareOrchestrator()

        profile = agent.profile_workload({"name": "small_cnn"}, 60000)
        hw_state = {"ane_available": False, "cpu_utilization": 0.1, "timestamp": 0}

        backend, confidence = agent.select_backend(profile, hw_state)

        # Should always select CPU when ANE unavailable
        assert backend == "cpu", f"Should select CPU when ANE unavailable, got {backend}"
        assert confidence == 1.0, f"Confidence should be 1.0 for forced CPU, got {confidence}"

    def test_routing_with_high_cpu_utilization(self):
        """Test routing with high CPU utilization."""
        agent = HardwareOrchestrator()

        profile = agent.profile_workload({"name": "large_dense"}, 60000)
        hw_state = {"ane_available": True, "cpu_utilization": 0.8, "timestamp": 0}

        backend, confidence = agent.select_backend(profile, hw_state)

        # With high CPU utilization, should prefer CPU
        assert backend == "cpu", f"High CPU util should prefer CPU, got {backend}"

    def test_act_without_prior_observe_still_works(self):
        """Test act() without prior observe() still completes successfully."""
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

        # Call act() without observe()
        result = agent.act("cpu", mock_env)

        # Should still return valid result
        assert result["status"] == "success"
        assert "execution_time" in result


class TestCompleteIntegrationScenarios:
    """Test complete end-to-end scenarios matching real usage."""

    def test_small_cnn_full_orchestration(self):
        """Full orchestration for small_cnn: profile -> observe -> think -> predict -> act."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "ane",
                "execution_time": 6.0,
                "message": "Training completed",
            }
        )

        # 1. Profile workload
        model_config = {"name": "small_cnn", "layers": 3, "has_conv": True}
        profile = agent.profile_workload(model_config, 60000)
        assert profile["has_conv"] is True

        # 2. Observe hardware state
        hw_state = {"ane_available": True, "cpu_utilization": 0.2, "timestamp": 0}
        agent.observe(hw_state, profile)

        # 3. Think (decide backend)
        backend = agent.think()
        assert backend == "ane", "small_cnn should route to ANE with low CPU util"

        # 4. Predict performance
        ane_time = agent.predict_performance(profile, "ane")
        cpu_time = agent.predict_performance(profile, "cpu")
        speedup_pct = ((cpu_time / ane_time) - 1.0) * 100
        assert speedup_pct >= 20.0, f"small_cnn should have >= 20% speedup: {speedup_pct}%"

        # 5. Act (execute training)
        result = agent.act(backend, mock_env)
        assert result["status"] == "success"

    def test_sequential_model_full_orchestration(self):
        """Full orchestration for sequential_model: profile -> observe -> think -> act."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "cpu",
                "execution_time": 12.0,
                "message": "Training completed",
            }
        )

        # 1. Profile workload
        model_config = {"name": "sequential_model", "layers": 3, "has_rnn": True}
        profile = agent.profile_workload(model_config, 60000)
        assert profile["has_rnn"] is True

        # 2. Observe hardware state
        hw_state = {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 0}
        agent.observe(hw_state, profile)

        # 3. Think (decide backend)
        backend = agent.think()
        assert backend == "cpu", "sequential_model should always route to CPU"

        # 4. Act (execute training)
        result = agent.act(backend, mock_env)
        assert result["status"] == "success"

    def test_large_dense_full_orchestration(self):
        """Full orchestration for large_dense: profile -> observe -> think -> act."""
        agent = HardwareOrchestrator()

        mock_env = Mock()
        mock_env.execute_training = Mock(
            return_value={
                "status": "success",
                "backend": "cpu",
                "execution_time": 15.0,
                "message": "Training completed",
            }
        )

        # 1. Profile workload
        model_config = {"name": "large_dense", "layers": 5, "has_conv": False}
        profile = agent.profile_workload(model_config, 60000)
        assert profile["has_conv"] is False
        assert profile["has_rnn"] is False

        # 2. Observe hardware state (low CPU util to allow ANE routing)
        hw_state = {"ane_available": True, "cpu_utilization": 0.2, "timestamp": 0}
        agent.observe(hw_state, profile)

        # 3. Think (decide backend)
        backend = agent.think()
        assert backend in ["ane", "cpu"], "large_dense should route to either ANE or CPU"

        # 4. Act (execute training)
        result = agent.act(backend, mock_env)
        assert result["status"] == "success"
