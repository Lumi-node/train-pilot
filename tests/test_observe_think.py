"""
Unit and integration tests for HardwareOrchestrator.observe() and think() methods.

Tests the observation and decision phases of the think-act-observe cycle.
"""

import pytest
from hardware_orchestrator import HardwareOrchestrator


class TestObserveMethod:
    """Tests for observe() method: stores hardware state and workload profile."""

    def test_observe_stores_state(self):
        """observe() stores hardware_state and workload_profile in instance variables."""
        agent = HardwareOrchestrator()
        hw_state = {
            "ane_available": True,
            "cpu_utilization": 0.25,
            "timestamp": 1234567890.0
        }
        profile = {
            "params": 50000,
            "layers": 3,
            "has_conv": True,
            "has_rnn": False,
            "depth": 5,
            "avg_layer_size": 1000
        }

        # Call observe
        agent.observe(hw_state, profile)

        # Verify state is stored
        assert agent._last_observation is not None
        assert "hardware_state" in agent._last_observation
        assert "workload_profile" in agent._last_observation
        assert agent._last_observation["hardware_state"] == hw_state
        assert agent._last_observation["workload_profile"] == profile

    def test_observe_stores_copies_not_references(self):
        """observe() stores copies of dicts, not references."""
        agent = HardwareOrchestrator()
        hw_state = {
            "ane_available": True,
            "cpu_utilization": 0.25,
            "timestamp": 1234567890.0
        }
        profile = {
            "params": 50000,
            "layers": 3,
            "has_conv": True,
            "has_rnn": False,
            "depth": 5,
            "avg_layer_size": 1000
        }

        # Call observe
        agent.observe(hw_state, profile)

        # Modify original dicts
        hw_state["ane_available"] = False
        profile["params"] = 100000

        # Verify stored state is unchanged
        assert agent._last_observation["hardware_state"]["ane_available"] == True
        assert agent._last_observation["workload_profile"]["params"] == 50000

    def test_observe_with_empty_dicts(self):
        """observe() does not crash with empty dicts."""
        agent = HardwareOrchestrator()
        hw_state = {}
        profile = {}

        # Should not raise
        agent.observe(hw_state, profile)

        # State should be stored (even if empty)
        assert agent._last_observation is not None

    def test_observe_overwrites_previous_observation(self):
        """Multiple observe() calls overwrite _last_observation correctly."""
        agent = HardwareOrchestrator()

        # First observation
        hw_state_1 = {"ane_available": True, "cpu_utilization": 0.25, "timestamp": 100}
        profile_1 = {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000}
        agent.observe(hw_state_1, profile_1)
        assert agent._last_observation["workload_profile"]["params"] == 50000

        # Second observation
        hw_state_2 = {"ane_available": False, "cpu_utilization": 0.75, "timestamp": 200}
        profile_2 = {"params": 250000, "layers": 3, "has_conv": False, "has_rnn": True, "depth": 12, "avg_layer_size": 5000}
        agent.observe(hw_state_2, profile_2)

        # Verify only second observation is stored
        assert agent._last_observation["hardware_state"]["ane_available"] == False
        assert agent._last_observation["workload_profile"]["params"] == 250000


class TestThinkMethod:
    """Tests for think() method: reads observation and delegates to select_backend()."""

    def test_think_returns_string(self):
        """think() returns a string."""
        agent = HardwareOrchestrator()
        hw_state = {"ane_available": True, "cpu_utilization": 0.25, "timestamp": 1234567890.0}
        profile = {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000}

        agent.observe(hw_state, profile)
        result = agent.think()

        assert isinstance(result, str)

    def test_think_returns_valid_backend(self):
        """think() returns either 'ane' or 'cpu'."""
        agent = HardwareOrchestrator()
        hw_state = {"ane_available": True, "cpu_utilization": 0.25, "timestamp": 1234567890.0}
        profile = {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000}

        agent.observe(hw_state, profile)
        result = agent.think()

        assert result in ["ane", "cpu"]

    def test_think_before_observe_returns_cpu(self):
        """think() called before observe() returns 'cpu' (safe fallback)."""
        agent = HardwareOrchestrator()

        # Call think without observe
        result = agent.think()

        assert result == "cpu"

    def test_think_reads_from_stored_observation(self):
        """think() reads from _last_observation and calls select_backend()."""
        agent = HardwareOrchestrator()
        hw_state = {"ane_available": True, "cpu_utilization": 0.25, "timestamp": 1234567890.0}
        profile = {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000}

        agent.observe(hw_state, profile)
        result = agent.think()

        # Manually call select_backend with same args to verify consistency
        expected_backend, _ = agent.select_backend(profile, hw_state)

        assert result == expected_backend

    def test_think_stores_decision_and_confidence(self):
        """think() stores decision and confidence in _last_decision and _last_confidence."""
        agent = HardwareOrchestrator()
        hw_state = {"ane_available": True, "cpu_utilization": 0.25, "timestamp": 1234567890.0}
        profile = {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000}

        agent.observe(hw_state, profile)
        backend = agent.think()

        # Verify decision and confidence are stored
        assert agent._last_decision == backend
        assert agent._last_confidence is not None
        assert isinstance(agent._last_confidence, float)
        assert 0.0 <= agent._last_confidence <= 1.0


class TestObserveThinkIntegration:
    """Integration tests: observe() + think() cycle with various profiles."""

    def test_small_cnn_routing(self):
        """small_cnn profile routes to 'ane' with ANE available and low CPU util (AC21)."""
        agent = HardwareOrchestrator()

        # Profile small_cnn
        profile = agent.profile_workload({"name": "small_cnn", "layers": 3, "has_conv": True}, 60000)

        # Hardware state: ANE available, low CPU utilization
        hw_state = {"ane_available": True, "cpu_utilization": 0.2, "timestamp": 0}

        # Observe and think
        agent.observe(hw_state, profile)
        backend = agent.think()

        # small_cnn is conv-heavy, should route to ANE with Rule 3
        assert backend == "ane", f"small_cnn should route to 'ane', got '{backend}'"

    def test_sequential_model_routing(self):
        """sequential_model profile routes to 'cpu' regardless of hardware (AC21)."""
        agent = HardwareOrchestrator()

        # Profile sequential_model
        profile = agent.profile_workload({"name": "sequential_model", "layers": 3, "has_rnn": True}, 60000)

        # Hardware state: ANE available, very low CPU utilization
        hw_state = {"ane_available": True, "cpu_utilization": 0.1, "timestamp": 0}

        # Observe and think
        agent.observe(hw_state, profile)
        backend = agent.think()

        # sequential_model is RNN, should always route to CPU (Rule 2)
        assert backend == "cpu", f"sequential_model should route to 'cpu', got '{backend}'"

    def test_large_dense_low_utilization_routing(self):
        """large_dense with low CPU util routes to 'ane'."""
        agent = HardwareOrchestrator()

        # Profile large_dense
        profile = agent.profile_workload({"name": "large_dense", "layers": 5, "has_conv": False}, 60000)

        # Hardware state: ANE available, very low CPU utilization
        hw_state = {"ane_available": True, "cpu_utilization": 0.2, "timestamp": 0}

        # Observe and think
        agent.observe(hw_state, profile)
        backend = agent.think()

        # large_dense is dense-only with low utilization, should route to ANE (Rule 4)
        assert backend == "ane", f"large_dense with low util should route to 'ane', got '{backend}'"

    def test_large_dense_high_utilization_routing(self):
        """large_dense with high CPU util routes to 'cpu'."""
        agent = HardwareOrchestrator()

        # Profile large_dense
        profile = agent.profile_workload({"name": "large_dense", "layers": 5, "has_conv": False}, 60000)

        # Hardware state: ANE available, high CPU utilization
        hw_state = {"ane_available": True, "cpu_utilization": 0.6, "timestamp": 0}

        # Observe and think
        agent.observe(hw_state, profile)
        backend = agent.think()

        # large_dense with high utilization should route to CPU (Rule 5)
        assert backend == "cpu", f"large_dense with high util should route to 'cpu', got '{backend}'"

    def test_small_cnn_no_ane_routing(self):
        """small_cnn routes to 'cpu' when ANE unavailable."""
        agent = HardwareOrchestrator()

        # Profile small_cnn
        profile = agent.profile_workload({"name": "small_cnn", "layers": 3, "has_conv": True}, 60000)

        # Hardware state: ANE unavailable
        hw_state = {"ane_available": False, "cpu_utilization": 0.2, "timestamp": 0}

        # Observe and think
        agent.observe(hw_state, profile)
        backend = agent.think()

        # No ANE available, should always choose CPU (Rule 1)
        assert backend == "cpu", f"No ANE should route to 'cpu', got '{backend}'"

    def test_observe_think_consistency_multiple_calls(self):
        """Multiple think() calls return consistent results for same observation."""
        agent = HardwareOrchestrator()

        hw_state = {"ane_available": True, "cpu_utilization": 0.25, "timestamp": 1234567890.0}
        profile = {"params": 50000, "layers": 3, "has_conv": True, "has_rnn": False, "depth": 5, "avg_layer_size": 1000}

        agent.observe(hw_state, profile)

        # Call think() multiple times
        result1 = agent.think()
        result2 = agent.think()
        result3 = agent.think()

        # All results should be identical
        assert result1 == result2 == result3

    def test_observe_think_matches_select_backend(self):
        """think() result matches select_backend() output for stored observation."""
        agent = HardwareOrchestrator()

        hw_state = {"ane_available": True, "cpu_utilization": 0.3, "timestamp": 1234567890.0}
        profile = {
            "params": 50000,
            "layers": 3,
            "has_conv": True,
            "has_rnn": False,
            "depth": 5,
            "avg_layer_size": 1000
        }

        # Observe
        agent.observe(hw_state, profile)

        # Think
        backend_from_think = agent.think()

        # Directly call select_backend
        backend_from_select, _ = agent.select_backend(profile, hw_state)

        # Should match
        assert backend_from_think == backend_from_select
