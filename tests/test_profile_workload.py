"""
Unit tests for HardwareOrchestrator.profile_workload() method.
"""

import pytest
from hardware_orchestrator import HardwareOrchestrator


class TestProfileWorkloadReturnStructure:
    """Test AC3: profile_workload returns dict with all 6 required keys."""

    def test_returns_dict(self):
        """Verify profile_workload returns a dict."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "small_cnn"}, 60000)
        assert isinstance(profile, dict)

    def test_has_all_required_keys(self):
        """Verify dict contains all 6 required keys."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "small_cnn"}, 60000)
        required_keys = {"params", "layers", "has_conv", "has_rnn", "depth", "avg_layer_size"}
        actual_keys = set(profile.keys())
        assert actual_keys == required_keys, f"Missing keys: {required_keys - actual_keys}"

    def test_correct_types(self):
        """Verify all values have correct types."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "small_cnn"}, 60000)

        assert isinstance(profile["params"], int), f"params must be int, got {type(profile['params'])}"
        assert isinstance(profile["layers"], int), f"layers must be int, got {type(profile['layers'])}"
        assert isinstance(profile["has_conv"], bool), f"has_conv must be bool, got {type(profile['has_conv'])}"
        assert isinstance(profile["has_rnn"], bool), f"has_rnn must be bool, got {type(profile['has_rnn'])}"
        assert isinstance(profile["depth"], int), f"depth must be int, got {type(profile['depth'])}"
        assert isinstance(profile["avg_layer_size"], int), f"avg_layer_size must be int, got {type(profile['avg_layer_size'])}"


class TestSmallCNN:
    """Test AC4: profile_workload correctly classifies small_cnn."""

    def test_small_cnn_basic_config(self):
        """Test small_cnn with basic config."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "small_cnn"}, 60000)

        assert profile["has_conv"] is True
        assert profile["has_rnn"] is False
        assert profile["layers"] == 3

    def test_small_cnn_full_values(self):
        """Test small_cnn returns exact values from lookup table."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "small_cnn"}, 60000)

        assert profile["params"] == 50000
        assert profile["layers"] == 3
        assert profile["has_conv"] is True
        assert profile["has_rnn"] is False
        assert profile["depth"] == 5
        assert profile["avg_layer_size"] == 1000

    def test_small_cnn_with_explicit_has_conv(self):
        """Test small_cnn with explicit has_conv in config."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload(
            {"name": "small_cnn", "layers": 3, "has_conv": True},
            60000
        )

        assert profile["has_conv"] is True
        assert profile["has_rnn"] is False
        assert profile["layers"] == 3


class TestLargeDense:
    """Test AC5: profile_workload correctly classifies large_dense."""

    def test_large_dense_basic_config(self):
        """Test large_dense with basic config."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "large_dense"}, 60000)

        assert profile["has_conv"] is False
        assert profile["has_rnn"] is False
        assert profile["layers"] == 5

    def test_large_dense_full_values(self):
        """Test large_dense returns exact values from lookup table."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "large_dense"}, 60000)

        assert profile["params"] == 1000000
        assert profile["layers"] == 5
        assert profile["has_conv"] is False
        assert profile["has_rnn"] is False
        assert profile["depth"] == 8
        assert profile["avg_layer_size"] == 10000

    def test_large_dense_with_explicit_has_conv(self):
        """Test large_dense with explicit has_conv in config."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload(
            {"name": "large_dense", "layers": 5, "has_conv": False},
            60000
        )

        assert profile["has_conv"] is False
        assert profile["has_rnn"] is False
        assert profile["layers"] == 5


class TestSequentialModel:
    """Test AC6: profile_workload correctly classifies sequential_model."""

    def test_sequential_model_basic_config(self):
        """Test sequential_model with basic config."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "sequential_model"}, 60000)

        assert profile["has_rnn"] is True
        assert profile["has_conv"] is False
        assert profile["layers"] == 3

    def test_sequential_model_full_values(self):
        """Test sequential_model returns exact values from lookup table."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "sequential_model"}, 60000)

        assert profile["params"] == 250000
        assert profile["layers"] == 3
        assert profile["has_conv"] is False
        assert profile["has_rnn"] is True
        assert profile["depth"] == 12
        assert profile["avg_layer_size"] == 5000

    def test_sequential_model_with_explicit_has_rnn(self):
        """Test sequential_model with explicit has_rnn in config."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload(
            {"name": "sequential_model", "layers": 3, "has_rnn": True},
            60000
        )

        assert profile["has_rnn"] is True
        assert profile["has_conv"] is False
        assert profile["layers"] == 3


class TestFallbackBehavior:
    """Test fallback behavior for unknown model names."""

    def test_unknown_model_name_returns_default(self):
        """Test unknown model uses fallback formula."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "unknown_model", "layers": 4}, 60000)

        assert isinstance(profile, dict)
        assert set(profile.keys()) == {"params", "layers", "has_conv", "has_rnn", "depth", "avg_layer_size"}

    def test_fallback_depth_formula(self):
        """Test fallback formula: depth = layers * 2."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "unknown_model", "layers": 4}, 60000)

        assert profile["depth"] == 4 * 2

    def test_fallback_avg_layer_size_formula(self):
        """Test fallback formula: avg_layer_size = params / layers."""
        agent = HardwareOrchestrator()
        params = 120000
        layers = 4
        profile = agent.profile_workload(
            {"name": "unknown_model", "layers": layers, "params": params},
            60000
        )

        expected_avg = params // layers
        assert profile["avg_layer_size"] == expected_avg

    def test_unknown_model_with_explicit_properties(self):
        """Test unknown model respects explicit has_conv/has_rnn."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload(
            {"name": "custom_model", "layers": 3, "has_conv": True, "has_rnn": False, "params": 80000},
            60000
        )

        assert profile["has_conv"] is True
        assert profile["has_rnn"] is False
        assert profile["layers"] == 3


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_config_dict(self):
        """Test with empty config dict."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({}, 60000)

        # Should return dict with fallback values
        assert isinstance(profile, dict)
        assert set(profile.keys()) == {"params", "layers", "has_conv", "has_rnn", "depth", "avg_layer_size"}

    def test_missing_name_key(self):
        """Test with config missing 'name' key."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"layers": 2}, 60000)

        # Should use fallback
        assert isinstance(profile, dict)
        assert set(profile.keys()) == {"params", "layers", "has_conv", "has_rnn", "depth", "avg_layer_size"}

    def test_missing_layers_key(self):
        """Test with config missing 'layers' key."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "unknown"}, 60000)

        # Should use default layers value
        assert isinstance(profile, dict)
        assert profile["layers"] == 1  # Default value

    def test_zero_dataset_size(self):
        """Test with dataset_size = 0."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "small_cnn"}, 0)

        # Should still work (dataset_size is not used in current implementation)
        assert profile["has_conv"] is True

    def test_negative_layers(self):
        """Test with negative layers value."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "unknown", "layers": -1}, 60000)

        # Should handle gracefully
        assert isinstance(profile, dict)
        # depth = layers * 2
        assert profile["depth"] == -2

    def test_zero_params_fallback(self):
        """Test fallback with zero params."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload({"name": "unknown", "layers": 2, "params": 0}, 60000)

        # Should handle division: max(1, 0 // 2) = max(1, 0) = 1
        assert isinstance(profile, dict)
        assert profile["avg_layer_size"] == 1


class TestConfigOverrides:
    """Test that explicit config values override lookup table defaults."""

    def test_small_cnn_layers_override(self):
        """Test that explicit layers key overrides lookup table."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload(
            {"name": "small_cnn", "layers": 5},
            60000
        )

        # Layers should be overridden
        assert profile["layers"] == 5
        # Other values from table
        assert profile["params"] == 50000
        assert profile["has_conv"] is True

    def test_large_dense_has_conv_override(self):
        """Test that explicit has_conv overrides lookup table."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload(
            {"name": "large_dense", "has_conv": True},
            60000
        )

        # has_conv should be overridden
        assert profile["has_conv"] is True
        # Other values from table
        assert profile["params"] == 1000000
        assert profile["layers"] == 5
        assert profile["has_rnn"] is False

    def test_sequential_model_has_rnn_override(self):
        """Test that explicit has_rnn overrides lookup table."""
        agent = HardwareOrchestrator()
        profile = agent.profile_workload(
            {"name": "sequential_model", "has_rnn": False},
            60000
        )

        # has_rnn should be overridden
        assert profile["has_rnn"] is False
        # Other values from table
        assert profile["params"] == 250000
        assert profile["layers"] == 3
