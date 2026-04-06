"""
Unit tests for predict_performance() method.

Tests the heuristic formula: base_time = (log2(params) + layers) / 10
Tests hardware-specific multipliers for conv, dense, and RNN models.
Tests range validation and edge cases.
"""

import pytest
import math
from hardware_orchestrator import HardwareOrchestrator


class TestPredictPerformanceRangeValidation:
    """Test AC11: Returns numeric float in valid range (0, 3600)."""

    def test_returns_float(self):
        """Verify return type is numeric."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 50000,
            "layers": 3,
            "has_conv": True,
            "has_rnn": False
        }

        for backend in ["ane", "cpu"]:
            result = agent.predict_performance(profile, backend)
            assert isinstance(result, (int, float)), f"Expected numeric, got {type(result)}"

    def test_returns_positive(self):
        """Verify return value is positive."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 50000,
            "layers": 3,
            "has_conv": True,
            "has_rnn": False
        }

        for backend in ["ane", "cpu"]:
            result = agent.predict_performance(profile, backend)
            assert result > 0, f"Expected positive, got {result}"

    def test_returns_less_than_3600(self):
        """Verify return value is less than 3600."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 50000,
            "layers": 3,
            "has_conv": True,
            "has_rnn": False
        }

        for backend in ["ane", "cpu"]:
            result = agent.predict_performance(profile, backend)
            assert result < 3600, f"Expected < 3600, got {result}"

    def test_small_params_in_range(self):
        """Test edge case: very small params."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 1,
            "layers": 1,
            "has_conv": False,
            "has_rnn": False
        }

        for backend in ["ane", "cpu"]:
            result = agent.predict_performance(profile, backend)
            assert 0 < result < 3600, f"Expected in range, got {result}"

    def test_large_params_in_range(self):
        """Test edge case: very large params (1M)."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 1000000,
            "layers": 5,
            "has_conv": False,
            "has_rnn": False
        }

        for backend in ["ane", "cpu"]:
            result = agent.predict_performance(profile, backend)
            assert 0 < result < 3600, f"Expected in range, got {result}"

    def test_typical_params_in_range(self):
        """Test typical case."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 100000,
            "layers": 3,
            "has_conv": False,
            "has_rnn": False
        }

        for backend in ["ane", "cpu"]:
            result = agent.predict_performance(profile, backend)
            assert 0 < result < 3600, f"Expected in range, got {result}"


class TestPredictPerformanceConvModels:
    """Test AC12: ANE advantage for conv models (ane_time < cpu_time)."""

    def test_conv_ane_faster_than_cpu(self):
        """Verify ANE is faster for conv models."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 50000,
            "layers": 5,
            "has_conv": True,
            "has_rnn": False
        }

        ane_time = agent.predict_performance(profile, "ane")
        cpu_time = agent.predict_performance(profile, "cpu")

        assert ane_time < cpu_time, \
            f"ANE must be faster for conv models: ane={ane_time}, cpu={cpu_time}"

    def test_conv_ane_multiplier(self):
        """Verify ANE multiplier is 0.4x for conv models."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 50000,
            "layers": 5,
            "has_conv": True,
            "has_rnn": False
        }

        ane_time = agent.predict_performance(profile, "ane")
        cpu_time = agent.predict_performance(profile, "cpu")

        # ANE should be 0.4x of CPU
        expected_ratio = 0.4
        actual_ratio = ane_time / cpu_time
        assert abs(actual_ratio - expected_ratio) < 0.01, \
            f"Expected ratio {expected_ratio}, got {actual_ratio}"

    def test_conv_ane_speedup_significant(self):
        """Verify speedup is significant (30-50% faster)."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 50000,
            "layers": 5,
            "has_conv": True,
            "has_rnn": False
        }

        ane_time = agent.predict_performance(profile, "ane")
        cpu_time = agent.predict_performance(profile, "cpu")

        # Speedup = (cpu_time / ane_time - 1) * 100
        speedup_pct = ((cpu_time / ane_time) - 1.0) * 100
        assert speedup_pct >= 20, \
            f"Expected >= 20% speedup, got {speedup_pct}%"

    def test_multiple_conv_models(self):
        """Test ANE advantage across different conv models."""
        agent = HardwareOrchestrator()
        test_cases = [
            {"params": 10000, "layers": 2},
            {"params": 50000, "layers": 3},
            {"params": 100000, "layers": 4},
            {"params": 500000, "layers": 5},
        ]

        for params_layers in test_cases:
            profile = {
                **params_layers,
                "has_conv": True,
                "has_rnn": False
            }
            ane_time = agent.predict_performance(profile, "ane")
            cpu_time = agent.predict_performance(profile, "cpu")
            assert ane_time < cpu_time, \
                f"ANE must be faster for conv: {params_layers}, ane={ane_time}, cpu={cpu_time}"


class TestPredictPerformanceRNNModels:
    """Test AC13: CPU advantage for RNN models (cpu_time < ane_time)."""

    def test_rnn_cpu_faster_than_ane(self):
        """Verify CPU is faster for RNN models."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 250000,
            "layers": 3,
            "has_conv": False,
            "has_rnn": True
        }

        ane_time = agent.predict_performance(profile, "ane")
        cpu_time = agent.predict_performance(profile, "cpu")

        assert cpu_time < ane_time, \
            f"CPU must be faster for RNN models: cpu={cpu_time}, ane={ane_time}"

    def test_rnn_ane_multiplier(self):
        """Verify ANE multiplier is 1.8x for RNN models."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 250000,
            "layers": 3,
            "has_conv": False,
            "has_rnn": True
        }

        ane_time = agent.predict_performance(profile, "ane")
        cpu_time = agent.predict_performance(profile, "cpu")

        # ANE should be 1.8x of CPU
        expected_ratio = 1.8
        actual_ratio = ane_time / cpu_time
        assert abs(actual_ratio - expected_ratio) < 0.01, \
            f"Expected ratio {expected_ratio}, got {actual_ratio}"

    def test_rnn_cpu_significantly_faster(self):
        """Verify CPU is significantly faster (45%+ faster)."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 250000,
            "layers": 3,
            "has_conv": False,
            "has_rnn": True
        }

        ane_time = agent.predict_performance(profile, "ane")
        cpu_time = agent.predict_performance(profile, "cpu")

        # Speedup = (ane_time / cpu_time - 1) * 100
        speedup_pct = ((ane_time / cpu_time) - 1.0) * 100
        assert speedup_pct >= 40, \
            f"Expected >= 40% slower on ANE, got {speedup_pct}%"

    def test_multiple_rnn_models(self):
        """Test CPU advantage across different RNN models."""
        agent = HardwareOrchestrator()
        test_cases = [
            {"params": 50000, "layers": 2},
            {"params": 100000, "layers": 3},
            {"params": 250000, "layers": 3},
            {"params": 500000, "layers": 4},
        ]

        for params_layers in test_cases:
            profile = {
                **params_layers,
                "has_conv": False,
                "has_rnn": True
            }
            ane_time = agent.predict_performance(profile, "ane")
            cpu_time = agent.predict_performance(profile, "cpu")
            assert cpu_time < ane_time, \
                f"CPU must be faster for RNN: {params_layers}, cpu={cpu_time}, ane={ane_time}"


class TestPredictPerformanceDenseModels:
    """Test dense models (no conv, no RNN) - ANE and CPU similar."""

    def test_dense_ane_slightly_slower(self):
        """Verify ANE is slightly slower for dense models."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 1000000,
            "layers": 5,
            "has_conv": False,
            "has_rnn": False
        }

        ane_time = agent.predict_performance(profile, "ane")
        cpu_time = agent.predict_performance(profile, "cpu")

        # ANE should be ~1.0x, CPU should be ~0.95x
        # So ANE should be slightly slower
        assert ane_time > cpu_time, \
            f"ANE should be slightly slower for dense: ane={ane_time}, cpu={cpu_time}"

    def test_dense_ane_multiplier(self):
        """Verify ANE multiplier is 1.0x for dense models."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 1000000,
            "layers": 5,
            "has_conv": False,
            "has_rnn": False
        }

        ane_time = agent.predict_performance(profile, "ane")
        base_time = (math.log2(1000000) + 5) / 10.0

        # ANE should be 1.0x of base_time
        assert abs(ane_time - base_time * 1.0) < 0.01, \
            f"Expected ANE to be 1.0x base_time, got {ane_time / base_time}x"

    def test_dense_cpu_multiplier(self):
        """Verify CPU multiplier is 0.95x for dense models."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 1000000,
            "layers": 5,
            "has_conv": False,
            "has_rnn": False
        }

        cpu_time = agent.predict_performance(profile, "cpu")
        base_time = (math.log2(1000000) + 5) / 10.0

        # CPU should be 0.95x of base_time
        assert abs(cpu_time - base_time * 0.95) < 0.01, \
            f"Expected CPU to be 0.95x base_time, got {cpu_time / base_time}x"

    def test_dense_similar_performance(self):
        """Verify ANE and CPU are within 10% for dense models."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 1000000,
            "layers": 5,
            "has_conv": False,
            "has_rnn": False
        }

        ane_time = agent.predict_performance(profile, "ane")
        cpu_time = agent.predict_performance(profile, "cpu")

        ratio = ane_time / cpu_time
        # Should be close to 1.0 / 0.95 ≈ 1.053
        assert 1.0 < ratio < 1.1, \
            f"Expected ANE/CPU ratio near 1.05, got {ratio}"


class TestBaseTimeFormula:
    """Test the base time formula: (log2(params) + layers) / 10."""

    def test_base_time_formula_small_model(self):
        """Verify base time formula with small model."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 1024,  # log2(1024) = 10
            "layers": 5,
            "has_conv": False,
            "has_rnn": False
        }

        cpu_time = agent.predict_performance(profile, "cpu")
        expected_base = (10 + 5) / 10.0  # = 1.5
        expected_cpu = expected_base * 0.95

        assert abs(cpu_time - expected_cpu) < 0.01, \
            f"Expected {expected_cpu}, got {cpu_time}"

    def test_base_time_formula_large_model(self):
        """Verify base time formula with large model."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 1000000,  # log2(1000000) ≈ 19.93
            "layers": 10,
            "has_conv": False,
            "has_rnn": False
        }

        cpu_time = agent.predict_performance(profile, "cpu")
        expected_base = (math.log2(1000000) + 10) / 10.0
        expected_cpu = expected_base * 0.95

        assert abs(cpu_time - expected_cpu) < 0.01, \
            f"Expected {expected_cpu}, got {cpu_time}"

    def test_base_time_increases_with_params(self):
        """Verify base time increases with more parameters."""
        agent = HardwareOrchestrator()

        profile_small = {
            "params": 10000,
            "layers": 3,
            "has_conv": False,
            "has_rnn": False
        }

        profile_large = {
            "params": 1000000,
            "layers": 3,
            "has_conv": False,
            "has_rnn": False
        }

        time_small = agent.predict_performance(profile_small, "cpu")
        time_large = agent.predict_performance(profile_large, "cpu")

        assert time_large > time_small, \
            f"Larger model should take more time: {time_small} vs {time_large}"

    def test_base_time_increases_with_layers(self):
        """Verify base time increases with more layers."""
        agent = HardwareOrchestrator()

        profile_shallow = {
            "params": 100000,
            "layers": 2,
            "has_conv": False,
            "has_rnn": False
        }

        profile_deep = {
            "params": 100000,
            "layers": 10,
            "has_conv": False,
            "has_rnn": False
        }

        time_shallow = agent.predict_performance(profile_shallow, "cpu")
        time_deep = agent.predict_performance(profile_deep, "cpu")

        assert time_deep > time_shallow, \
            f"Deeper model should take more time: {time_shallow} vs {time_deep}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_params_handled(self):
        """Verify zero params is handled gracefully."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 0,
            "layers": 3,
            "has_conv": False,
            "has_rnn": False
        }

        result = agent.predict_performance(profile, "cpu")
        assert 0 < result < 3600, f"Should handle zero params, got {result}"

    def test_one_param(self):
        """Verify single param (log2(1)=0) is handled."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 1,
            "layers": 1,
            "has_conv": False,
            "has_rnn": False
        }

        result = agent.predict_performance(profile, "cpu")
        expected_base = (0 + 1) / 10.0  # = 0.1
        expected = expected_base * 0.95
        assert abs(result - expected) < 0.01, \
            f"Expected {expected}, got {result}"

    def test_one_layer(self):
        """Verify single layer is handled."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 100000,
            "layers": 1,
            "has_conv": False,
            "has_rnn": False
        }

        result = agent.predict_performance(profile, "cpu")
        assert 0 < result < 3600, f"Should handle single layer, got {result}"

    def test_both_conv_and_rnn_prefers_rnn(self):
        """Verify RNN check takes precedence if both flags set."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 100000,
            "layers": 3,
            "has_conv": True,
            "has_rnn": True
        }

        ane_time = agent.predict_performance(profile, "ane")
        cpu_time = agent.predict_performance(profile, "cpu")

        # If both are set, should use RNN multipliers (1.8x for ANE)
        expected_ane = (math.log2(100000) + 3) / 10.0 * 1.8
        assert abs(ane_time - expected_ane) < 0.01, \
            f"Expected RNN multiplier when both set, got {ane_time}"

    def test_missing_profile_keys_use_defaults(self):
        """Verify missing profile keys use sensible defaults."""
        agent = HardwareOrchestrator()
        profile = {
            # Minimal profile - missing keys should use defaults
        }

        result = agent.predict_performance(profile, "cpu")
        assert 0 < result < 3600, f"Should handle minimal profile, got {result}"

    def test_same_backend_consistency(self):
        """Verify calling same backend twice gives same result."""
        agent = HardwareOrchestrator()
        profile = {
            "params": 100000,
            "layers": 3,
            "has_conv": True,
            "has_rnn": False
        }

        result1 = agent.predict_performance(profile, "ane")
        result2 = agent.predict_performance(profile, "ane")

        assert result1 == result2, \
            f"Same input should give same output: {result1} vs {result2}"
