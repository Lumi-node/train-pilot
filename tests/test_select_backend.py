"""
Unit tests for HardwareOrchestrator.select_backend() method.

Tests the 5-rule decision cascade with exact confidence values and boundary conditions.
"""

import pytest
from hardware_orchestrator import HardwareOrchestrator


class TestSelectBackendReturnType:
    """AC7: Verify return type is (str, float) tuple."""

    def test_returns_tuple(self):
        """select_backend returns a tuple."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 3, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.3, 'timestamp': 1234567890.0}

        result = agent.select_backend(profile, hw_state)

        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"

    def test_returns_str_backend(self):
        """First element of tuple is str."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 3, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.3, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert isinstance(backend, str), f"Backend must be str, got {type(backend)}"

    def test_returns_float_confidence(self):
        """Second element of tuple is float."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 3, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.3, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert isinstance(confidence, float), f"Confidence must be float, got {type(confidence)}"

    def test_backend_in_valid_set(self):
        """Backend is either 'ane' or 'cpu'."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 3, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.3, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend in ['ane', 'cpu'], f"Backend must be 'ane' or 'cpu', got '{backend}'"

    def test_confidence_in_valid_range(self):
        """Confidence is in [0.0, 1.0]."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 3, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.3, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert 0.0 <= confidence <= 1.0, f"Confidence must be in [0.0, 1.0], got {confidence}"


class TestRule1AneUnavailable:
    """AC8: Rule 1 - When ANE unavailable, return CPU with confidence 1.0."""

    def test_rule_1_basic(self):
        """When ane_available=False, return ('cpu', 1.0)."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 3, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}
        hw_state = {'ane_available': False, 'cpu_utilization': 0.3, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'cpu', f"Expected 'cpu', got '{backend}'"
        assert confidence == 1.0, f"Expected 1.0, got {confidence}"

    def test_rule_1_high_utilization(self):
        """Rule 1 applies even when CPU utilization is high."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 3, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}
        hw_state = {'ane_available': False, 'cpu_utilization': 0.9, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'cpu'
        assert confidence == 1.0

    def test_rule_1_with_conv_model(self):
        """Rule 1 applies even with conv-heavy model."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 5, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}
        hw_state = {'ane_available': False, 'cpu_utilization': 0.1, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'cpu'
        assert confidence == 1.0

    def test_rule_1_with_dense_model(self):
        """Rule 1 applies even with dense model."""
        agent = HardwareOrchestrator()
        profile = {'params': 1000000, 'layers': 5, 'has_conv': False, 'has_rnn': False, 'depth': 8, 'avg_layer_size': 10000}
        hw_state = {'ane_available': False, 'cpu_utilization': 0.1, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'cpu'
        assert confidence == 1.0


class TestRule2RnnModels:
    """AC10: Rule 2 - When has_rnn=True, return CPU with confidence 0.95."""

    def test_rule_2_basic(self):
        """When has_rnn=True, return ('cpu', 0.95)."""
        agent = HardwareOrchestrator()
        profile = {'params': 250000, 'layers': 3, 'has_conv': False, 'has_rnn': True, 'depth': 8, 'avg_layer_size': 128}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.1, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'cpu', f"Expected 'cpu', got '{backend}'"
        assert confidence == 0.95, f"Expected 0.95, got {confidence}"

    def test_rule_2_high_ane_available_low_util(self):
        """Rule 2 applies even when ANE available and CPU utilization low."""
        agent = HardwareOrchestrator()
        profile = {'params': 250000, 'layers': 3, 'has_conv': False, 'has_rnn': True, 'depth': 8, 'avg_layer_size': 128}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.1, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'cpu'
        assert confidence == 0.95

    def test_rule_2_with_conv_and_rnn(self):
        """Rule 2 applies when has_rnn=True, even if has_conv=True."""
        agent = HardwareOrchestrator()
        profile = {'params': 200000, 'layers': 5, 'has_conv': True, 'has_rnn': True, 'depth': 10, 'avg_layer_size': 100}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.1, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'cpu'
        assert confidence == 0.95

    def test_rule_2_high_cpu_utilization(self):
        """Rule 2 applies even with high CPU utilization."""
        agent = HardwareOrchestrator()
        profile = {'params': 250000, 'layers': 3, 'has_conv': False, 'has_rnn': True, 'depth': 8, 'avg_layer_size': 128}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.9, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'cpu'
        assert confidence == 0.95


class TestRule3ConvHeavyLowUtil:
    """AC9: Rule 3 - Conv-heavy + low CPU util → ANE with confidence 0.85."""

    def test_rule_3_basic(self):
        """When has_conv=True and cpu_util < 0.5, return ('ane', 0.85)."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 5, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.2, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'ane', f"Expected 'ane', got '{backend}'"
        assert confidence == 0.85, f"Expected 0.85, got {confidence}"

    def test_rule_3_at_threshold_boundary(self):
        """Rule 3 applies when cpu_util is just below 0.5."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 5, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.49, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'ane'
        assert confidence == 0.85

    def test_rule_3_not_at_threshold(self):
        """Rule 3 does not apply when cpu_util >= 0.5."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 5, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.5, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        # Should fall through to Rule 5
        assert backend == 'cpu'
        assert confidence == 0.8

    def test_rule_3_very_low_util(self):
        """Rule 3 applies with very low CPU utilization."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 5, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.01, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'ane'
        assert confidence == 0.85

    def test_rule_3_zero_util(self):
        """Rule 3 applies when CPU utilization is 0."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 5, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.0, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'ane'
        assert confidence == 0.85


class TestRule4DenseVeryLowUtil:
    """Rule 4 - Dense-only + very low CPU util → ANE with confidence 0.7."""

    def test_rule_4_basic(self):
        """When no conv/rnn and cpu_util < 0.3, return ('ane', 0.7)."""
        agent = HardwareOrchestrator()
        profile = {'params': 1000000, 'layers': 5, 'has_conv': False, 'has_rnn': False, 'depth': 8, 'avg_layer_size': 10000}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.2, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'ane', f"Expected 'ane', got '{backend}'"
        assert confidence == 0.7, f"Expected 0.7, got {confidence}"

    def test_rule_4_at_threshold_boundary_below(self):
        """Rule 4 applies when cpu_util is just below 0.3."""
        agent = HardwareOrchestrator()
        profile = {'params': 1000000, 'layers': 5, 'has_conv': False, 'has_rnn': False, 'depth': 8, 'avg_layer_size': 10000}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.2999, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'ane'
        assert confidence == 0.7

    def test_rule_4_not_at_threshold(self):
        """Rule 4 does not apply when cpu_util >= 0.3."""
        agent = HardwareOrchestrator()
        profile = {'params': 1000000, 'layers': 5, 'has_conv': False, 'has_rnn': False, 'depth': 8, 'avg_layer_size': 10000}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.3, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        # Should fall through to Rule 5
        assert backend == 'cpu'
        assert confidence == 0.8

    def test_rule_4_just_above_threshold(self):
        """Rule 4 does not apply when cpu_util is just above 0.3."""
        agent = HardwareOrchestrator()
        profile = {'params': 1000000, 'layers': 5, 'has_conv': False, 'has_rnn': False, 'depth': 8, 'avg_layer_size': 10000}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.3001, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'cpu'
        assert confidence == 0.8

    def test_rule_4_zero_util(self):
        """Rule 4 applies when CPU utilization is 0."""
        agent = HardwareOrchestrator()
        profile = {'params': 1000000, 'layers': 5, 'has_conv': False, 'has_rnn': False, 'depth': 8, 'avg_layer_size': 10000}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.0, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'ane'
        assert confidence == 0.7


class TestRule5DefaultFallback:
    """Rule 5 - Default fallback → CPU with confidence 0.8."""

    def test_rule_5_basic(self):
        """When no other rules match, return ('cpu', 0.8)."""
        agent = HardwareOrchestrator()
        profile = {'params': 1000000, 'layers': 5, 'has_conv': False, 'has_rnn': False, 'depth': 8, 'avg_layer_size': 10000}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.5, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'cpu', f"Expected 'cpu', got '{backend}'"
        assert confidence == 0.8, f"Expected 0.8, got {confidence}"

    def test_rule_5_conv_high_util(self):
        """When has_conv=True but cpu_util >= 0.5, use Rule 5."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 5, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.6, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'cpu'
        assert confidence == 0.8

    def test_rule_5_dense_moderate_util(self):
        """When no conv/rnn but cpu_util >= 0.3, use Rule 5."""
        agent = HardwareOrchestrator()
        profile = {'params': 1000000, 'layers': 5, 'has_conv': False, 'has_rnn': False, 'depth': 8, 'avg_layer_size': 10000}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.5, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'cpu'
        assert confidence == 0.8

    def test_rule_5_high_utilization(self):
        """Rule 5 applies with high CPU utilization."""
        agent = HardwareOrchestrator()
        profile = {'params': 1000000, 'layers': 5, 'has_conv': False, 'has_rnn': False, 'depth': 8, 'avg_layer_size': 10000}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.95, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'cpu'
        assert confidence == 0.8


class TestModelProfiles:
    """Test routing decisions with realistic model profiles."""

    def test_small_cnn_routes_to_ane_low_util(self):
        """Small CNN with low util should route to ANE (Rule 3)."""
        agent = HardwareOrchestrator()
        # small_cnn profile
        profile = agent.profile_workload({"name": "small_cnn"}, 60000)
        hw_state = {'ane_available': True, 'cpu_utilization': 0.3, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'ane'
        assert confidence == 0.85

    def test_sequential_model_routes_to_cpu(self):
        """Sequential model should always route to CPU (Rule 2)."""
        agent = HardwareOrchestrator()
        # sequential_model profile
        profile = agent.profile_workload({"name": "sequential_model"}, 60000)
        hw_state = {'ane_available': True, 'cpu_utilization': 0.1, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'cpu'
        assert confidence == 0.95

    def test_large_dense_low_util_routes_to_ane(self):
        """Large dense model with very low util should route to ANE (Rule 4)."""
        agent = HardwareOrchestrator()
        # large_dense profile
        profile = agent.profile_workload({"name": "large_dense"}, 60000)
        hw_state = {'ane_available': True, 'cpu_utilization': 0.2, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'ane'
        assert confidence == 0.7

    def test_large_dense_moderate_util_routes_to_cpu(self):
        """Large dense model with moderate util should route to CPU (Rule 5)."""
        agent = HardwareOrchestrator()
        # large_dense profile
        profile = agent.profile_workload({"name": "large_dense"}, 60000)
        hw_state = {'ane_available': True, 'cpu_utilization': 0.5, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        assert backend == 'cpu'
        assert confidence == 0.8


class TestEdgeCases:
    """Test edge cases and missing keys."""

    def test_missing_ane_available_defaults_false(self):
        """Missing ane_available defaults to False."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 3, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}
        hw_state = {'cpu_utilization': 0.3, 'timestamp': 1234567890.0}  # missing ane_available

        backend, confidence = agent.select_backend(profile, hw_state)

        # Should apply Rule 1 (ANE unavailable)
        assert backend == 'cpu'
        assert confidence == 1.0

    def test_missing_cpu_utilization_defaults_05(self):
        """Missing cpu_utilization defaults to 0.5."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 3, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}
        hw_state = {'ane_available': True, 'timestamp': 1234567890.0}  # missing cpu_utilization

        backend, confidence = agent.select_backend(profile, hw_state)

        # has_conv=True but cpu_util >= 0.5, so Rule 5
        assert backend == 'cpu'
        assert confidence == 0.8

    def test_missing_has_conv_defaults_false(self):
        """Missing has_conv defaults to False."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 3, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}  # missing has_conv
        hw_state = {'ane_available': True, 'cpu_utilization': 0.2, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        # No conv, no rnn, low util -> Rule 4
        assert backend == 'ane'
        assert confidence == 0.7

    def test_missing_has_rnn_defaults_false(self):
        """Missing has_rnn defaults to False."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 3, 'has_conv': True, 'depth': 5, 'avg_layer_size': 100}  # missing has_rnn
        hw_state = {'ane_available': True, 'cpu_utilization': 0.3, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        # has_conv=True, low util -> Rule 3
        assert backend == 'ane'
        assert confidence == 0.85

    def test_empty_profile_dict(self):
        """Empty profile dict uses defaults."""
        agent = HardwareOrchestrator()
        profile = {}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.2, 'timestamp': 1234567890.0}

        backend, confidence = agent.select_backend(profile, hw_state)

        # All defaults: no conv, no rnn, low util -> Rule 4
        assert backend == 'ane'
        assert confidence == 0.7

    def test_empty_hardware_state_dict(self):
        """Empty hardware_state dict uses defaults."""
        agent = HardwareOrchestrator()
        profile = {'params': 50000, 'layers': 3, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 100}
        hw_state = {}

        backend, confidence = agent.select_backend(profile, hw_state)

        # Defaults: ane_available=False -> Rule 1
        assert backend == 'cpu'
        assert confidence == 1.0


class TestConfidenceValues:
    """Verify exact confidence values for each rule."""

    def test_rule_1_confidence_1_0(self):
        """Rule 1 confidence is exactly 1.0."""
        agent = HardwareOrchestrator()
        profile = {'has_conv': True, 'has_rnn': False}
        hw_state = {'ane_available': False, 'cpu_utilization': 0.3}

        backend, confidence = agent.select_backend(profile, hw_state)
        assert confidence == 1.0

    def test_rule_2_confidence_0_95(self):
        """Rule 2 confidence is exactly 0.95."""
        agent = HardwareOrchestrator()
        profile = {'has_conv': False, 'has_rnn': True}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.1}

        backend, confidence = agent.select_backend(profile, hw_state)
        assert confidence == 0.95

    def test_rule_3_confidence_0_85(self):
        """Rule 3 confidence is exactly 0.85."""
        agent = HardwareOrchestrator()
        profile = {'has_conv': True, 'has_rnn': False}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.3}

        backend, confidence = agent.select_backend(profile, hw_state)
        assert confidence == 0.85

    def test_rule_4_confidence_0_7(self):
        """Rule 4 confidence is exactly 0.7."""
        agent = HardwareOrchestrator()
        profile = {'has_conv': False, 'has_rnn': False}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.2}

        backend, confidence = agent.select_backend(profile, hw_state)
        assert confidence == 0.7

    def test_rule_5_confidence_0_8(self):
        """Rule 5 confidence is exactly 0.8."""
        agent = HardwareOrchestrator()
        profile = {'has_conv': False, 'has_rnn': False}
        hw_state = {'ane_available': True, 'cpu_utilization': 0.5}

        backend, confidence = agent.select_backend(profile, hw_state)
        assert confidence == 0.8


class TestStrictLessThanOperator:
    """Verify that < operator is used (not <=) for thresholds."""

    def test_rule_3_threshold_is_exclusive(self):
        """Rule 3 uses < not <= for 0.5 threshold."""
        agent = HardwareOrchestrator()
        profile = {'has_conv': True, 'has_rnn': False}

        # Just below threshold -> Rule 3 (ANE)
        backend_below, conf_below = agent.select_backend(profile, {'ane_available': True, 'cpu_utilization': 0.49})

        # At threshold -> Rule 5 (CPU)
        backend_at, conf_at = agent.select_backend(profile, {'ane_available': True, 'cpu_utilization': 0.5})

        assert backend_below == 'ane' and conf_below == 0.85
        assert backend_at == 'cpu' and conf_at == 0.8

    def test_rule_4_threshold_is_exclusive(self):
        """Rule 4 uses < not <= for 0.3 threshold."""
        agent = HardwareOrchestrator()
        profile = {'has_conv': False, 'has_rnn': False}

        # Just below threshold -> Rule 4 (ANE)
        backend_below, conf_below = agent.select_backend(profile, {'ane_available': True, 'cpu_utilization': 0.2999})

        # At threshold -> Rule 5 (CPU)
        backend_at, conf_at = agent.select_backend(profile, {'ane_available': True, 'cpu_utilization': 0.3})

        assert backend_below == 'ane' and conf_below == 0.7
        assert backend_at == 'cpu' and conf_at == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
