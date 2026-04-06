"""
Test module: Basic import and structure validation for HardwareOrchestrator.

Tests:
- AC1: Module imports without error
- AC2: Required methods exist with correct signatures
"""

import inspect
import pytest


def test_import_hardware_orchestrator():
    """AC1: Module imports without error."""
    try:
        from hardware_orchestrator import HardwareOrchestrator
        assert HardwareOrchestrator is not None
    except ImportError as e:
        pytest.fail(f"Failed to import HardwareOrchestrator: {e}")


def test_instantiate_hardware_orchestrator():
    """Test that HardwareOrchestrator can be instantiated."""
    from hardware_orchestrator import HardwareOrchestrator

    agent = HardwareOrchestrator()
    assert agent is not None
    assert isinstance(agent, HardwareOrchestrator)


def test_internal_state_initialization():
    """Test that internal state is properly initialized."""
    from hardware_orchestrator import HardwareOrchestrator

    agent = HardwareOrchestrator()

    # Verify internal state attributes exist and are initialized to None
    assert hasattr(agent, '_last_observation')
    assert hasattr(agent, '_last_decision')
    assert hasattr(agent, '_last_confidence')

    assert agent._last_observation is None
    assert agent._last_decision is None
    assert agent._last_confidence is None


def test_required_methods_exist_and_callable():
    """AC2: All 6 required methods exist and are callable."""
    from hardware_orchestrator import HardwareOrchestrator

    agent = HardwareOrchestrator()
    required_methods = [
        'profile_workload',
        'observe',
        'think',
        'act',
        'predict_performance',
        'select_backend'
    ]

    for method_name in required_methods:
        assert hasattr(agent, method_name), f"Missing method: {method_name}"
        method = getattr(agent, method_name)
        assert callable(method), f"{method_name} is not callable"


def test_profile_workload_signature():
    """Test profile_workload has correct signature (2 parameters)."""
    from hardware_orchestrator import HardwareOrchestrator

    agent = HardwareOrchestrator()
    sig = inspect.signature(agent.profile_workload)
    params = list(sig.parameters.keys())

    assert len(params) == 2, f"profile_workload must take 2 args, has {len(params)}"
    assert params[0] == 'model_config', f"First param should be model_config, got {params[0]}"
    assert params[1] == 'dataset_size', f"Second param should be dataset_size, got {params[1]}"


def test_observe_signature():
    """Test observe has correct signature (2 parameters)."""
    from hardware_orchestrator import HardwareOrchestrator

    agent = HardwareOrchestrator()
    sig = inspect.signature(agent.observe)
    params = list(sig.parameters.keys())

    assert len(params) == 2, f"observe must take 2 args, has {len(params)}"
    assert params[0] == 'hardware_state'
    assert params[1] == 'workload_profile'


def test_think_signature():
    """Test think has correct signature (0 parameters)."""
    from hardware_orchestrator import HardwareOrchestrator

    agent = HardwareOrchestrator()
    sig = inspect.signature(agent.think)
    params = list(sig.parameters.keys())

    assert len(params) == 0, f"think must take 0 args, has {len(params)}"


def test_act_signature():
    """Test act has correct signature (2 parameters)."""
    from hardware_orchestrator import HardwareOrchestrator

    agent = HardwareOrchestrator()
    sig = inspect.signature(agent.act)
    params = list(sig.parameters.keys())

    assert len(params) == 2, f"act must take 2 args, has {len(params)}"
    assert params[0] == 'action'
    assert params[1] == 'environment'


def test_predict_performance_signature():
    """Test predict_performance has correct signature (2 parameters)."""
    from hardware_orchestrator import HardwareOrchestrator

    agent = HardwareOrchestrator()
    sig = inspect.signature(agent.predict_performance)
    params = list(sig.parameters.keys())

    assert len(params) == 2, f"predict_performance must take 2 args, has {len(params)}"
    assert params[0] == 'profile'
    assert params[1] == 'backend'


def test_select_backend_signature():
    """Test select_backend has correct signature (2 parameters)."""
    from hardware_orchestrator import HardwareOrchestrator

    agent = HardwareOrchestrator()
    sig = inspect.signature(agent.select_backend)
    params = list(sig.parameters.keys())

    assert len(params) == 2, f"select_backend must take 2 args, has {len(params)}"
    assert params[0] == 'profile'
    assert params[1] == 'hardware_state'
