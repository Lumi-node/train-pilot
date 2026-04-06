"""
Hardware-Aware Neural Network Training Orchestrator.

Autonomously routes neural network training to optimal backends (ANE or CPU)
based on workload characteristics and hardware state.
"""

import math
import time


class HardwareOrchestrator:
    """Autonomous agent that routes neural network training to optimal backend."""

    def __init__(self):
        """Initialize orchestrator with empty observation state."""
        self._last_observation = None
        self._last_decision = None
        self._last_confidence = None

    def profile_workload(self, model_config: dict, dataset_size: int) -> dict:
        """
        Extract workload characteristics without executing training.

        Args:
            model_config: Dict with keys {name, layers, has_conv, has_rnn, ...}
            dataset_size: Number of training samples (int)

        Returns:
            Dict with keys:
            - params: int (total parameters)
            - layers: int (layer count)
            - has_conv: bool (has convolution)
            - has_rnn: bool (has recurrent)
            - depth: int (max sequential depth)
            - avg_layer_size: int (avg units per layer)
        """
        name = model_config.get("name", "unknown")

        # Extract explicit properties if provided
        has_conv = model_config.get("has_conv", False)
        has_rnn = model_config.get("has_rnn", False)
        layers = model_config.get("layers", 1)

        # Model class profiles (hardcoded lookup table)
        profiles = {
            "small_cnn": {
                "params": 50000,
                "layers": 3,
                "has_conv": True,
                "has_rnn": False,
                "depth": 5,
                "avg_layer_size": 1000
            },
            "large_dense": {
                "params": 1000000,
                "layers": 5,
                "has_conv": False,
                "has_rnn": False,
                "depth": 8,
                "avg_layer_size": 10000
            },
            "sequential_model": {
                "params": 250000,
                "layers": 3,
                "has_conv": False,
                "has_rnn": True,
                "depth": 12,
                "avg_layer_size": 5000
            }
        }

        # Use explicit config properties if provided, else lookup table
        if name in profiles:
            profile = profiles[name].copy()
            # Override with explicit config if provided
            profile["has_conv"] = model_config.get("has_conv", profile["has_conv"])
            profile["has_rnn"] = model_config.get("has_rnn", profile["has_rnn"])
            profile["layers"] = model_config.get("layers", profile["layers"])
            return profile
        else:
            # Fallback for unknown models
            return {
                "params": model_config.get("params", 100000),
                "layers": layers,
                "has_conv": has_conv,
                "has_rnn": has_rnn,
                "depth": layers * 2,
                "avg_layer_size": max(1, model_config.get("params", 100000) // max(1, layers))
            }

    def observe(self, hardware_state: dict, workload_profile: dict) -> None:
        """
        Store current observation for think() to use.

        Args:
            hardware_state: Dict with keys ane_available, cpu_utilization, timestamp
            workload_profile: Dict from profile_workload()

        Returns:
            None (stores state internally)
        """
        self._last_observation = {
            "hardware_state": hardware_state.copy(),
            "workload_profile": workload_profile.copy()
        }

    def think(self) -> str:
        """
        Decide backend based on last observation.

        Args:
            None (reads stored observation from observe())

        Returns:
            "ane" or "cpu"
        """
        if self._last_observation is None:
            return "cpu"  # Safe fallback

        hardware_state = self._last_observation["hardware_state"]
        workload_profile = self._last_observation["workload_profile"]

        backend, confidence = self.select_backend(workload_profile, hardware_state)
        self._last_decision = backend
        self._last_confidence = confidence

        return backend

    def predict_performance(self, profile: dict, backend: str) -> float:
        """
        Estimate training time using heuristics grounded in hardware characteristics.

        Args:
            profile: Dict from profile_workload()
            backend: "ane" or "cpu"

        Returns:
            float > 0 and < 3600 (seconds)
        """
        params = profile.get("params", 100000)
        layers = profile.get("layers", 3)
        has_conv = profile.get("has_conv", False)
        has_rnn = profile.get("has_rnn", False)

        # Base computation cost
        base_time = (math.log2(max(1, params)) + layers) / 10.0

        if has_rnn:
            # RNN: CPU much faster
            if backend == "ane":
                return base_time * 1.8
            else:
                return base_time * 1.0
        elif has_conv:
            # Convolution: ANE faster
            if backend == "ane":
                return base_time * 0.4
            else:
                return base_time * 1.0
        else:
            # Dense-only: ANE slightly faster
            if backend == "ane":
                return base_time * 0.8
            else:
                return base_time * 1.0

    def select_backend(self, profile: dict, hardware_state: dict) -> tuple:
        """
        Select optimal backend using deterministic rule cascade.

        Args:
            profile: Dict from profile_workload()
            hardware_state: Dict with ane_available, cpu_utilization, timestamp

        Returns:
            Tuple (backend_name: str, confidence: float)
        """
        has_conv = profile.get("has_conv", False)
        has_rnn = profile.get("has_rnn", False)
        ane_available = hardware_state.get("ane_available", False)
        cpu_utilization = hardware_state.get("cpu_utilization", 0.5)

        # Rule 1: ANE unavailable
        if not ane_available:
            return ("cpu", 1.0)

        # Rule 2: RNN models → CPU
        if has_rnn:
            return ("cpu", 0.95)

        # Rule 3: Conv-heavy + low CPU util → ANE
        if has_conv and cpu_utilization < 0.5:
            return ("ane", 0.85)

        # Rule 4: Dense-only + very low CPU util → ANE
        if not has_conv and not has_rnn and cpu_utilization < 0.3:
            return ("ane", 0.7)

        # Rule 5: Default → CPU
        return ("cpu", 0.8)

    def act(self, action: str, environment) -> dict:
        """
        Execute training on selected backend via environment interface.

        Args:
            action: "ane" or "cpu" (the backend decision from think())
            environment: Object implementing ITrainingEnvironment interface

        Returns:
            Dict with keys: status, backend, execution_time, message
        """
        try:
            # Call environment's execute_training method with the decision
            result = environment.execute_training(
                backend=action,
                workload_profile=self._last_observation.get("workload_profile", {})
                if self._last_observation else {}
            )
            return result
        except Exception as e:
            return {
                "status": "failure",
                "backend": action,
                "execution_time": 0.0,
                "message": f"Execution failed: {str(e)}"
            }
