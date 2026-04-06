#!/usr/bin/env python3
"""
Entry point: Hardware-orchestrated training with backend selection.

Usage:
    python3 train_orchestrated.py --model {small_cnn|large_dense|sequential_model} --dataset <path>

Exit codes:
    0 = success
    1 = failure
"""

import sys
import argparse
import os
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingEnvironment:
    """Concrete implementation of training execution environment.

    Implements the ITrainingEnvironment interface with execute_training()
    method that handles backend-specific training execution (ANE or CPU).
    """

    def __init__(self, X_train, y_train, model_config: dict):
        """
        Initialize training environment with data and model config.

        Args:
            X_train: Training data (numpy array, shape (N, 28, 28))
            y_train: Training labels (numpy array, shape (N,))
            model_config: Model configuration dict with keys like 'name', 'layers', etc.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.model_config = model_config

    def execute_training(self, backend: str, workload_profile: dict) -> dict:
        """
        Execute training on specified backend.

        Args:
            backend: "ane" or "cpu" (the decision from think())
            workload_profile: Dict from HardwareOrchestrator.profile_workload()

        Returns:
            Dict with keys:
            - status: "success" or "failure" (str)
            - backend: backend used (str)
            - execution_time: actual training time in seconds (float)
            - message: human-readable result (str)

        Raises:
            Should not raise; catches all exceptions and returns failure dict
        """
        start_time = time.time()

        try:
            if backend == "ane":
                self._train_on_ane()
            elif backend == "cpu":
                self._train_on_cpu()
            else:
                raise ValueError(f"Unknown backend: {backend}")

            execution_time = time.time() - start_time
            return {
                "status": "success",
                "backend": backend,
                "execution_time": execution_time,
                "message": f"Training completed on {backend.upper()} in {execution_time:.2f}s"
            }
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "status": "failure",
                "backend": backend,
                "execution_time": execution_time,
                "message": f"Training failed: {str(e)}"
            }

    def _train_on_ane(self):
        """
        Training implementation for ANE backend.

        Attempts to use ane_trainer module. If unavailable, gracefully
        falls back to CPU training.
        """
        try:
            from ane_trainer.models import build_model
            from ane_trainer.core import train_step
            import torch

            # Build model
            input_size = 28 * 28
            hidden_size = 128
            output_size = 10
            model = build_model(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size
            )

            # Setup training (simplified)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            loss_fn = torch.nn.CrossEntropyLoss()

            # One epoch on ANE
            X_flat = self.X_train.reshape(-1, 28 * 28).astype("float32")
            for i in range(0, len(X_flat), 32):
                x_batch = X_flat[i:i+32]
                y_batch = self.y_train[i:i+32]
                train_step(model, x_batch, y_batch, optimizer, loss_fn)

            logger.info("Training completed on ANE")
        except ImportError:
            logger.warning("ANE trainer not available, falling back to CPU")
            self._train_on_cpu()

    def _train_on_cpu(self):
        """
        Training implementation for CPU backend.

        Uses standard PyTorch training on CPU backend.
        """
        import torch
        import numpy as np

        input_size = 28 * 28
        hidden_size = 128
        output_size = 10

        # Simple feedforward model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(input_size, hidden_size)
                self.fc2 = torch.nn.Linear(hidden_size, output_size)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        # One epoch on CPU
        X_flat = self.X_train.reshape(-1, 28 * 28).astype("float32")
        for i in range(0, len(X_flat), 32):
            x_batch = torch.FloatTensor(X_flat[i:i+32])
            y_batch = torch.LongTensor(self.y_train[i:i+32])

            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info("Training completed on CPU")


def get_hardware_state() -> dict:
    """Observe current hardware state."""
    import psutil
    try:
        import ane_trainer
        ane_available = True
    except ImportError:
        ane_available = False

    cpu_util = psutil.cpu_percent(interval=0.1) / 100.0

    return {
        "ane_available": ane_available,
        "cpu_utilization": min(1.0, cpu_util),
        "timestamp": time.time()
    }


def load_model_config(model_type: str) -> dict:
    """Load predefined model configuration."""
    configs = {
        "small_cnn": {
            "name": "small_cnn",
            "layers": 3,
            "has_conv": True,
            "has_rnn": False
        },
        "large_dense": {
            "name": "large_dense",
            "layers": 5,
            "has_conv": False,
            "has_rnn": False
        },
        "sequential_model": {
            "name": "sequential_model",
            "layers": 3,
            "has_conv": False,
            "has_rnn": True
        }
    }
    return configs[model_type]


def load_dataset(dataset_path: str) -> tuple:
    """Load training data from directory."""
    import numpy as np

    x_path = os.path.join(dataset_path, "X_train.npy")
    y_path = os.path.join(dataset_path, "y_train.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Dataset files not found in {dataset_path}")

    X_train = np.load(x_path)
    y_train = np.load(y_path)

    return X_train, y_train


def main():
    """Main orchestration loop: observe → think → act."""
    parser = argparse.ArgumentParser(
        description="Hardware-orchestrated neural network training"
    )
    parser.add_argument("--model", required=True,
                       choices=["small_cnn", "large_dense", "sequential_model"],
                       help="Model type")
    parser.add_argument("--dataset", required=True,
                       help="Path to dataset directory")

    try:
        args = parser.parse_args()

        # Step 1: Load model config and data
        logger.info(f"Loading model: {args.model}")
        model_config = load_model_config(args.model)

        logger.info(f"Loading dataset from: {args.dataset}")
        X_train, y_train = load_dataset(args.dataset)
        dataset_size = len(X_train)

        # Step 2: Create orchestrator and environment
        from hardware_orchestrator import HardwareOrchestrator
        agent = HardwareOrchestrator()
        environment = TrainingEnvironment(X_train, y_train, model_config)

        # Step 3: Profile workload
        logger.info("Profiling workload...")
        profile = agent.profile_workload(model_config, dataset_size)
        logger.info(f"  Profile: {profile}")

        # Step 4: Observe hardware state
        logger.info("Observing hardware state...")
        hw_state = get_hardware_state()
        logger.info(f"  ANE available: {hw_state['ane_available']}")
        logger.info(f"  CPU utilization: {hw_state['cpu_utilization']:.2%}")

        # Step 5: OBSERVE → THINK cycle
        logger.info("Agent decision cycle...")
        agent.observe(hw_state, profile)
        backend = agent.think()
        logger.info(f"  Selected backend: {backend}")

        # Step 6: Predict performance
        ane_time = agent.predict_performance(profile, "ane")
        cpu_time = agent.predict_performance(profile, "cpu")
        logger.info(f"  Predicted time - ANE: {ane_time:.2f}s, CPU: {cpu_time:.2f}s")

        # Step 7: ACT → Execute training on selected backend
        logger.info(f"Executing training on {backend.upper()}...")
        result = agent.act(backend, environment)

        if result["status"] == "success":
            logger.info(f"Training result: {result['message']}")
            return 0
        else:
            logger.error(f"Training result: {result['message']}")
            return 1

    except Exception as e:
        logger.error(f"Orchestration failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
