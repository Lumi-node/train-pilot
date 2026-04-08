<p align="center">
  <img src="assets/hero.jpg" alt="TrainPilot" width="900">
</p>

<h1 align="center">TrainPilot</h1>

<p align="center">
  <strong>Hardware-aware agent to orchestrate and optimize NN training workloads.</strong>
</p>

<p align="center">
  <a href="https://github.com/Lumi-node/train-pilot"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <a href="https://github.com/Lumi-node/train-pilot"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python Version"></a>
  <a href="https://github.com/Lumi-node/train-pilot"><img src="https://img.shields.io/badge/Tests-11%2B-green.svg" alt="Tests"></a>
</p>

---

TrainPilot is a specialized agent designed to intelligently orchestrate and optimize the execution of neural network training workloads across heterogeneous hardware environments, specifically targeting Apple Neural Engine (ANE) and CPU backends. It addresses the complex challenge of routing ML tasks to the most performant accelerator available, a problem currently unaddressed by commercial tooling.

The core functionality revolves around a ReasoningAgent that observes workload characteristics—such as model architecture complexity, dataset size, and real-time hardware utilization—to make informed decisions. It dynamically selects the optimal training backend, aiming for significant speedups over naive or random hardware assignment.

---

## Quick Start

```bash
pip install train-pilot
```

```bash
# Run end-to-end orchestration
python train_orchestrated.py --model small_cnn --dataset ./test_data_setup

# Exit code 0 = success, 1 = failure
```

Or use the components directly in Python:

```python
from hardware_orchestrator import HardwareOrchestrator
from train_orchestrated import TrainingEnvironment
import numpy as np
import time

# Initialize orchestrator and environment
agent = HardwareOrchestrator()
X_train = np.random.rand(100, 28, 28)
y_train = np.random.randint(0, 10, 100)
environment = TrainingEnvironment(X_train, y_train, {"name": "small_cnn", "layers": 3, "has_conv": True, "has_rnn": False})

# Profile workload
profile = agent.profile_workload({"name": "small_cnn", "layers": 3, "has_conv": True, "has_rnn": False}, 100)

# Observe hardware and decide backend
hw_state = {"ane_available": False, "cpu_utilization": 0.3, "timestamp": time.time()}
agent.observe(hw_state, profile)
backend = agent.think()

# Execute training
result = agent.act(backend, environment)
print(f"Training result: {result['message']}")
```

## What Can You Do?

### Workload Profiling
The agent can analyze a given model and dataset to extract crucial complexity metrics necessary for performance prediction.

```python
from hardware_orchestrator import HardwareOrchestrator

orchestrator = HardwareOrchestrator()
profile = orchestrator.profile_workload(
    {"name": "small_cnn", "layers": 3, "has_conv": True, "has_rnn": False},
    dataset_size=10000
)
print(f"Model parameters: {profile['params']}, Depth: {profile['depth']}")
```

### Performance Prediction and Selection
Based on workload profile and hardware state, TrainPilot estimates training time and selects the best backend (ANE or CPU).

```python
# Assuming profile was gathered previously
hw_state = {"ane_available": True, "cpu_utilization": 0.2, "timestamp": time.time()}
ane_time = orchestrator.predict_performance(profile, "ane")
cpu_time = orchestrator.predict_performance(profile, "cpu")
backend, confidence = orchestrator.select_backend(profile, hw_state)
print(f"ANE: {ane_time:.2f}s, CPU: {cpu_time:.2f}s => Selected: {backend} (confidence: {confidence})")
```

## Architecture

TrainPilot operates around a central `HardwareOrchestrator` agent housed within `hardware_orchestrator.py`. This agent acts as the decision-maker, consuming inputs from workload analysis and real-time hardware monitoring.

The core loop follows an **Observe → Think → Act** pattern:

1. **Input:** Model config + dataset size $\rightarrow$ `profile_workload()`
2. **Profile:** Extract metrics (params, layers, conv/rnn architecture) → dict with performance characteristics
3. **Observe:** Read hardware state (ANE available, CPU utilization) → `observe(hw_state, profile)`
4. **Think:** Deterministic rule cascade selects optimal backend → `think()` returns "ane" or "cpu"
5. **Predict:** Estimate execution time for each backend → `predict_performance(profile, backend)`
6. **Act:** Execute training on selected backend via `TrainingEnvironment` → `act(backend, environment)`
7. **Execution:** `TrainingEnvironment` routes to ANE trainer or PyTorch CPU backend

```
┌─────────────────────────────────────────────────────────────┐
│            HardwareOrchestrator (Decision Loop)             │
├─────────────────────────────────────────────────────────────┤
│  observe(hw_state, profile) ──→ think() ──→ act(backend)   │
│                                                              │
│  Rules: ANE unavailable? → CPU                             │
│         RNN model? → CPU (0.95 confidence)                 │
│         Conv + low CPU? → ANE (0.85 confidence)            │
│         Default → CPU                                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                   ┌───────▼────────┐
                   │ TrainingEnv    │
                   ├────────────────┤
                   │ execute_train  │
                   │ (backend, cfg) │
                   └───────┬────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
    ┌─────▼────┐                   ┌──────▼──────┐
    │ ANE Path │                   │ CPU Path    │
    │ (PyTorch)│                   │ (PyTorch)   │
    └──────────┘                   └─────────────┘
```

## API Reference

### `HardwareOrchestrator`
Core decision-making agent for backend selection.

**Constructor:**
- `__init__()`: Initialize orchestrator with empty observation state.

**Methods:**
- `profile_workload(model_config: dict, dataset_size: int) -> dict`: Extracts workload characteristics. Returns dict with keys: `params`, `layers`, `has_conv`, `has_rnn`, `depth`, `avg_layer_size`.
  ```python
  model_config = {"name": "small_cnn", "layers": 3, "has_conv": True, "has_rnn": False}
  profile = orchestrator.profile_workload(model_config, dataset_size=10000)
  # {'params': 50000, 'layers': 3, 'has_conv': True, 'has_rnn': False, 'depth': 5, 'avg_layer_size': 1000}
  ```

- `observe(hardware_state: dict, workload_profile: dict) -> None`: Store current observation (called before `think()`).
  ```python
  hw_state = {"ane_available": True, "cpu_utilization": 0.3, "timestamp": time.time()}
  orchestrator.observe(hw_state, profile)
  ```

- `think() -> str`: Decide backend ("ane" or "cpu") based on last observation.
  ```python
  backend = orchestrator.think()  # Returns "ane" or "cpu"
  ```

- `predict_performance(profile: dict, backend: str) -> float`: Estimate training time (seconds).
  ```python
  ane_time = orchestrator.predict_performance(profile, "ane")
  cpu_time = orchestrator.predict_performance(profile, "cpu")
  ```

- `select_backend(profile: dict, hardware_state: dict) -> tuple`: Returns `(backend: str, confidence: float)` using deterministic rules.
  ```python
  backend, confidence = orchestrator.select_backend(profile, hw_state)
  # ("cpu", 0.8) or ("ane", 0.85)
  ```

- `act(action: str, environment) -> dict`: Execute training on selected backend. Returns dict with keys: `status`, `backend`, `execution_time`, `message`.
  ```python
  result = orchestrator.act("cpu", training_environment)
  # {'status': 'success', 'backend': 'cpu', 'execution_time': 2.34, 'message': '...'}
  ```

### `TrainingEnvironment`
Manages training execution on ANE or CPU backends.

**Constructor:**
- `__init__(X_train: np.ndarray, y_train: np.ndarray, model_config: dict)`: Initialize with training data and model configuration.

**Methods:**
- `execute_training(backend: str, workload_profile: dict) -> dict`: Execute training on specified backend. Returns dict with keys: `status`, `backend`, `execution_time`, `message`.

## Research Background

This project is inspired by the growing need for efficient resource utilization in edge and local ML inference/training, particularly where proprietary hardware accelerators (like ANE) are involved. The underlying concepts draw from dynamic resource scheduling algorithms used in cloud computing, adapted for the constraints of local, heterogeneous silicon.

## Testing

Tests are maintained in the `tests/` directory and cover the core logic of workload profiling and backend selection, ensuring the agent correctly routes workloads for small CNNs, large dense models, and sequential models.

## Contributing

We welcome contributions! Please see our contribution guidelines for details on submitting pull requests, reporting bugs, or suggesting features.

## Citation

This work is independent, but related research in heterogeneous computing can be found in papers concerning dynamic workload migration and accelerator offloading.

## License
The project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.