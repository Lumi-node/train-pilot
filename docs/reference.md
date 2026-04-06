# TrainPilot API Reference

TrainPilot is a system designed to intelligently orchestrate machine learning training jobs by dynamically selecting the optimal hardware backend (ANE or CPU) based on workload characteristics and real-time system metrics.

---

## 🧠 `hardware_orchestrator.py`

This module contains the core logic for the `ReasoningAgent`, which analyzes training requirements and system state to decide the best execution path.

### `ReasoningAgent` Class

Manages the decision-making process for hardware selection.

**Signature:**
```python
class ReasoningAgent:
    def __init__(self, config: dict): ...
    def profile_workload(self, model_architecture: dict, dataset_size: int) -> dict: ...
    def predict_performance(self, workload_profile: dict, current_utilization: dict) -> float: ...
    def select_backend(self, workload_profile: dict, current_utilization: dict) -> str: ...
```

**Description:**
The central intelligence of TrainPilot. It ingests workload definitions and system telemetry to predict training efficiency and select either the specialized ANE hardware or the general-purpose CPU.

**Methods:**

*   **`__init__(self, config: dict)`**
    *   **Description:** Initializes the agent with configuration parameters (e.g., performance thresholds, hardware limits).
    *   **Example Usage:**
        ```python
        agent = ReasoningAgent({"ane_threshold": 0.85, "cpu_fallback_time": 120.0})
        ```

*   **`profile_workload(self, model_architecture: dict, dataset_size: int) -> dict`**
    *   **Description:** Analyzes the provided model structure and dataset size to extract complexity metrics (e.g., FLOPs, parameter count, data throughput requirements).
    *   **Returns:** A dictionary containing workload metrics.
    *   **Example Usage:**
        ```python
        cnn_profile = agent.profile_workload(
            model_architecture={"layers": 5, "params": 100000},
            dataset_size=5000
        )
        # cnn_profile might look like: {'flops': 5e8, 'complexity_score': 0.6}
        ```

*   **`predict_performance(self, workload_profile: dict, current_utilization: dict) -> float`**
    *   **Description:** Estimates the total training time (in seconds) based on the workload profile and the current utilization of available hardware (ANE/CPU).
    *   **Parameters:**
        *   `workload_profile`: Output from `profile_workload()`.
        *   `current_utilization`: Dictionary containing current ANE/CPU utilization (e.g., `{'ane_util': 0.9, 'cpu_util': 0.3}`).
    *   **Returns:** Estimated training time in seconds.
    *   **Example Usage:**
        ```python
        time_estimate = agent.predict_performance(cnn_profile, {'ane_util': 0.8, 'cpu_util': 0.2})
        print(f"Estimated time: {time_estimate}s")
        ```

*   **`select_backend(self, workload_profile: dict, current_utilization: dict) -> str`**
    *   **Description:** The primary decision function. Compares predicted performance against configured thresholds to decide between `'ANE'` or `'CPU'`.
    *   **Returns:** A string indicating the chosen backend (`"ANE"` or `"CPU"`).
    *   **Example Usage:**
        ```python
        backend = agent.select_backend(cnn_profile, {'ane_util': 0.95, 'cpu_util': 0.1})
        print(f"Selected backend: {backend}")
        ```

---

## 🧪 `tests/__init__.py`

This module serves as the entry point for the testing suite, ensuring that the core components of TrainPilot meet the specified success criteria.

### Test Suite Functions

**Signature:**
```python
def run_all_tests(): ...
```

**Description:**
Executes a comprehensive battery of tests to validate the `ReasoningAgent`'s ability to correctly route workloads under various conditions.

**Example Usage:**
```python
from tests import run_all_tests

# Execute the entire test suite
run_all_tests()
```

---

## 🚀 `train_orchestrated.py`

This module acts as the high-level controller, utilizing the `ReasoningAgent` to manage the lifecycle of a training job, including invoking the appropriate backend trainer.

### `TrainingOrchestrator` Class

Manages the end-to-end training workflow.

**Signature:**
```python
class TrainingOrchestrator:
    def __init__(self, agent: ReasoningAgent): ...
    def execute_training_job(self, job_spec: dict) -> dict: ...
```

**Description:**
Takes a high-level job specification, passes it to the `ReasoningAgent` for backend selection, and then invokes the corresponding trainer (`ane_trainer` or CPU fallback).

**Methods:**

*   **`__init__(self, agent: ReasoningAgent)`**
    *   **Description:** Initializes the orchestrator, requiring an already configured `ReasoningAgent`.
    *   **Example Usage:**
        ```python
        from hardware_orchestrator import ReasoningAgent
        agent = ReasoningAgent({...})
        orchestrator = TrainingOrchestrator(agent)
        ```

*   **`execute_training_job(self, job_spec: dict) -> dict`**
    *   **Description:** The main execution pipeline. It profiles the job, selects the hardware, and runs the training.
    *   **Parameters:**
        *   `job_spec`: A dictionary containing `model_architecture`, `dataset_size`, and initial system metrics.
    *   **Returns:** A summary dictionary of the training run (e.g., `{'status': 'SUCCESS', 'backend_used': 'ANE', 'duration': 350.5}`).
    *   **Example Usage:**
        ```python
        job = {
            "model_architecture": {"layers": 10, "params": 500000},
            "dataset_size": 100000,
            "initial_metrics": {'ane_util': 0.5, 'cpu_util': 0.1}
        }
        results = orchestrator.execute_training_job(job)
        print(results)
        ```