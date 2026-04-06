# 🚂 TrainPilot Quick Start Guide

TrainPilot is a system designed to intelligently manage and orchestrate neural network training workloads by dynamically selecting the optimal hardware backend (e.g., ANE or CPU) based on workload characteristics and current system utilization.

This guide focuses on setting up and using the core logic within `hardware_orchestrator.py`.

## Prerequisites

Ensure you have the `train_pilot` package installed:

```bash
pip install train_pilot
```

## Core Concept: The ReasoningAgent

The heart of TrainPilot is the `ReasoningAgent` located in `hardware_orchestrator.py`. This agent acts as the decision-maker. It ingests system telemetry (utilization, dataset size, model specs) and uses internal logic to predict the best training path.

**Key Responsibilities:**
1. **Profiling:** Analyzing the input workload (model architecture, dataset) to derive complexity metrics.
2. **Prediction:** Estimating the required training time and resource needs.
3. **Selection:** Deciding whether to invoke the high-performance `ane_trainer` or fall back to the standard CPU training path.

## Getting Started: Usage Examples

The following examples demonstrate how to instantiate and use the `ReasoningAgent` to make training decisions for different scenarios.

### Example 1: Small, Simple Workload (Likely CPU Friendly)

For a small Convolutional Neural Network (CNN) on a modest dataset, the agent might determine that the overhead of setting up the ANE is not worth the marginal gain, opting for CPU training.

```python
from train_pilot.hardware_orchestrator import ReasoningAgent

# 1. Initialize the Agent
agent = ReasoningAgent()

# 2. Define the workload profile
workload_small_cnn = {
    "model_architecture": "SmallCNN_V1",
    "dataset_size": 10000,  # Small dataset
    "current_utilization": {"ANE": 0.2, "CPU": 0.4}, # Low utilization
    "past_performance_logs": []
}

# 3. Profile and Select Backend
print("--- Running Small CNN Workload Test ---")
profile = agent.profile_workload(workload_small_cnn)
print(f"Profiled Metrics: {profile}")

backend = agent.select_backend(workload_small_cnn, profile)
print(f"Decision: Recommended backend is {backend}")

# Expected Output: 'CPU' or 'ANE' depending on internal thresholds
```

### Example 2: Large, Complex Workload (ANE Candidate)

For a large Transformer model with a massive dataset, the agent should strongly favor the ANE backend to meet performance goals.

```python
from train_pilot.hardware_orchestrator import ReasoningAgent

agent = ReasoningAgent()

# Define a demanding workload
workload_large_transformer = {
    "model_architecture": "LargeTransformer_BERT_XL",
    "dataset_size": 500000, # Large dataset
    "current_utilization": {"ANE": 0.8, "CPU": 0.9}, # High utilization (but ANE is still better)
    "past_performance_logs": [{"time_taken": 3600, "backend": "CPU"}]
}

# Profile and Select Backend
print("\n--- Running Large Transformer Workload Test ---")
profile = agent.profile_workload(workload_large_transformer)
print(f"Profiled Metrics: {profile}")

backend = agent.select_backend(workload_large_transformer, profile)
print(f"Decision: Recommended backend is {backend}")

# If the agent predicts significant time savings, it should select 'ANE'
```

### Example 3: Dynamic Fallback Scenario

This example simulates a scenario where the agent initially favors ANE, but due to extremely high current ANE utilization, it intelligently falls back to CPU training to prevent queuing delays.

```python
from train_pilot.hardware_orchestrator import ReasoningAgent

agent = ReasoningAgent()

# Define a workload that is inherently ANE-suitable, but system is saturated
workload_high_demand = {
    "model_architecture": "DeepResNet_V3",
    "dataset_size": 150000,
    "current_utilization": {"ANE": 0.98, "CPU": 0.3}, # ANE is nearly maxed out
    "past_performance_logs": []
}

# Profile and Select Backend
print("\n--- Running High Demand Fallback Test ---")
profile = agent.profile_workload(workload_high_demand)
print(f"Profiled Metrics: {profile}")

backend = agent.select_backend(workload_high_demand, profile)
print(f"Decision: Recommended backend is {backend}")

# Success Criteria Check: The agent should detect the high ANE utilization 
# and choose 'CPU' to maintain throughput, even if ANE is theoretically faster.
```

## Summary of Key Functions

| Function | Module | Purpose | Input | Output |
| :--- | :--- | :--- | :--- | :--- |
| `profile_workload()` | `hardware_orchestrator.py` | Extracts complexity metrics (e.g., FLOPs, parameter count) from the model architecture. | Workload Dictionary | Profile Dictionary |
| `predict_performance()` | `hardware_orchestrator.py` | Estimates training duration based on workload profile and current hardware load. | Workload & Profile | Estimated Time (seconds) |
| `select_backend()` | `hardware_orchestrator.py` | The final decision point. Chooses 'ANE' or 'CPU' based on predictions and utilization thresholds. | Workload & Profile | String ('ANE' or 'CPU') |