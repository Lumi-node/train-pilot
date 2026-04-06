# TrainPilot Architecture

TrainPilot is a sophisticated training orchestration system designed to intelligently manage and optimize the execution of neural network training jobs. Its core purpose is to dynamically select the most efficient hardware backend—either specialized Accelerated Neural Engine (ANE) hardware or standard CPU—based on a comprehensive analysis of the workload characteristics and current system resource availability. This intelligent routing minimizes training time and maximizes hardware utilization.

## System Diagram

The following diagram illustrates the high-level relationships between the core components of TrainPilot.

```mermaid
graph TD
    A[User Input/Job Request] --> B(train_orchestrated.py);
    B --> C{ReasoningAgent};
    C --> D[hardware_orchestrator.py];
    D --> E[ANE Trainer Interface];
    D --> F[CPU Trainer Interface];
    C --> G[Performance Logs];
    C --> H[System Metrics (ANE/CPU Utilization)];
    D --> I[Model Architecture/Dataset Info];
    E --> J(Training Execution);
    F --> J;
    J --> G;
```

## Module Descriptions

### `train_orchestrated.py`
This serves as the primary entry point and high-level workflow manager for TrainPilot. It receives the initial training job request (specifying model architecture, dataset size, etc.). It initializes the `ReasoningAgent` within the `hardware_orchestrator` and drives the overall training lifecycle: profiling the workload, receiving the backend decision, and initiating the appropriate training execution path.

### `hardware_orchestrator.py`
This is the brain of the system, housing the `ReasoningAgent`. The agent is responsible for the decision-making logic. It consumes various inputs—model complexity, dataset size, real-time utilization data, and historical performance logs—to make an informed choice. Its key responsibilities include:
*   **`profile_workload()`**: Analyzes the input model architecture and dataset to extract quantifiable complexity metrics (e.g., FLOPs, parameter count).
*   **`predict_performance()`**: Uses historical data and current metrics to estimate the expected training time on both ANE and CPU.
*   **`select_backend()`**: Compares the predicted performance and current resource availability to decide whether to invoke the ANE trainer or fall back to CPU training.

### `tests/__init__.py`
This module contains the testing infrastructure for TrainPilot. It is crucial for validating the correctness of the `ReasoningAgent`'s decision-making process. Tests are specifically designed to verify that the agent correctly routes workloads based on defined success criteria (e.g., correctly routing small CNNs to ANE, large RNNs to CPU, etc.).

## Data Flow Explanation

The data flow in TrainPilot follows a sequential decision-making pattern:

1.  **Initialization:** `train_orchestrated.py` receives the raw training job parameters (Model Architecture, Dataset Size).
2.  **Profiling:** These parameters are passed to the `ReasoningAgent` within `hardware_orchestrator.py`, which executes `profile_workload()` to generate complexity metrics.
3.  **Observation & Prediction:** The Agent gathers real-time data (ANE/CPU utilization) and historical data (Performance Logs). It then uses these observations to execute `predict_performance()`, generating time estimates for both potential backends.
4.  **Decision Making:** Based on the predictions and current constraints, the Agent calls `select_backend()`, outputting a definitive choice (ANE or CPU).
5.  **Execution:** `train_orchestrated.py` receives this decision and delegates the training job to the corresponding backend interface (ANE Trainer or CPU Trainer).
6.  **Feedback Loop:** As training progresses, performance metrics are captured and fed back into the **Performance Logs**, enriching the data available for future iterations of the `ReasoningAgent`.