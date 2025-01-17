# Evaluation Module for Time-Series Autoencoders

This module provides three evaluation functions for time-series anomaly detection models. These functions can be used with any autoencoder implementation, including simple autoencoders, variational autoencoders, and stationary autoencoders.

---

## Functions

### 1. **`evaluate`**
- **Purpose**: Computes per-timestep anomaly scores by aggregating reconstruction errors from overlapping windows.
- **Key Features**:
  - Maps window-level errors to timesteps.
  - Reconstructs the full test data by averaging overlapping predictions.
  - Detects anomalies based on a predefined threshold.
- **Use Case**: General anomaly detection where timestep-level reconstruction and anomaly detection are required.

---

### 2. **`evaluate_window_level`**
- **Purpose**: Computes anomaly scores for each window without mapping to timesteps.
- **Key Features**:
  - Evaluates reconstruction errors for each window independently.
  - Returns window-level anomaly scores as a pandas Series.
- **Use Case**: Coarse-grained anomaly detection at the window level.

---

### 3. **`evaluate_timestep_based`**
- **Purpose**: Directly computes per-timestep reconstruction errors by aggregating errors from all overlapping windows covering each timestep.
- **Key Features**:
  - Provides fine-grained, timestep-level anomaly detection.
  - Reconstructs the full test data from overlapping predictions.
- **Use Case**: When precise, timestep-level anomaly scores are needed.

---

## Inputs
All functions require the following inputs:
- **`model`**: A trained autoencoder model.
- **`test_data_window`**: Overlapping windows of the test data.
- **`test_data`**: Original test data (1D or 2D array, used in `evaluate` and `evaluate_timestep_based`).
- **`timesteps`**: Number of timesteps per window.
- **`threshold`**: Precomputed anomaly threshold from training data.
- **`batch_size`**: Batch size for model predictions (default: `32`).

---

## Outputs
1. **`evaluate`**:
   - **`anomaly_preds`**: Binary flags for timestep anomalies.
   - **`anomaly_errors`**: Per-timestep reconstruction errors.
   - **`predictions`**: Reconstructed test data.

2. **`evaluate_window_level`**:
   - **`window_anomaly_scores`**: A pandas Series with window-level anomaly scores.

3. **`evaluate_timestep_based`**:
   - **`timestep_preds`**: Binary flags for timestep anomalies.
   - **`timestep_errors`**: Per-timestep reconstruction errors.
   - **`reconstructed_data`**: Reconstructed test data at the timestep level.
