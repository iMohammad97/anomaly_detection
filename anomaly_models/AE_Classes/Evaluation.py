import numpy as np
import pandas as pd

def evaluate(model, test_data_window, test_data, timesteps, threshold, batch_size=32):

    length = test_data.shape[0]
    
    # Generate predictions for the test data windows
    predictions_windows = model.predict(test_data_window, batch_size=batch_size)[0]
    mse = np.mean(np.square(test_data_window - predictions_windows), axis=(1, 2))
    
    # Initialize arrays for timestep-level errors and counts
    M = mse.shape[0]
    timestep_errors = np.zeros(length)
    counts = np.zeros(length)
    predictions = np.zeros(length)
    
    # Aggregate errors and predictions over overlapping windows
    for i in range(M):
        start = i
        end = i + timesteps - 1
        timestep_errors[start:end + 1] += mse[i]
        counts[start:end + 1] += 1

    counts[counts == 0] = 1  # Avoid division by zero
    timestep_errors /= counts  # Average overlapping window errors

    for i in range(M):
        for j in range(timesteps):
            timestep_index = i + j
            if timestep_index < length:
                predictions[timestep_index] += np.mean(predictions_windows[i, j, :])

    predictions /= counts
    predictions = np.nan_to_num(predictions)

    # Generate anomaly predictions
    anomaly_preds = (timestep_errors > threshold).astype(int)

    return anomaly_preds, timestep_errors, predictions


def evaluate_window_level(model, test_data_window, threshold, batch_size=32):

    # Generate predictions for the test data windows
    predictions_windows = model.predict(test_data_window, batch_size=batch_size)[0]
    
    # Compute mean squared error for each window
    mse = np.mean(np.square(test_data_window - predictions_windows), axis=(1, 2))
    
    # Return window-level anomaly scores as a pandas Series
    return pd.Series(mse, name="Window-Level Anomaly Scores")


def evaluate_timestep_based(model, test_data_window, test_data, timesteps, threshold, batch_size=32):
  
    length = test_data.shape[0]
    
    # Generate predictions for the test data windows
    predictions_windows = model.predict(test_data_window, batch_size=batch_size)[0]
    
    # Initialize arrays for timestep-level errors, counts, and predictions
    timestep_errors = np.zeros(length)
    counts = np.zeros(length)
    reconstructed_data = np.zeros(length)

    # Compute per-timestep errors by averaging over all covering windows
    for i, window in enumerate(predictions_windows):
        for j in range(timesteps):
            timestep_index = i + j
            if timestep_index < length:
                # Compute the reconstruction error for this timestep
                timestep_errors[timestep_index] += np.mean(np.abs(test_data_window[i, j, :] - window[j, :]))
                counts[timestep_index] += 1
                # Accumulate reconstructed data
                reconstructed_data[timestep_index] += np.mean(window[j, :])

    counts[counts == 0] = 1  # Avoid division by zero
    timestep_errors /= counts
    reconstructed_data /= counts

    # Generate anomaly predictions
    anomaly_preds_timestep_based = (timestep_errors > threshold).astype(int)

    return anomaly_preds_timestep_based, timestep_errors, reconstructed_data
