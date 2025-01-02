import numpy as np
from matplotlib import pyplot as plt

# Example: plotting reconstruction error over time with threshold
def plot_reconstruction_error_over_time(timestep_errors, threshold, Y_test=None, anomaly_preds=None):
    plt.figure(figsize=(15,5))
    plt.plot(timestep_errors, label='Reconstruction Error')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    if anomaly_preds is not None:
        # Highlight anomalies predicted
        anomaly_indices = np.where(anomaly_preds == 1)[0]
        plt.scatter(anomaly_indices, timestep_errors[anomaly_indices], color='red', label='Anomalies Predicted')
    if Y_test is not None: 
        # Highlight real anomalies 
        real_anomaly_indices = np.where(Y_test == 1)[0] 
        plt.scatter(real_anomaly_indices, timestep_errors[real_anomaly_indices], color='blue', label='Real Anomalies', marker='x')
    plt.title('Reconstruction Error over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
