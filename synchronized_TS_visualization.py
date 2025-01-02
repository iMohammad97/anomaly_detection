import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_sync_series_with_anomalies(time_series, labels_file, title="Synchronized Time Series with Anomalies"):
    """
    Helper function to visualize multiple time series as separate subplots
    and scatter anomalies based on labels file.
    
    :param time_series: Dictionary of time series dataframes or series.
    :param labels_file: Path to the numpy file containing anomaly labels.
    :param title: Title of the plot.
    """
    # Load anomaly labels
    anomaly_labels = np.load(labels_file)
    
    # Ensure labels match the length of time series indices
    if len(anomaly_labels) != len(next(iter(time_series.values())).index):
        raise ValueError("Labels file does not match the length of time series data.")

    # Create subplots
    fig = make_subplots(rows=len(time_series), cols=1, shared_xaxes=True)

    # Adding time series and anomalies to subplots
    i = 1
    for key, value in time_series.items():
        # Add the time series line
        fig.add_trace(go.Scatter(
            x=value.index,
            y=value.values.reshape(-1),
            mode='lines',
            name=f'{key}'), row=i, col=1)
        
        # Add anomaly points
        anomaly_indices = value.index[anomaly_labels.astype(bool)]  # Indices where anomalies are True
        anomaly_values = value.values[anomaly_labels.astype(bool)]
        fig.add_trace(go.Scatter(
            x=anomaly_indices,
            y=anomaly_values.reshape(-1),
            mode='markers',
            marker=dict(color='red', size=8),
            name=f'Anomalies - {key}'
        ), row=i, col=1)
        i += 1

    # Update layout for better visibility and interaction
    fig.update_layout(height=300 * len(time_series), width=1500, title_text=title)

    # Plot the figure
    fig.show()
