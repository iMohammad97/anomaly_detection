import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualize_synchronized_time_series(directory, selected_i, file_prefix_template="{i}_test", plot_title_template="Synchronized Time Series for i={i}"):
    """
    Visualizes synchronized time series data stored in a specified directory.

    Args:
        directory (str): Path to the directory containing the time series files.
        selected_i (list of int): List of `i` values to visualize.
        file_prefix_template (str, optional): Template for the file prefix based on `i`. Defaults to "{i}_test".
        plot_title_template (str, optional): Template for the plot title based on `i`. Defaults to "Synchronized Time Series for i={i}".

    Usage:
        - Place the `.npy` files in the specified `directory`.
        - The file naming convention should match the provided `file_prefix_template`.
        - Call the function with the directory path and a list of `i` values to visualize.

    Example:
        visualize_synchronized_time_series(
            directory="/path/to/directory",
            selected_i=[19, 23, 45]
        )
    """
    def plot_sync_series(time_series, title="Synchronized Time Series Plots"):
        """Helper function to visualize multiple time series as separate subplots."""
        # Create subplots
        fig = make_subplots(rows=len(time_series), cols=1, shared_xaxes=True)

        # Adding time series to subplots
        i = 1
        for key, value in time_series.items():
            fig.add_trace(go.Scatter(x=value.index, y=value.values.reshape(-1), mode='lines', name=f'{key}'), row=i, col=1)
            i += 1

        # Update layout for better visibility and interaction
        fig.update_layout(height=300 * len(time_series), width=1500, title_text=title)

        # Plot the figure
        fig.show()

    # Iterate over the selected `i` values
    for i in selected_i:
        print(f"Visualizing for i={i}...")

        # Filter matching files for the current `i`
        file_prefix = file_prefix_template.format(i=i)
        files = os.listdir(directory)
        matching_files = [file for file in files if file.startswith(file_prefix)]

        # Load and organize the time series data
        time_series = {}
        for file_name in matching_files:
            file_path = os.path.join(directory, file_name)
            time_series[file_name] = pd.Series(np.load(file_path).reshape(-1))
        
        # Call the helper function to plot the synchronized time series
        plot_sync_series(time_series, title=plot_title_template.format(i=i))
