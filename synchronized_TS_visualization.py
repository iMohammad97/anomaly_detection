import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_sync_series(time_series, title="Synchronized Time Series Plots"):

    fig = make_subplots(rows=len(time_series), cols=1, shared_xaxes=True)

    labels = time_series.pop('labels', None)

    i = 1
    for key, value in time_series.items():
      fig.add_trace(go.Scatter(
          x=value.index, 
          y=value.values.reshape(-1), 
          mode='lines', 
          name=f'{key}'
      ), row=i, col=1)

      if labels.any():  # Ensure there are labels to plot
          # Filter for points where labels are 1
          scatter_points = labels[labels == 1].index
          scatter_values = value[scatter_points].values.reshape(-1)
          fig.add_trace(go.Scatter(
              x=scatter_points, 
              y=scatter_values, 
              mode='markers', 
              name=f'{key} - Labels'
          ), row=i, col=1)
      i += 1
    # Update layout for better visibility and interaction
    fig.update_layout(
        height=300 * len(time_series), 
        width=1500, 
        title_text=title
    )

    fig.show()
