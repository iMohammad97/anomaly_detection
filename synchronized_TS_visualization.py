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
