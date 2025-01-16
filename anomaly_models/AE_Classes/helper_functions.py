# helper_functions.py
import plotly.graph_objects as go

def plot_results(test_data, anomaly_preds, anomaly_errors, predictions, labels, size=800):
    # Flattening arrays to ensure they are 1D
    test_data = test_data.ravel()  # Convert to 1D array
    anomaly_preds = anomaly_preds  # Already 1D
    anomaly_errors = anomaly_errors  # Already 1D
    predictions = predictions  # Already 1D
    labels = labels.ravel()  # Convert to 1D array

    # Check if all inputs have the same length
    if not (len(test_data) == len(labels) == len(anomaly_preds) == len(anomaly_errors) == len(predictions)):
        raise ValueError("All input arrays must have the same length.")

    # Create a figure
    fig = go.Figure()

    # Add traces for test data, predictions, and anomaly errors
    fig.add_trace(go.Scatter(x=list(range(len(test_data))),
                             y=test_data,
                             mode='lines',
                             name='Test Data'))

    fig.add_trace(go.Scatter(x=list(range(len(predictions))),
                             y=predictions,
                             mode='lines',
                             name='Predictions'))

    fig.add_trace(go.Scatter(x=list(range(len(anomaly_errors))),
                             y=anomaly_errors,
                             mode='lines',
                             name='Anomaly Errors'))

    # Set the layout
    fig.update_layout(title='Test Data, Predictions, and Anomalies',
                      xaxis_title='Time Steps',
                      yaxis_title='Value',
                      template='plotly')
    fig.show()
