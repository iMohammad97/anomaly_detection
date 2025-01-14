import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class TimeSeriesAnomalyDetectorKNN:
    def __init__(self, k, train, test, Y_labels=None, dim=1, metric='cosine',step_period=10,window_length=100):
        self.window_length = window_length
        self.step_period = step_period
        self.training_data = train
        self.test_data = test
        self.labels = None
        self.Y_labels = Y_labels  # Ground truth labels for actual anomalies
        self.k = k
        self.dim = dim
        self.sigma = np.diag(np.ones(self.dim))
        self.scores = None
        self.train_data_matrix = None
        self.test_data_matrix = None
        self.y_anomaly = None
        self.metric = metric

    def transform_to_matrix(self, time_series):
        if len(time_series) < self.window_length:
            raise ValueError(f"Time series length ({len(time_series)}) must be greater than or equal to the window length ({self.window_length}).")
        num_rows = len(time_series) - self.window_length + 1

        matrix = np.zeros((num_rows, self.window_length))
        for i in range(num_rows):
            matrix[i, :] = time_series[i:i + self.window_length]
        return matrix

    def train_func(self, X_train, y_train):
        flattened_train = X_train.flatten()
        if len(flattened_train) < self.window_length:
            raise ValueError(f"Training data length ({len(flattened_train)}) must be greater than or equal to the window length ({self.window_length}).")
        self.training_data = self.transform_to_matrix(flattened_train)
        return None

    def test_func(self, X_test, batch_size=100):
        flattened_test = X_test.flatten()
        self.test_data = self.transform_to_matrix(flattened_test)
    
        n_test = self.test_data.shape[0]
        n_batches = (n_test + batch_size - 1) // batch_size  # Compute number of batches
        all_scores = []
    
        for batch_idx in range(n_batches):
            # Select a batch of test windows
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_test)
            batch = self.test_data[start_idx:end_idx]
    
            # Compute distances for the batch
            if self.metric == 'cosine':
                distance_matrix = self.calculate_cosine_distances(batch, self.training_data)
            elif self.metric == 'mahalanobis':
                distance_matrix = self.calculate_mahalanobis_distances(batch, self.training_data, self.sigma)
    
            # Sum the k smallest distances for each test sample in the batch
            batch_scores = np.sum(np.sort(distance_matrix, axis=1)[:, :self.k], axis=1)
            all_scores.append(batch_scores)
    
        # Concatenate scores from all batches
        return np.concatenate(all_scores)

    def calc_anomaly(self, mode='expanding'):
        self.train_func(self.training_data,self.training_data)
        self.y_anomaly = np.zeros(self.window_length-1)
        self.y_anomaly = np.append(self.y_anomaly, self.test_func(self.test_data))
        return pd.Series(self.y_anomaly)


    def calc_anomaly_window_based(self):
        """
        Calculate anomaly scores using a window-based aggregation approach.
        """
        self.train_func(self.training_data, self.training_data)

        # Step 1: Compute anomaly scores for all windows
        anomaly_scores = self.test_func(self.test_data)

        # Step 2: Initialize arrays to store per-timestep errors and counts
        length = len(self.test_data)
        M = len(anomaly_scores)  # Number of windows
        timestep_errors = np.zeros(length)
        counts = np.zeros(length)

        # Step 3: Aggregate window-based scores into per-timestep scores
        for i in range(M):
            start = i
            end = i + self.window_length - 1
            timestep_errors[start:end + 1] += anomaly_scores[i]
            counts[start:end + 1] += 1

        # Step 4: Average scores for overlapping windows
        counts[counts == 0] = 1  # Avoid division by zero
        timestep_errors /= counts  # Average overlapping window scores

        # Step 5: Return per-timestep anomaly scores as a pandas Series
        return pd.Series(timestep_errors)
    
    

    def calculate_cosine_distances(self, test_data, train_data):
        # Normalize training data
        train_norms = np.linalg.norm(train_data, axis=1, keepdims=True)
        train_normalized = train_data / train_norms  # Shape: (n_train, d)
    
        # Normalize test data
        test_norms = np.linalg.norm(test_data, axis=1, keepdims=True)
        test_normalized = test_data / test_norms  # Shape: (n_test, d)
    
        # Batch compute cosine similarity (dot product of normalized vectors)
        similarity_matrix = np.dot(test_normalized, train_normalized.T)  # Shape: (n_test, n_train)
    
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix  # Shape: (n_test, n_train)
    
        return distance_matrix

    def calculate_mahalanobis_distances(self, test_data, train_data, inv_covmat):
        # Compute differences between test and train batches
        diff = test_data[:, np.newaxis, :] - train_data[np.newaxis, :, :]  # Shape: (n_test, n_train, d)
    
        # Batch apply Mahalanobis formula
        left_term = np.einsum('ijk,kl->ijl', diff, inv_covmat)  # Shape: (n_test, n_train, d)
        mahalanobis_distance_squared = np.einsum('ijk,ijk->ij', left_term, diff)  # Shape: (n_test, n_train)
    
        return np.sqrt(mahalanobis_distance_squared)  # Shape: (n_test, n_train)

    def calculate_anomaly_threshold(self, quantile=0.95):
        logging.info("Calculating anomaly threshold from training data.")
        
        self.train_func(self.training_data, self.training_data)
        
        # Compute anomaly scores for training data
        anomaly_scores = self.test_func(self.training_data)
        
        # Compute the threshold based on the specified quantile
        threshold = np.quantile(anomaly_scores, quantile)
        
        logging.info(f"Calculated threshold: {threshold}")
        return threshold


    def save_state(self, file_path: str):
        state = {
            'training_data': self.training_data.tolist() if self.training_data is not None else None,
            'test_data': self.test_data.tolist() if self.test_data is not None else None,
            'labels': self.labels.tolist() if self.labels is not None else None,
            'window_length': self.window_length,
            'step_period': self.step_period,
            'k': self.k,
            'dim': self.dim,
            'metric': self.metric,
            'anomaly_scores': self.y_anomaly.tolist() if self.y_anomaly is not None else None
        }

        with open(file_path, 'w') as file:
            json.dump(state, file)
        print(f"State saved to {file_path}")

    def load_state(self, file_path: str):
        with open(file_path, 'r') as file:
            state = json.load(file)

        # Restore attributes
        self.training_data = np.array(state['training_data']) if state['training_data'] is not None else None
        self.test_data = np.array(state['test_data']) if state['test_data'] is not None else None
        self.labels = np.array(state['labels']) if state['labels'] is not None else None
        self.window_length = state['window_length']
        self.step_period = state['step_period']
        self.k = state['k']
        self.dim = state['dim']
        self.metric = state['metric']
        self.y_anomaly = np.array(state['anomaly_scores']) if state['anomaly_scores'] is not None else None

        print(f"State loaded from {file_path}")
    
    def plot_results(self, size=800, threshold=None):
        """
        Plot the test data, anomaly scores, and highlight anomalies using both internal labels and ground truth labels.
    
        :param size: Width of the plot.
        :param threshold: Optional threshold to highlight predicted anomalies.
        """
        # Flatten and prepare data
        test_data = self.test_data.ravel() if self.test_data is not None else []
        anomaly_scores = self.y_anomaly if self.y_anomaly is not None else []
        Y_labels = self.Y_labels.ravel() if self.Y_labels is not None else []
        labels = self.labels.ravel() if self.labels is not None else []
    
        # Check if lengths match
        if len(test_data) != len(anomaly_scores):
            raise ValueError("Test data and anomaly scores must have the same length.")
        if self.Y_labels is not None and len(test_data) != len(Y_labels):
            raise ValueError("Test data and ground truth labels must have the same length.")
        if self.labels is not None and len(test_data) != len(labels):
            raise ValueError("Test data and internal labels must have the same length.")
    
        # Determine plot width
        plot_width = max(size, len(test_data) * 2)
    
        # Create a figure
        fig = go.Figure()
    
        # Add test data and anomaly scores
        fig.add_trace(go.Scatter(x=list(range(len(test_data))),
                                 y=test_data,
                                 mode='lines',
                                 name='Test Data',
                                 line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=list(range(len(anomaly_scores))),
                                 y=anomaly_scores,
                                 mode='lines',
                                 name='Anomaly Scores',
                                 line=dict(color='red')))
    
        # Highlight true anomalies (Y_labels)
        if self.Y_labels is not None:
            true_anomaly_indices = [i for i, label in enumerate(Y_labels) if label == 1]
            if true_anomaly_indices:
                fig.add_trace(go.Scatter(x=true_anomaly_indices,
                                         y=[test_data[i] for i in true_anomaly_indices],
                                         mode='markers',
                                         name='True Anomalies (Y_labels)',
                                         marker=dict(color='orange', size=10)))
    
        # Highlight predicted anomalies (threshold-based)
        if threshold is not None:
            predicted_anomalies = [i for i, score in enumerate(anomaly_scores) if score > threshold]
            if predicted_anomalies:
                fig.add_trace(go.Scatter(x=predicted_anomalies,
                                         y=[test_data[i] for i in predicted_anomalies],
                                         mode='markers',
                                         name='Predicted Anomalies',
                                         marker=dict(color='green', size=10)))
    
        # Highlight internal labels
        if self.labels is not None:
            label_indices = [i for i, label in enumerate(labels) if label == 1]
            if label_indices:
                fig.add_trace(go.Scatter(x=label_indices,
                                         y=[test_data[i] for i in label_indices],
                                         mode='markers',
                                         name='Internal Labels',
                                         marker=dict(color='purple', size=10)))
    
        # Set layout
        fig.update_layout(title='Test Data, Anomaly Scores, and Anomalies',
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
                          template='plotly',
                          width=plot_width)
    
        # Show the figure
        fig.show()


#use knn

    # knn_model = TimeSeriesAnomalyDetectorKNN(window_length=30, k=10, train=train_data, test=test_data, metric='cosine')
    # try:
    #     knn_model.calc_anomaly()
    # except ValueError as e:
    #     logging.error(e)


