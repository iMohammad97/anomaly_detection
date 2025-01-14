import numpy as np
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
from matplotlib import pyplot as plt
import plotly.graph_objects as go


# Set up logging
logging.basicConfig(level=logging.INFO)

class TimeSeriesAnomalyDetectorKNN:
    def __init__(self, k, train, test, Y_labels=None, dim=1, metric='cosine',step_period=10,window_length=100):
        self.window_length = window_length
        self.step_period = step_period
        self.training_data = train
        self.test_data = test
        self.labels = Y_labels
        self.k = k
        self.dim = dim
        self.sigma = np.diag(np.ones(self.dim))
        self.scores = None
        self.train_data_matrix = None
        self.test_data_matrix = None
        self.anomaly_errors = np.zeros(len(self.test_data))
        self.metric = metric
        self.anomaly_preds = np.zeros(len(self.test_data))
        self.threshold = 0
        

    def transform_to_matrix(self, time_series):
        if len(time_series) < self.window_length:
            raise ValueError(f"Time series length ({len(time_series)}) must be greater than or equal to the window length ({self.window_length}).")
        num_rows = len(time_series) - self.window_length + 1

        matrix = np.zeros((num_rows, self.window_length))
        for i in range(num_rows):
            matrix[i, :] = time_series[i:i + self.window_length]
        return matrix

    def train_func(self):
        flattened_train = self.training_data.flatten()
        if len(flattened_train) < self.window_length:
            raise ValueError(f"Training data length ({len(flattened_train)}) must be greater than or equal to the window length ({self.window_length}).")
        self.train_data_matrix = self.transform_to_matrix(flattened_train)
        anomaly_scores = self.test_func(True)
        
        # Compute the threshold based on the specified quantile
        self.threshold = np.quantile(anomaly_scores, quantile)
        
        logging.info(f"Calculated threshold: {threshold}")
        return None


    def test_func(self, compute_threshold=False, batch_size=100, threads=10):
        """
        Process the test or train data in batches, using multi-threading for each batch.
    
        :param is_threshold: Whether to compute for threshold (uses training data).
        :param batch_size: Size of each batch.
        :param threads: Number of threads to use for parallel processing.
        """
        # Select data
        if compute_threshold:
            data = self.train_data_matrix
        else:
            flattened_test = self.test_data.flatten()
            self.test_data_matrix = self.transform_to_matrix(flattened_test)
            data = self.test_data_matrix
    
        n_test = data.shape[0]
        n_batches = (n_test + batch_size - 1) // batch_size  # Compute number of batches
        all_scores = []
    
        # Helper function to process a single batch
        def process_batch(start_idx, end_idx):
            batch = data[start_idx:end_idx]
            if self.metric == 'cosine':
                distance_matrix = self.calculate_cosine_distances(batch, self.train_data_matrix)
            elif self.metric == 'mahalanobis':
                distance_matrix = self.calculate_mahalanobis_distances(batch, self.train_data_matrix, self.sigma)
            # Return batch scores
            return np.sum(np.sort(distance_matrix, axis=1)[:, :self.k], axis=1)
    
        
            # Multi-threaded batch processing
        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Submit each batch to the thread pool, keeping track of batch indices
            futures = []
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_test)
                future = executor.submit(process_batch, start_idx, end_idx)
                futures.append((batch_idx, future))
            
            # Collect results in order
            results = [None] * n_batches  # Pre-allocate list for ordered results
            for batch_idx, future in futures:
                results[batch_idx] = future.result()
        
        # Concatenate all scores in the correct order
        all_scores = np.concatenate(results)
        return all_scores


    def calc_anomaly(self, mode='expanding'):
        self.train_func()
        self.anomaly_errors = np.zeros(self.window_length-1)
        self.anomaly_errors = np.append(self.anomaly_errors, self.test_func())
        return pd.Series(self.anomaly_errors)


    def calc_anomaly_window_based(self):
        """
        Calculate anomaly scores using a window-based aggregation approach.
        """
        self.train_func()

        # Step 1: Compute anomaly scores for all windows
        anomaly_scores = self.test_func()

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
        self.anomaly_errors = pd.Series(timestep_errors)
        # Step 5: Return per-timestep anomaly scores as a pandas Series
        return pd.Series(timestep_errors)
    
    
    def calculate_cosine_distances(self,test, training_data):
        # Normalize training data
        train_norms = np.linalg.norm(training_data, axis=1, keepdims=True)
        train_normalized = training_data / train_norms  # Shape: (n_train, d)
    
        # Normalize test data
        test_norms = np.linalg.norm(test, axis=1, keepdims=True)
        test_normalized = test / test_norms  # Shape: (n_test, d)
    
        # Batch compute cosine similarity (dot product of normalized vectors)
        similarity_matrix = np.dot(test_normalized, train_normalized.T)  # Shape: (n_test, n_train)
    
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix  # Shape: (n_test, n_train)
    
        return distance_matrix

    def calculate_mahalanobis_distances(self,test, training_data, inv_covmat):
        # Compute differences between test and train batches
        diff = test[:, np.newaxis, :] - training_data[np.newaxis, :, :]  # Shape: (n_test, n_train, d)
    
        # Batch apply Mahalanobis formula
        left_term = np.einsum('ijk,kl->ijl', diff, inv_covmat)  # Shape: (n_test, n_train, d)
        mahalanobis_distance_squared = np.einsum('ijk,ijk->ij', left_term, diff)  # Shape: (n_test, n_train)
    
        return np.sqrt(mahalanobis_distance_squared)  # Shape: (n_test, n_train)



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
            'anomaly_scores': self.anomaly_errors.tolist() if self.anomaly_errors is not None else None
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
        self.anomaly_errors = np.array(state['anomaly_scores']) if state['anomaly_scores'] is not None else None

        print(f"State loaded from {file_path}")
    
    def plot_results(self,size=800):
        # Flattening arrays to ensure they are 1D
        test_data = self.test_data.ravel()  # Convert to 1D array
        anomaly_preds = self.anomaly_preds  # Already 1D
        anomaly_errors = (self.anomaly_preds  > self.threshold).astype(int)
        labels = self.labels.ravel()  # Convert to 1D array
    
        # Check if all inputs have the same length
        if not (len(test_data) == len(labels) == len(anomaly_preds) == len(anomaly_errors)):
            raise ValueError("All input arrays must have the same length.")
    
        # Determine plot width based on length of test_data
        plot_width = min(size, len(test_data) //10)  # Ensure a minimum width of 800, scale with data length
    
        # Create a figure
        fig = go.Figure()
    
        # Add traces for test data, predictions, and anomaly errors
        fig.add_trace(go.Scatter(x=list(range(len(test_data))),
                                 y=test_data,
                                 mode='lines',
                                 name='Test Data',
                                 line=dict(color='blue')))
    
    
        fig.add_trace(go.Scatter(x=list(range(len(anomaly_errors))),
                                 y=anomaly_errors,
                                 mode='lines',
                                 name='Anomaly Errors',
                                 line=dict(color='red')))
    
        # Highlight points in test_data where label is 1
        label_indices = [i for i in range(len(labels)) if labels[i] == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[test_data[i] for i in label_indices],
                                     mode='markers',
                                     name='Labels on Test Data',
                                     marker=dict(color='orange', size=10)))
    
        # Highlight points in predictions where anomaly_preds is 1
        anomaly_pred_indices = [i for i in range(len(anomaly_preds)) if anomaly_preds[i] == 1]
        if anomaly_pred_indices:
            fig.add_trace(go.Scatter(x=anomaly_pred_indices,
                                     y=[anomaly_errors[i] for i in anomaly_pred_indices],
                                     mode='markers',
                                     name='Anomaly Predictions',
                                     marker=dict(color='green', size=10)))
    
        # Set the layout
        fig.update_layout(title='Test Data, Predictions, and Anomalies',
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
    
    
