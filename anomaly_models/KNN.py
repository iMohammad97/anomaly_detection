import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class TimeSeriesAnomalyDetectorKNN:
    def __init__(self, k, train, test, dim=1, metric='cosine',step_period=10,window_length=100):
        self.window_length = window_length
        self.step_period = step_period
        self.training_data = train
        self.test_data = test
        self.labels = None
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
        """
        Calculate anomaly threshold based on training data.
        
        :param quantile: The quantile to use for the threshold. Defaults to 0.95.
        :return: Threshold value for anomaly detection.
        """
        logging.info("Calculating anomaly threshold from training data.")
        
        self.train_func(self.training_data, self.training_data)
        
        # Compute anomaly scores for training data
        anomaly_scores = self.test_func(self.training_data)
        
        # Compute the threshold based on the specified quantile
        threshold = np.quantile(anomaly_scores, quantile)
        
        logging.info(f"Calculated threshold: {threshold}")
        return threshold

#use knn

    # knn_model = TimeSeriesAnomalyDetectorKNN(window_length=30, k=10, train=train_data, test=test_data, metric='cosine')
    # try:
    #     knn_model.calc_anomaly()
    # except ValueError as e:
    #     logging.error(e)


