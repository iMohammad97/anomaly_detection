import numpy as np
import pandas as pd
import logging
from back_tester import ExpandingWalkForward, RollingWalkForward

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
        flattened_train = X_train.to_numpy().flatten()
        if len(flattened_train) < self.window_length:
            raise ValueError(f"Training data length ({len(flattened_train)}) must be greater than or equal to the window length ({self.window_length}).")
        self.training_data = self.transform_to_matrix(flattened_train)
        return None

    def test_func(self, X_test):
        flattened_test = X_test.to_numpy().flatten()
        # print(X_test)
        # if len(flattened_test) < self.window_length:
        #     raise ValueError(f"Test data length ({len(flattened_test)}) must be greater than or equal to the window length ({self.window_length}).")
        self.test_data = self.transform_to_matrix(flattened_test)

        if self.metric == 'cosine':
            batch = self.test_data
            distance_matrix = self.calculate_cosine_distances(batch, self.training_data)
        elif self.metric == 'mahalanobis':
            batch = self.test_data
            distance_matrix = self.calculate_mahalanobis_distances(batch, self.training_data, self.sigma)

        # Find the k smallest distances and sum them up for each test data point in the batch
        results = np.sum(np.sort(distance_matrix, axis=1)[:, :self.k], axis=1)
        return results

    def calc_anomaly(self, mode='expanding'):
        self.train_func(self.training_data,self.training_data)
        self.y_anomaly = np.zeros(self.window_length-1)
        self.y_anomaly = np.append(self.y_anomaly, self.test_func(self.test_data))
        return pd.Series(self.y_anomaly)

    def calculate_cosine_distances(self, test_data, train_data):
        # Normalize training data
        train_norms = np.linalg.norm(train_data, axis=1)
        train_normalized = train_data / train_norms[:, np.newaxis]

        # Normalize test data
        test_norms = np.linalg.norm(test_data, axis=1)
        test_normalized = test_data / test_norms[:, np.newaxis]

        # Compute cosine similarity matrix
        similarity_matrix = np.dot(test_normalized, train_normalized.T)

        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix

        return distance_matrix

    def calculate_mahalanobis_distances(self, test_data, train_data, inv_covmat):
        # Calculate differences between each test and train pair
        diff = test_data[:, np.newaxis, :] - train_data
        # Apply the Mahalanobis formula using matrix multiplication
        left_term = np.dot(diff, inv_covmat)
        mahalanobis_distance_squared = np.einsum('ijk,ijk->ij', left_term, diff)

        return np.sqrt(mahalanobis_distance_squared)

#use knn

    # knn_model = TimeSeriesAnomalyDetectorKNN(window_length=30, k=10, train=train_data, test=test_data, metric='cosine')
    # try:
    #     knn_model.calc_anomaly()
    # except ValueError as e:
    #     logging.error(e)


