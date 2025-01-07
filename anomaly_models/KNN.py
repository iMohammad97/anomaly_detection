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
        # If X_train is a Pandas DataFrame, you can convert it to NumPy:
        # flattened_train = X_train.to_numpy().flatten()
        # If X_train is already a NumPy array, just do:
        # flattened_train = X_train.flatten()

        # A safe approach: detect type before flattening
        if hasattr(X_train, "to_numpy"):
            # It's probably a Pandas object
            flattened_train = X_train.to_numpy().flatten()
        else:
            # It's already a NumPy array
            flattened_train = X_train.flatten()

        if len(flattened_train) < self.window_length:
            raise ValueError(
                f"Training data length ({len(flattened_train)}) must be greater than or equal to window_length ({self.window_length})."
            )

        self.training_data = self.transform_to_matrix(flattened_train)
        return None

    def test_func(self, X_test):
        """
        Instead of creating the full distance matrix for all test windows,
        compute the k-NN distance row by row (or in small batches).
        This greatly reduces memory usage.
        """
        # 1) Flatten
        if hasattr(X_test, "to_numpy"):
            flattened_test = X_test.to_numpy().flatten()
        else:
            flattened_test = X_test.flatten()

        # 2) Create test windows
        self.test_data = self.transform_to_matrix(flattened_test)  # shape (num_test_windows, window_size)
        # shape of self.training_data is (num_train_windows, window_size)

        # 3) Normalize training data if metric == "cosine"
        # so we don't do it repeatedly for each row
        if self.metric == 'cosine':
            train_norms = np.linalg.norm(self.training_data, axis=1)
            train_normalized = self.training_data / train_norms[:, np.newaxis]

        # 4) For each test window, compute distances to all train windows,
        #    find k nearest, sum them.
        results = []
        for test_row in self.test_data:
            if self.metric == 'cosine':
                # compute dot w/ train_normalized
                test_norm = np.linalg.norm(test_row)
                if test_norm == 0.0:
                    # edge case: if test window is all zeros, set distance = 1?
                    # or skip. We'll handle by test_norm=1 => distance=1
                    test_norm = 1e-8
                test_normalized = test_row / test_norm
                # similarity = test_normalized ⋅ train_normalized^T => shape (num_train_windows,)
                similarity = np.dot(train_normalized, test_normalized)
                dist = 1.0 - similarity  # shape (num_train_windows,)
            elif self.metric == 'mahalanobis':
                # compute row-by-row or your existing logic
                diff = self.training_data - test_row
                left_term = np.dot(diff, self.sigma)  # shape (num_train_windows, window_size)
                # Ein-sum each row
                dist = np.sqrt(np.einsum('ij,ij->i', left_term, diff))
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            # 5) find the k smallest distances and sum them
            # np.partition is O(n) to find top k, so it’s memory-friendly
            nearest_k = np.partition(dist, self.k)[:self.k]
            knn_sum = np.sum(nearest_k)
            results.append(knn_sum)

        # Convert to a NumPy array, shape (num_test_windows,)
        return np.array(results)

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


