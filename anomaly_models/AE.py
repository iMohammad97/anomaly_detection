import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


############################################
# Utility functions
############################################

def impute_nans(X: np.ndarray):
    """
    Replace NaNs with the mean value of the array.
    Return None if the entire array is NaN or empty.
    """
    if X.size == 0:
        return None
    if np.isnan(X).all():
        return None
    mean_val = np.nanmean(X)
    X = np.where(np.isnan(X), mean_val, X)
    return X


def create_windows(data, window_size: int, step_size: int = 1): 
    """ Given a 2D array `data` of shape (N, features), create overlapping windows of shape (window_size, features). 
    Returns array of shape (M, window_size, features). If data is shorter than window_size, returns None. """ 
    N = data.shape[0] 
    if N < window_size: 
        return None 
    windows = [] 
    for i in range(0, N - window_size + 1, step_size): 
        window = data[i:i+window_size] 
        windows.append(window)
    return np.stack(windows, axis=0)


############################################
# Model building and training
############################################

def build_lstm_autoencoder(timesteps: int, features: int, latent_dim: int = 32, lstm_units: int = 64):
    """
    Build a simple LSTM-based autoencoder for time-series data.
    """
    # Encoder
    inputs = tf.keras.Input(shape=(timesteps, features))
    x = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    x = layers.LSTM(latent_dim, return_sequences=False)(x)

    # Decoder
    x = layers.RepeatVector(timesteps)(x)
    x = layers.LSTM(latent_dim, return_sequences=True)(x)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(features))(x)

    model = models.Model(inputs, outputs)
    return model


def build_lstm_vae(timesteps, features, latent_dim=32, lstm_units=64):
    class Sampling(layers.Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # Encoder
    inputs = tf.keras.Input(shape=(timesteps, features))
    x = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    x = layers.LSTM(latent_dim, return_sequences=False)(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = Sampling()([z_mean, z_log_var])

    # Decoder
    x = layers.RepeatVector(timesteps)(z)
    x = layers.LSTM(latent_dim, return_sequences=True)(x)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(features))(x)

    # VAE Model
    vae = models.Model(inputs, outputs)

    # Custom KL divergence loss layer
    class KLDivergenceLayer(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            kl_loss = -0.5 * tf.reduce_mean(
                z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
            self.add_loss(kl_loss)
            return z_mean

    kl_layer = KLDivergenceLayer()([z_mean, z_log_var])

    return vae

# Impose loss on the latent space to make it stationary
class LatentAverageMSELossLayer(layers.Layer):
    def call(self, latent, mean_coef: float = 1, var_coef: float = 1):
        # Calculate the average of the latent space
        latent_avg = tf.reduce_mean(latent, axis=0)
        mse_loss = tf.reduce_mean(tf.square(latent_avg))
        self.add_loss(mean_coef * mse_loss)

        # Calculate the variance of the latent space
        latent_var = tf.reduce_mean(tf.square(latent - latent_avg), axis=0)
        var_loss = tf.reduce_mean(tf.square(latent_var - 1.0))
        self.add_loss(var_coef * var_loss)

        return latent

# Impose loss on the latent space to make it stationary
class StationaryLoss(layers.Layer):
    def call(self, latent, mean_coef: float = 1, std_coef: float = 1):
        # Calculate the average of the latent space
        latent_avg = tf.reduce_mean(latent, axis=0)
        mse_loss = tf.reduce_mean(tf.square(latent_avg))
        self.add_loss(mean_coef * mse_loss)

        # Calculate the standard deviation of the latent space
        latent_std = tf.math.reduce_std(latent, axis=0)
        std_loss = tf.reduce_mean(tf.square(latent_std - 1.0))
        self.add_loss(std_coef * std_loss)

        return latent


def build_lstm_dae(timesteps, features, latent_dim=32, lstm_units=64, mean_coef: float = 1, var_coef: float = 1):
    # Encoder
    inputs = tf.keras.Input(shape=(timesteps, features))
    x = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    x = layers.LSTM(latent_dim, return_sequences=False)(x)
    latent = layers.Dense(latent_dim)(x)

    # Apply custom loss to the latent space
    latent_with_loss = StationaryLoss()(latent, mean_coef, var_coef)

    # Decoder
    x = layers.RepeatVector(timesteps)(latent_with_loss)
    x = layers.LSTM(latent_dim, return_sequences=True)(x)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(features))(x)

    # DAE Model
    dae = models.Model(inputs, outputs)

    return dae

def build_lstm_dae2(timesteps, features, latent_dim=32, lstm_units=64):
    # Encoder
    inputs = tf.keras.Input(shape=(timesteps, features))
    x = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    x = layers.LSTM(latent_dim, return_sequences=False)(x)
    latent = layers.Dense(latent_dim)(x)

    # Decoder
    x = layers.RepeatVector(timesteps)(latent)
    x = layers.LSTM(latent_dim, return_sequences=True)(x)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(features))(x)

    # DAE Model
    dae = models.Model(inputs, outputs)

    # Custom MSE loss on latent space
    class MSELossLayer(layers.Layer):
        def call(self, inputs):
            latent = inputs
            mse_loss = tf.reduce_mean(tf.square(latent))
            self.add_loss(mse_loss)
            return latent

    mse_layer = MSELossLayer()(latent)

    return dae


def train_autoencoder(
        model: tf.keras.Model,
        X_train_windows: np.ndarray,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        validation_split: float = 0.1
):
    """
    Compile and train the LSTM autoencoder model on the given windowed data.
    Returns the training history and the trained model.
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    history = model.fit(
        X_train_windows, X_train_windows,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )
    return history, model


############################################
# Inference and anomaly scoring
############################################

def compute_reconstruction_error(model: tf.keras.Model, data_windows: np.ndarray):
    """
    Given a trained model and a set of windowed data, compute the reconstruction MSE.
    Returns a 1D array (one error value per window).
    """
    recon = model.predict(data_windows, verbose=0)
    mse = np.mean(np.square(data_windows - recon), axis=(1, 2))
    return mse


def assign_window_errors_to_timesteps(window_errors: np.ndarray, length: int, window_size: int):
    """
    Expand the window-level errors back to the original time-series length
    by averaging overlapping windows. Returns a 1D array of shape (length,).
    """
    M = window_errors.shape[0]
    timestep_errors = np.zeros(length)
    counts = np.zeros(length)

    # Each window i covers timesteps [i, i+window_size-1]
    for i in range(M):
        start = i
        end = i + window_size - 1
        timestep_errors[start:end + 1] += window_errors[i]
        counts[start:end + 1] += 1

    counts[counts == 0] = 1
    timestep_errors /= counts
    return timestep_errors


def infer_anomalies(
        model: tf.keras.Model,
        X_test_norm: np.ndarray,
        threshold: float,
        window_size: int
):
    """
    Given a trained model, normalized 1D test data, and an anomaly threshold,
    compute reconstruction errors, map them to timesteps, and return:
      - anomaly_preds (0/1 array of shape (len(X_test_norm),))
      - timestep_errors (continuous anomaly scores, same shape)
    """
    length = X_test_norm.shape[0]
    test_windows = create_windows(X_test_norm, window_size)
    if test_windows is None:
        # If the test is too short, return all zeros
        return np.zeros(length, dtype=int), np.zeros(length, dtype=float)

    # Replace any NaNs in windowed data
    if np.isnan(test_windows).any():
        test_windows = np.nan_to_num(test_windows, nan=0.0)

    # Expand dims to (num_windows, window_size, 1) if needed
    if len(test_windows.shape) == 2:
        test_windows = np.expand_dims(test_windows, axis=-1)

    test_recon_errors = compute_reconstruction_error(model, test_windows)
    timestep_errors = assign_window_errors_to_timesteps(test_recon_errors, length, window_size)
    anomaly_preds = (timestep_errors > threshold).astype(int)

    return anomaly_preds, timestep_errors
