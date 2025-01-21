from tensorflow.keras import layers, models
from anomaly_models.AE import create_windows
import os
import json
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from tqdm.notebook import trange
from tensorflow.keras.utils import custom_object_scope


# Custom loss class for stationarity
class StationaryLoss(layers.Layer):
    def call(self, latent, mean_coef: float = 1.0, std_coef: float = 1.0):
        latent_avg = tf.reduce_mean(latent, axis=0)
        mse_loss = tf.reduce_mean(tf.abs(latent_avg))
        self.add_loss(mean_coef * mse_loss)
        
        latent_std = tf.math.reduce_std(latent, axis=0)
        std_loss = tf.reduce_mean(tf.abs(latent_std - 1.0))
        self.add_loss(std_coef * std_loss)
        
        # Store losses for tracking
        self.mse_loss = mean_coef * mse_loss
        self.std_loss = std_coef * std_loss
        
        return latent


class SeasonalStationaryLSTMAutoencoder:
    def __init__(self, train_data, test_data, labels, timesteps: int = 128, features: int = 1, latent_dim: int = 32, lstm_units: int = 64, step_size: int = 1, threshold_sigma=2.0, seed: int = 0):
        self.train_data = train_data
        self.test_data = test_data
        self.train_data_window = create_windows(self.train_data, timesteps, step_size)
        self.test_data_window = create_windows(self.test_data, timesteps, 1)
        self.timesteps = timesteps
        self.features = features
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.model = None
        self.threshold_sigma = threshold_sigma
        self.threshold = 0
        self.predictions_windows = np.zeros(len(self.test_data_window))
        self.anomaly_preds = np.zeros(len(self.test_data))
        self.anomaly_errors = np.zeros(len(self.test_data))
        self.predictions = np.zeros(len(self.test_data))
        self.labels = labels
        self.name = "LSTM_SSAE"
        self.losses = {"mse": [], "mean": [], "std": [], "seasonality": []}
        set_seed(seed)
        self._build_model()

    def _build_model(self):
        # Encoder
        inputs = tf.keras.Input(shape=(self.timesteps, self.features))
        x = layers.LSTM(self.lstm_units, return_sequences=True)(inputs)
        x = layers.LSTM(self.latent_dim, return_sequences=False)(x)
        latent = layers.Dense(self.latent_dim)(x)

        # Apply custom loss to the latent space
        latent_with_loss = StationaryLoss()(latent, mean_coef=1.0, std_coef=1.0)

        # Decoder for the seasonality component
        seasonality = layers.RepeatVector(self.timesteps)(latent_with_loss)
        seasonality = layers.LSTM(self.latent_dim, return_sequences=True)(seasonality)
        seasonality = layers.TimeDistributed(layers.Dense(self.features))(seasonality)

        # Decoder for residuals (non-seasonal part)
        residual = layers.Subtract()([inputs, seasonality])

        # Outputs
        seasonality_length = layers.Dense(1, activation='linear', name='seasonality_length')(latent_with_loss)

        # Final model outputs
        outputs = [seasonality_length, seasonality, residual]

        # Model
        self.model = models.Model(inputs, outputs)

    def train(self, batch_size: int = 32, epochs: int = 50, optimizer: str = 'adam', patience: int = 5, seed: int = 42, shuffle: bool = False):
        set_seed(seed)
        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)

        mse_loss_fn = tf.keras.losses.MeanSquaredError()

        # Tracking the losses
        mse_loss_tracker = tf.keras.metrics.Mean(name="mse_loss")
        mean_loss_tracker = tf.keras.metrics.Mean(name="mean_loss")
        std_loss_tracker = tf.keras.metrics.Mean(name="std_loss")
        seasonality_loss_tracker = tf.keras.metrics.Mean(name="seasonality_loss")

        best_epoch_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in (pbar := trange(epochs)):
            mse_loss_tracker.reset_state()
            mean_loss_tracker.reset_state()
            std_loss_tracker.reset_state()
            seasonality_loss_tracker.reset_state()

            if shuffle:
                np.random.shuffle(self.train_data_window)

            for step in trange(0, len(self.train_data_window), batch_size, leave=False):
                batch_data = self.train_data_window[step:step + batch_size]
                epoch_loss = 0  # To track total loss in the epoch
                with tf.GradientTape() as tape:
                    # Forward pass
                    seasonality_steps, seasonality, residual = self.model(batch_data, training=True)
                    seasonality_steps = tf.cast(seasonality_steps, tf.int32)

                    # Compute the reconstruction loss (difference between input and the predicted residuals + seasonality)
                    reconstruction = seasonality + residual
                    mse_loss = mse_loss_fn(batch_data, reconstruction)

                    # Seasonality loss (self-contrast)
                    seasonality_loss = tf.reduce_mean(
                        tf.abs(seasonality[:, seasonality_steps:, :] - seasonality[:, :-seasonality_steps, :])
                    )
                    
                    # Get custom losses from StationaryLoss layers
                    mean_loss = tf.reduce_mean([layer.mse_loss for layer in self.model.layers if isinstance(layer, StationaryLoss)])
                    std_loss = tf.reduce_mean([layer.std_loss for layer in self.model.layers if isinstance(layer, StationaryLoss)])

                    # Total loss
                    total_loss = mse_loss + mean_loss + std_loss + seasonality_loss

                # Backpropagation
                gradients = tape.gradient(total_loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

                # Update trackers
                mse_loss_tracker.update_state(mse_loss)
                mean_loss_tracker.update_state(mean_loss)
                std_loss_tracker.update_state(std_loss)
                seasonality_loss_tracker.update_state(seasonality_loss)

                epoch_loss += total_loss

            # Log losses after each epoch
            self.losses['mse'].append(float(mse_loss_tracker.result().numpy()))
            self.losses['mean'].append(float(mean_loss_tracker.result().numpy()))
            self.losses['std'].append(float(std_loss_tracker.result().numpy()))
            self.losses['seasonality'].append(float(seasonality_loss_tracker.result().numpy()))
            pbar.set_description(
                f"MSE Loss = {self.losses['mse'][-1]:.4f}, Mean Loss = {self.losses['mean'][-1]:.4f}, "
                f"STD Loss = {self.losses['std'][-1]:.4f}, Seasonality Loss = {self.losses['seasonality'][-1]:.4f}"
            )

            # Early stopping logic
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

    def compute_threshold(self):
        rec = self.model.predict(self.train_data_window, verbose=0)
        mse = np.mean(np.square(self.train_data_window - rec), axis=(1, 2))
        self.threshold = np.mean(mse) + self.threshold_sigma * np.std(mse)

    def evaluate(self, batch_size=32):
        length = self.test_data.shape[0]
        self.compute_threshold()

        # Generate predictions for the test data windows
        seasonality_length, seasonality, residual = self.model.predict(self.test_data_window, batch_size=batch_size)
        mse = np.mean(np.square(self.test_data_window - (seasonality + residual)), axis=(1, 2))

        # Expand errors to original length
        M = mse.shape[0]
        timestep_errors = np.zeros(length)
        counts = np.zeros(length)

        for i in range(M):
            start = i
            end = i + self.timesteps - 1
            timestep_errors[start:end + 1] += mse[i]
            counts[start:end + 1] += 1

        counts[counts == 0] = 1  # Avoid division by zero
        timestep_errors /= counts  # Average overlapping windows

        # Generate anomaly predictions based on the threshold
        self.anomaly_preds = (timestep_errors > self.threshold).astype(int)
        self.anomaly_errors = timestep_errors

        counts = np.zeros(length)
        for i in range(M):
            for j in range(self.timesteps):
                timestep_index = i + j
                counts[timestep_index] += 1

        return self.anomaly_preds, self.anomaly_errors


    def save_model(self, model_path: str = "model.h5"):
        # Save the Keras model
        if self.model is not None:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save.")

    def load_model(self, model_path: str, train_path: str, test_path: str, label_path: str):
        # Use custom_object_scope for the custom layer
        with custom_object_scope({'StationaryLoss': StationaryLoss}):
            self.model = models.load_model(
                model_path,
                compile=False  # Avoid recompiling until the model is fully loaded
            )
        
        # Compile the model for evaluation or retraining
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_squared_error']
        )
        
        # Load data
        self.train_data = np.load(train_path)
        self.test_data = np.load(test_path)
        self.labels = np.load(label_path)
        
        # Recreate the windows with the newly loaded data
        self.train_data_window = create_windows(self.train_data, self.timesteps)
        self.test_data_window = create_windows(self.test_data, self.timesteps)

    def plot_results(self, size=800):
        # Flattening arrays to ensure they are 1D
        test_data = self.test_data.ravel()  # Convert to 1D array
        anomaly_preds = self.anomaly_preds  # Already 1D
        anomaly_errors = self.anomaly_errors  # Already 1D
        predictions = self.predictions  # Already 1D
        labels = self.labels.ravel()  # Convert to 1D array

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

    def plot_losses(self):
        # Plot the loss values
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses['mse'], label='MSE Reconstruction Loss')
        plt.plot(self.losses['mean'], label='Latent Mean Loss')
        plt.plot(self.losses['std'], label='Latent Standard Deviation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
