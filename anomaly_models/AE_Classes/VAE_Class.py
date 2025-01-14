from tensorflow.keras import layers, models
from anomaly_models.AE import create_windows
import os
import json
import numpy as np
import tensorflow as tf
from tqdm.notebook import trange
import plotly.graph_objects as go
from matplotlib import pyplot as plt


class VariationalLSTMAutoencoder:
    def __init__(self, train_data, test_data, labels, timesteps: int = 128, features: int = 1, latent_dim: int = 32,
                 lstm_units: int = 64, step_size: int = 1, threshold_sigma=2.0):

        self.train_data = train_data
        self.test_data = test_data
        self.train_data_window = create_windows(self.train_data, timesteps, step_size)
        self.test_data_window = create_windows(self.test_data, timesteps, 1)
        self.timesteps = timesteps
        self.features = features
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.model = None  # Model is not built yet.
        self.threshold_sigma = threshold_sigma
        self.threshold = 0
        self.predictions_windows = np.zeros(len(self.test_data_window))
        self.anomaly_preds = np.zeros(len(self.test_data))
        self.anomaly_errors = np.zeros(len(self.test_data))
        self.predictions = np.zeros(len(self.test_data))
        self.labels = labels
        self.name = 'LSTM_VAE'
        self.losses = {'mse': [], 'kld': []}

    def _build_model(self):
        class Sampling(layers.Layer):
            """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

            def call(self, inputs):
                z_mean, z_log_var = inputs
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        # Encoder
        inputs = tf.keras.Input(shape=(self.timesteps, self.features))
        x = layers.LSTM(self.lstm_units, return_sequences=True)(inputs)
        x = layers.LSTM(self.latent_dim, return_sequences=False)(x)
        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)
        z = Sampling()([z_mean, z_log_var])

        # KL Divergence
        class KLDivergenceLayer(layers.Layer):
            def call(self, inputs):
                z_mean, z_log_var = inputs
                kl_loss = -0.5 * tf.reduce_sum(
                    z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1, axis=-1)
                self.add_loss(tf.reduce_mean(kl_loss))  # Add KL divergence to total loss
                return kl_loss

        kl_loss = KLDivergenceLayer()([z_mean, z_log_var])

        # Decoder
        x = layers.RepeatVector(self.timesteps)(z)
        x = layers.LSTM(self.latent_dim, return_sequences=True)(x)
        x = layers.LSTM(self.lstm_units, return_sequences=True)(x)
        outputs = layers.TimeDistributed(layers.Dense(self.features))(x)

        # VAE Model
        self.model = models.Model(inputs, [outputs, kl_loss])  # Return both outputs and KL loss
        return self.model

    def compute_threshold(self):

        rec = self.model.predict(self.train_data_window, verbose=0)[0]
        mse = np.mean(np.square(self.train_data_window - rec), axis=(1, 2))
        self.threshold = np.mean(mse) + self.threshold_sigma * np.std(mse)

    def train(self, batch_size: int = 32, epochs: int = 50, optimizer: str = 'adam', patience: int = 5):
        # Ensure the optimizer is set up correctly
        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)  # Get optimizer by name
        elif not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise ValueError("Optimizer must be a string or a tf.keras.optimizers.Optimizer instance.")

        self._build_model()  # Build the model

        # Loss function
        mse_loss_fn = tf.keras.losses.MeanSquaredError()

        # Track losses
        mse_loss_tracker = tf.keras.metrics.Mean(name="mse_loss")
        kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        # Early stopping variables 
        best_epoch_loss = float('inf') 
        patience_counter = 0

        # Training loop
        for epoch in (pbar := trange(epochs)):

            mse_loss_tracker.reset_state()
            kl_loss_tracker.reset_state()
            epoch_loss = 0

            for step in trange(0, len(self.train_data_window), batch_size, leave=False):
                batch_data = self.train_data_window[step:step + batch_size]

                with tf.GradientTape() as tape:
                    # Forward pass
                    reconstructed, kl_loss = self.model(batch_data, training=True)

                    # Compute losses
                    mse_loss = mse_loss_fn(batch_data, reconstructed)  # Use MeanSquaredError class
                    total_loss = mse_loss + tf.reduce_mean(kl_loss)

                # Compute gradients and update weights
                gradients = tape.gradient(total_loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

                # Track losses
                mse_loss_tracker.update_state(mse_loss)
                kl_loss_tracker.update_state(kl_loss)

                epoch_loss += total_loss

            # Log losses after each epoch
            self.losses['mse'].append(float(mse_loss_tracker.result().numpy()))
            self.losses['kld'].append(float(kl_loss_tracker.result().numpy()))
            pbar.set_description(
                f"MSE Loss = {self.losses['mse'][-1]:.4f}, KL Divergence Loss = {self.losses['kld'][-1]:.4f}")

            # Eearly stopping
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience: 
                    print(f"Early stopping triggered after {epoch + 1} epochs.") 
                    break

    def evaluate(self, batch_size=32):
        length = self.test_data.shape[0]
        self.compute_threshold()
        # Generate predictions for the test data windows
        self.predictions_windows = self.model.predict(self.test_data_window, batch_size=batch_size)
        mse = np.mean(np.square(self.test_data_window - self.predictions_windows), axis=(1, 2))
        # Expand errors to original length
        M = mse.shape[0]
        timestep_errors = np.zeros(length)
        counts = np.zeros(length)

        # Each window i covers timesteps [i, i+window_size-1]
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
                timestep_index = i + j  # This is the index in the timestep corresponding to the current prediction
                if timestep_index < length:  # Ensure we don't go out of bounds
                    self.predictions[timestep_index] += self.predictions_windows[
                        i, j]  # Accumulate each prediction appropriately
                    counts[timestep_index] += 1

        # Divide by counts to get the average prediction
        for i in range(length):
            if counts[i] > 0:
                self.predictions[i] /= counts[i]

        # You can additionally handle any NaN values here if necessary
        self.predictions = np.nan_to_num(self.predictions)

    def get_latent(self, x):
        encoder_model = models.Model(inputs=self.model.input, outputs=self.model.get_layer('latent').output)
        latent_representations = encoder_model.predict(x)
        return latent_representations

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

        # Determine plot width based on length of test_data
        plot_width = max(size, len(test_data) * 2)  # Ensure a minimum width of 800, scale with data length

        # Create a figure
        fig = go.Figure()

        # Add traces for test data, predictions, and anomaly errors
        fig.add_trace(go.Scatter(x=list(range(len(test_data))),
                                 y=test_data,
                                 mode='lines',
                                 name='Test Data',
                                 line=dict(color='blue')))

        fig.add_trace(go.Scatter(x=list(range(len(predictions))),
                                 y=predictions,
                                 mode='lines',
                                 name='Predictions',
                                 line=dict(color='purple')))

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
                                     y=[predictions[i] for i in anomaly_pred_indices],
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

    def save_state(self, file_path: str, model_path: str = "model.h5"):
        """Save the state of the object and the Keras model."""
        # Save the Keras model
        if self.model is not None:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save.")

        # Save the rest of the attributes
        state = {
            'train_data': self.train_data.tolist(),
            'test_data': self.test_data.tolist(),
            'labels': self.labels.tolist(),
            'timesteps': self.timesteps,
            'features': self.features,
            'latent_dim': self.latent_dim,
            'lstm_units': self.lstm_units,
            'threshold': self.threshold,
            'predictions_windows': self.predictions_windows.tolist(),
            'anomaly_preds': self.anomaly_preds.tolist(),
            'anomaly_errors': self.anomaly_errors.tolist(),
            'predictions': self.predictions.tolist(),
            'losses': self.losses,
            'model_path': model_path  # Save the model path for loading later
        }
        with open(file_path, 'w') as file:
            json.dump(state, file)
        print(f"State saved to {file_path}")

    def load_state(self, file_path: str):
        """Load the state of the object and the Keras model."""
        with open(file_path, 'r') as file:
            state = json.load(file)

        # Restore the attributes
        self.train_data = np.array(state['train_data'])
        self.test_data = np.array(state['test_data'])
        self.labels = np.array(state['labels'])
        self.timesteps = state['timesteps']
        self.features = state['features']
        self.latent_dim = state['latent_dim']
        self.lstm_units = state['lstm_units']
        self.threshold = state['threshold']
        self.predictions_windows = np.array(state['predictions_windows'])
        self.anomaly_preds = np.array(state['anomaly_preds'])
        self.anomaly_errors = np.array(state['anomaly_errors'])
        self.predictions = np.array(state['predictions'])
        self.losses = state['losses']

        # Load the Keras model if a path is provided
        model_path = state.get('model_path', None)
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print("No model found to load.")

    def plot_losses(self):
        # Plot the loss values
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses['mse'], label='MSE Loss')
        plt.plot(self.losses['kld'], label='KLD Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()
