from tensorflow.keras import layers, models, callbacks
from anomaly_models.AE import create_windows
import os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
import json


class FourierAutoEncoder:
    def __init__(self, train_data, test_data, labels, timesteps: int = 128, features: int = 1, latent_dim: int = 32,
                 dense_units: int = 64, step_size: int = 1, threshold_sigma=2.0, seed: int = 0):
        self.train_data = train_data
        self.test_data = test_data
        self.train_data_window = create_windows(self.train_data, timesteps, step_size)
        self.test_data_window = create_windows(self.test_data, timesteps, 1)
        self.timesteps = timesteps
        self.features = features
        self.latent_dim = latent_dim
        self.dense_units = dense_units
        self.model = None  # Model is not built yet.
        self.threshold_sigma = threshold_sigma
        self.threshold = 0
        self.predictions_windows = np.zeros(len(self.test_data_window))
        self.anomaly_preds = np.zeros(len(self.test_data))
        self.anomaly_errors = np.zeros(len(self.test_data))
        self.predictions = np.zeros(len(self.test_data))
        self.labels = labels
        self.name = 'FullyConnectedAutoencoder'  # Add a name attribute to the class
        self.losses = {'train': [], 'valid': []}
        set_seed(seed)
        self._build_model()

    def _build_model(self):
        inputs = tf.keras.Input(shape=(self.timesteps, self.features), name='input_layer')

        # Apply Fourier Transform
        x = tf.signal.rfft(inputs)

        # Flatten the data
        x = layers.Flatten()(x)

        # Encoder
        x = layers.Dense(self.dense_units, activation='relu')(x)
        x = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)

        # Decoder
        x = layers.Dense(self.dense_units, activation='relu')(x)
        x = layers.Dense(self.timesteps * self.features)(x)
        x = layers.Reshape((self.timesteps, self.features))(x)

        # Apply inverse Fourier Transform
        outputs = tf.signal.irfft(x)

        self.model = models.Model(inputs, outputs, name='model')

    def compute_threshold(self):
        rec = self.model.predict(self.train_data_window, verbose=0)
        mse = np.mean(np.square(self.train_data_window - rec), axis=(1, 2))
        self.threshold = np.mean(mse) + self.threshold_sigma * np.std(mse)

    def train(self, batch_size=32, epochs=50, optimizer='adam', loss='mse', patience=10, shuffle: bool = False,
              seed: int = 42):
        set_seed(seed)

        # Custom loss function to calculate max difference
        def max_diff_loss(y_true, y_pred):
            return tf.reduce_max(tf.abs(y_true - y_pred), axis=[1, 2])

        # Determine which loss function to use
        loss_function = 'mse' if loss == 'mse' else max_diff_loss

        # Compile the model with the specified optimizer and selected loss function
        self.model.compile(optimizer=optimizer, loss=loss_function)

        # Early stopping callback
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Train the model
        history = self.model.fit(
            self.train_data_window, self.train_data_window,  # Use self.train_data for both input and output
            batch_size=batch_size,
            shuffle=shuffle,
            validation_split=0.1,  # Split 10% of the training data for validation
            epochs=epochs,
            verbose=1,
            callbacks=[early_stopping]
        )

        self.losses['train'] = [float(loss) for loss in history.history['loss']]
        self.losses['valid'] = [float(loss) for loss in history.history['val_loss']]

    def evaluate(self, batch_size=32, loss='mse'):
        length = self.test_data.shape[0]
        self.compute_threshold()
        # Generate predictions for the test data windows
        self.predictions_windows = self.model.predict(self.test_data_window, batch_size=batch_size)

        if loss == 'mse':
            errors = np.mean(np.square(self.test_data_window - self.predictions_windows), axis=(1, 2))
        else:
            errors = np.max(np.abs(self.test_data_window - self.predictions_windows), axis=(1, 2))

        # Expand errors to original length
        M = errors.shape[0]
        timestep_errors = np.zeros(length)
        counts = np.zeros(length)

        for i in range(M):
            start = i
            end = i + self.timesteps - 1
            timestep_errors[start:end + 1] += errors[i]
            counts[start:end + 1] += 1

        counts[counts == 0] = 1  # Avoid division by zero
        timestep_errors /= counts  # Average overlapping windows

        self.anomaly_preds = (timestep_errors > self.threshold).astype(int)
        self.anomaly_errors = timestep_errors

        counts = np.zeros(length)
        self.predictions = np.zeros(length)
        for i in range(M):
            for j in range(self.timesteps):
                timestep_index = i + j
                if timestep_index < length:
                    self.predictions[timestep_index] += self.predictions_windows[i, j]
                    counts[timestep_index] += 1

        for i in range(length):
            if counts[i] > 0:
                self.predictions[i] /= counts[i]

        self.predictions = np.nan_to_num(self.predictions)

    def get_latent(self, x):
        encoder_model = models.Model(inputs=self.model.input, outputs=self.model.get_layer('latent').output)
        latent_representations = encoder_model.predict(x)
        return latent_representations

    def save_model(self, model_path: str = "model.h5"):
        """Save the state of the object and the Keras model."""
        # Save the Keras model
        if self.model is not None:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save.")

    def load_model(self, model_path: str, train_path: str, test_path: str, label_path: str):
        """
        Load the Keras model from the specified file paths and evaluate it.

        :param model_path: Path to the saved Keras .h5 (or SavedModel) file.
        :param train_path: Path to a file containing the training data (e.g., .npy).
        :param test_path: Path to a file containing the test data (e.g., .npy).
        :param label_path: Path to a file containing the labels (e.g., .npy).
        """

        # Load the model
        self.model = models.load_model(
            model_path,
            compile=False
        )

        # As we DO need to compile for evaluation or re-training
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

        # Evaluate the model on the newly loaded data
        # This will populate self.threshold, self.predictions_windows, self.anomaly_preds, etc.
        # self.evaluate()
    def plot_results(self,size=800):
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
        plot_width = max(size, len(test_data) //10)  # Ensure a minimum width of 800, scale with data length

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

    def plot_losses(self):
        # Plot the loss values
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses['train'], label='Training Loss')
        plt.plot(self.losses['valid'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
