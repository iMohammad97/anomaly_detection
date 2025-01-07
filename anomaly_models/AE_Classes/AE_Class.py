from tensorflow.keras import layers, models
from anomaly_models.AE import create_windows
import os
import glob
import numpy as np
import tensorflow as tf



class LSTMAutoencoder:
    def __init__(self, train_data, test_data, timesteps: int = 128, features: int = 1, latent_dim: int = 32, lstm_units: int = 64, step_size: int = 1, threshold_sigma=2.0):

        self.train_data = train_data
        self.test_data = test_data 
        self.train_data_window = create_windows(self.train_data, timesteps, step_size)
        self.test_data_window = create_windows(self.test_data, timesteps, step_size)
        self.timesteps = timesteps
        self.features = features
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.model = None  # Model is not built yet.
        self.threshold = 0

    def _build_model(self):

        inputs = tf.keras.Input(shape=(self.timesteps, self.features), name='input_layer')
        x = layers.LSTM(self.lstm_units, return_sequences=True, name='lstm_1')(inputs)
        x = layers.LSTM(self.latent_dim, return_sequences=False, name='latent')(x)

        # Decoder
        x = layers.RepeatVector(self.timesteps, name='repeat_vector')(x)
        x = layers.LSTM(self.latent_dim, return_sequences=True, name='lstm_3')(x)
        x = layers.LSTM(self.lstm_units, return_sequences=True, name='lstm_4')(x)
        outputs = layers.TimeDistributed(layers.Dense(self.features, name='dense_output'))(x)

        self.model = models.Model(inputs, outputs, name='model')

        return self.model


    def compute_threshold(self, threshold_sigma=2.0):

        rec = self.model.predict(self.train_data_window, verbose=0)
        mse = np.mean(np.square(self.train_data_window - rec), axis=(1, 2))
        self.threshold = np.mean(mse) + threshold_sigma * np.std(mse)

    def infer_anomalies(self):
        test_windows = create_windows(self.test_data, window_size)
        if np.isnan(test_windows).any():
            test_windows = np.nan_to_num(test_windows, nan=0.0)
        if len(test_windows.shape) == 2:
            test_windows = np.expand_dims(test_windows, axis=-1)
        test_recon_errors = compute_reconstruction_error(self.model, test_windows)
        timestep_errors = assign_window_errors_to_timesteps(test_recon_errors, len(self.test_data), window_size)
        anomaly_preds = (timestep_errors > self.threshold).astype(int)
        return anomaly_preds, timestep_errors

    def train(self, batch_size=32, epochs=50,  optimizer='adam', loss='mse'):
        # Ensure the model is built before training
        self.model = self._build_model()

        # Compile the model with the specified optimizer and loss function
        self.model.compile(optimizer=optimizer, loss=loss)

        # Train the model
        self.model.fit(
            self.train_data_window, self.train_data_window,  # Use self.train_data for both input and output
            batch_size=batch_size,
            validation_split=0.1,  # Split 10% of the training data for validation
            epochs=epochs,
            verbose=1
        )


    def evaluate(self, batch_size=32):
        return self.model.evaluate(self.test_data_window, batch_size=batch_size)

    def get_latent(self, x):
        encoder_model = models.Model(inputs=self.model.input, outputs=self.model.get_layer('latent').output)
        latent_representations = encoder_model.predict(x)
        return latent_representations

    def save_model(self, path: str):

        model = self._build_model()  # Ensure model is built before saving.
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):

        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")
 
