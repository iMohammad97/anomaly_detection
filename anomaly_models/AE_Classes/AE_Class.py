from tensorflow.keras import layers, models
from anomaly_models.AE import create_windows
import os
import glob
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go



class LSTMAutoencoder:
    def __init__(self, train_data, test_data, labels,timesteps: int = 128, features: int = 1, latent_dim: int = 32, lstm_units: int = 64, step_size: int = 1, threshold_sigma=2.0):

        self.train_data = train_data
        self.test_data = test_data 
        self.train_data_window = create_windows(self.train_data, timesteps, step_size)
        self.test_data_window = create_windows(self.test_data, timesteps, 1)
        self.timesteps = timesteps
        self.features = features
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.model = None  # Model is not built yet.
        self.threshold = 0
        self.predictions_windows = np.zeros(len(self.test_data))
        self.anomaly_preds  = np.zeros(len(self.test_data))
        self.anomaly_errors = np.zeros(len(self.test_data))
        self.predictions = np.zeros(len(self.test_data))
        self.labels=labels
        
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
        self.anomaly_preds   = (timestep_errors > self.threshold).astype(int)
        self.anomaly_errors = timestep_errors


        counts = np.zeros(length)
        for i in range(M):
            for j in range(self.timesteps):
                timestep_index = i + j  # This is the index in the timestep corresponding to the current prediction
                if timestep_index < length:  # Ensure we don't go out of bounds
                    self.predictions[timestep_index] += self.predictions_windows[i, j]  # Accumulate each prediction appropriately
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

    def save_model(self, path: str):

        model = self._build_model()  # Ensure model is built before saving.
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):

        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")

    def plot_results(self):
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
        
        # Add traces for test data, labels, anomaly predictions, anomaly errors, and predictions
        fig.add_trace(go.Scatter(x=list(range(len(test_data))), y=test_data, mode='lines', name='Test Data', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=list(range(len(labels))), y=labels, mode='markers', name='Labels', marker=dict(color='orange', size=8)))  # Use markers for labels
        fig.add_trace(go.Scatter(x=list(range(len(anomaly_preds))), y=anomaly_preds, mode='markers', name='Anomaly Predictions', marker=dict(color='red', size=5)))
        fig.add_trace(go.Scatter(x=list(range(len(anomaly_errors))), y=anomaly_errors, mode='lines', name='Anomaly Errors', line=dict(color='green', dash='dot')))
        fig.add_trace(go.Scatter(x=list(range(len(predictions))), y=predictions, mode='lines', name='Predictions', line=dict(color='purple')))
        
        # Set the layout
        fig.update_layout(title='Test Data and Anomalies',
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
                          template='plotly')
        
        # Show the figure
        fig.show() 
