import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import pywt
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

###############################################################################
#  Don't forget to install pywavelets
#  pip install pywavelets
###############################################################################


###############################################################################
#  UTILITY FUNCTIONS
###############################################################################

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def create_wavelet_windows(data, timesteps, step_size=1, wavelet='db4', level=1):
    """
    Decompose each window into multiple wavelet channels using
    the Stationary Wavelet Transform (SWT). Return a 3D array:
        [number_of_windows, timesteps, wavelet_channels]

    :param data: 1D NumPy array of shape [time, ]
    :param timesteps: Size of each window
    :param step_size: Step size for the sliding window
    :param wavelet: Wavelet type used in pywt.swt
    :param level: Number of SWT levels (each level gives two sub-bands)
    :return: windows of shape (num_windows, timesteps, wavelet_channels)
    """
    wavelet_windows = []
    n = len(data)

    # Slide over the data and build windows
    for start in range(0, n - timesteps + 1, step_size):
        end = start + timesteps
        window = data[start:end]  # shape = (timesteps,)

        # Ensure 1D
        window = window.reshape(-1)  # shape = (timesteps,)

        # Perform Stationary Wavelet Transform
        coeffs = pywt.swt(window, wavelet=wavelet, level=level)
        # coeffs is a list of length = level
        # Each element is a tuple (cA, cD), both shape=(timesteps,)

        # Collect all cA and cD across the chosen 'level'
        channels = []
        for (cA, cD) in coeffs:
            channels.append(cA)
            channels.append(cD)

        # wavelet_data shape => (timesteps, 2*level)
        wavelet_data = np.stack(channels, axis=-1)
        wavelet_windows.append(wavelet_data)

    # Convert to NumPy array: shape => (num_windows, timesteps, wavelet_channels)
    wavelet_windows = np.array(wavelet_windows)
    return wavelet_windows


###############################################################################
#  MAIN CLASS: LSTM Autoencoder with Wavelet Decomposition
###############################################################################

class WaveletLSTMAutoencoder:
    def __init__(
        self,
        train_data,
        test_data,
        labels,
        timesteps: int = 128,
        step_size: int = 1,
        wavelet: str = 'db4',
        level: int = 1,
        lstm_units: int = 64,
        latent_dim: int = 32,
        threshold_sigma: float = 2.0,
        seed: int = 0
    ):
        """
        :param train_data: 1D numpy array of training data
        :param test_data: 1D numpy array of test data
        :param labels: 1D numpy array of labels (same length as test_data)
        :param timesteps: window size
        :param step_size: step size for sliding window
        :param wavelet: wavelet name for the SWT
        :param level: number of levels for the SWT
        :param lstm_units: number of units in LSTM layers
        :param latent_dim: dimensionality of the bottleneck LSTM
        :param threshold_sigma: sigma factor for threshold: mean + sigma*std
        :param seed: random seed for reproducibility
        """
        set_seed(seed)

        self.train_data = train_data
        self.test_data = test_data
        self.labels = labels
        self.timesteps = timesteps
        self.step_size = step_size

        # Wavelet parameters
        self.wavelet = wavelet
        self.level = level
        # wavelet_channels = 2 x level for SWT (cA & cD at each level)
        self.wavelet_channels = 2 * level

        # Prepare wavelet windows for training and testing
        self.train_data_window = create_wavelet_windows(
            self.train_data, timesteps, step_size, wavelet, level
        )
        self.test_data_window = create_wavelet_windows(
            self.test_data, timesteps, 1, wavelet, level
        )

        # LSTM autoencoder hyperparams
        self.lstm_units = lstm_units
        self.latent_dim = latent_dim
        self.threshold_sigma = threshold_sigma

        # We'll maintain a threshold per channel ( shape = [wavelet_channels] )
        self.thresholds = np.zeros(self.wavelet_channels)

        # Post-evaluation arrays
        self.anomaly_preds = np.zeros(len(self.test_data))
        self.anomaly_errors = np.zeros(len(self.test_data))
        self.predictions_windows = None  # shape => (num_test_windows, timesteps, wavelet_channels)
        self.predictions = np.zeros(len(self.test_data))

        # We'll store the channel-wise errors and the channel-wise wavelet values
        self.channel_errors = None      # shape => (len(test_data), wavelet_channels)
        self.channel_values = None      # shape => (len(test_data), wavelet_channels)

        self.losses = {'train': [], 'valid': []}

        # Build the LSTM Autoencoder
        self.model = self._build_model()

    def _build_model(self):
        """
        Create a model that takes input of shape (timesteps, wavelet_channels).
        We'll have an LSTM encoder and decoder, symmetrical around the bottleneck.
        """
        inputs = tf.keras.Input(shape=(self.timesteps, self.wavelet_channels), name='input_layer')

        # Encoder
        x = layers.LSTM(self.lstm_units, return_sequences=True, name='lstm_encoder_1')(inputs)
        x = layers.LSTM(self.latent_dim, return_sequences=False, name='lstm_encoder_2')(x)

        # Decoder
        x = layers.RepeatVector(self.timesteps, name='repeat_vector')(x)
        x = layers.LSTM(self.latent_dim, return_sequences=True, name='lstm_decoder_1')(x)
        x = layers.LSTM(self.lstm_units, return_sequences=True, name='lstm_decoder_2')(x)

        outputs = layers.TimeDistributed(layers.Dense(self.wavelet_channels), name='decoder_output')(x)

        model = models.Model(inputs, outputs, name='wavelet_lstm_autoencoder')
        return model

    def train(self, batch_size=32, epochs=50, optimizer='adam', loss='mse', patience=10, shuffle=False, seed=42):
        set_seed(seed)

        def max_diff_loss(y_true, y_pred):
            return tf.reduce_max(tf.abs(y_true - y_pred), axis=[1, 2])

        loss_function = 'mse' if loss == 'mse' else max_diff_loss

        self.model.compile(optimizer=optimizer, loss=loss_function)
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        history = self.model.fit(
            self.train_data_window,
            self.train_data_window,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=shuffle,
            validation_split=0.1,
            verbose=1,
            callbacks=[early_stopping]
        )

        self.losses['train'] = [float(l) for l in history.history['loss']]
        self.losses['valid'] = [float(l) for l in history.history['val_loss']]

    def compute_channel_thresholds(self):
        """
        Compute a separate threshold for each channel using training reconstruction errors.
        threshold[channel] = mean_error + threshold_sigma * std_error
        """
        train_recon = self.model.predict(self.train_data_window, verbose=0)
        # Shape => (num_train_windows, timesteps, wavelet_channels)

        error = np.square(self.train_data_window - train_recon)  # shape => same as above

        error_flat = error.reshape(-1, self.wavelet_channels)
        channel_means = np.mean(error_flat, axis=0)
        channel_stds  = np.std(error_flat, axis=0)

        self.thresholds = channel_means + self.threshold_sigma * channel_stds

    def evaluate(self, batch_size=32, loss='mse'):
        """
        1) Compute channel thresholds if needed
        2) Predict on test wavelet windows
        3) Compute channel-wise errors for each time step
        4) Compute channel-wise wavelet values for each time step
        5) Threshold => union => anomalies
        """
        self.compute_channel_thresholds()

        length = len(self.test_data)
        num_windows = self.test_data_window.shape[0]

        # (num_windows, timesteps, wavelet_channels)
        self.predictions_windows = self.model.predict(self.test_data_window, batch_size=batch_size)

        if loss == 'mse':
            errors = np.square(self.test_data_window - self.predictions_windows)
        else:
            errors = np.abs(self.test_data_window - self.predictions_windows)

        # We'll accumulate channel-wise errors into channel_errors[t, c]
        channel_errors = np.zeros((length, self.wavelet_channels))
        # We'll also accumulate the actual wavelet sub-band values from the test data
        channel_values = np.zeros((length, self.wavelet_channels))

        counts = np.zeros(length)

        for i in range(num_windows):
            start = i
            end = i + self.timesteps

            window_errors = errors[i]                # shape => (timesteps, wavelet_channels)
            window_values = self.test_data_window[i] # shape => (timesteps, wavelet_channels)

            for j in range(self.timesteps):
                t_index = start + j
                if t_index < length:
                    channel_errors[t_index] += window_errors[j]
                    channel_values[t_index] += window_values[j]
                    counts[t_index] += 1

        counts[counts == 0] = 1
        channel_errors /= counts[:, None]
        channel_values /= counts[:, None]

        self.channel_errors = channel_errors
        self.channel_values = channel_values

        # Decide anomaly per channel
        anomalies_by_channel = (channel_errors > self.thresholds)  # (length, wavelet_channels)
        self.anomaly_preds = anomalies_by_channel.any(axis=1).astype(int)

        # For a single numeric anomaly_error[t], we take the max across channels
        self.anomaly_errors = channel_errors.max(axis=1)

        # We won't do wavelet inverse transform; store dummy time-domain predictions
        self.predictions = np.zeros(length)

    def get_latent(self, x):
        encoder_model = models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('lstm_encoder_2').output
        )
        latent_representations = encoder_model.predict(x)
        return latent_representations

    def save_model(self, model_path="model.h5"):
        if self.model is not None:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save.")

    def load_model(self, model_path: str):
        self.model = models.load_model(model_path, compile=False)
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
        print(f"Model loaded from {model_path}")

    ############################################################################
    # PLOTTING UTILITIES
    ############################################################################
    def plot_results(self, size=800):
        """
        Plot original test data, max channel errors, anomalies, and optional labels.
        """
        test_data = self.test_data.ravel()
        anomaly_preds = self.anomaly_preds
        anomaly_errors = self.anomaly_errors
        labels = self.labels.ravel()

        if not (len(test_data) == len(labels) == len(anomaly_preds) == len(anomaly_errors)):
            raise ValueError("All input arrays must have the same length.")

        plot_width = max(size, len(test_data) // 10)
        fig = go.Figure()

        # 1) Test data
        fig.add_trace(go.Scatter(
            x=list(range(len(test_data))),
            y=test_data,
            mode='lines',
            name='Test Data',
            line=dict(color='blue')
        ))

        # 2) Anomaly errors (max across channels)
        fig.add_trace(go.Scatter(
            x=list(range(len(anomaly_errors))),
            y=anomaly_errors,
            mode='lines',
            name='Anomaly Errors (max over channels)',
            line=dict(color='red')
        ))

        # 3) True labels (optional)
        label_indices = [i for i in range(len(labels)) if labels[i] == 1]
        if label_indices:
            fig.add_trace(go.Scatter(
                x=label_indices,
                y=[test_data[i] for i in label_indices],
                mode='markers',
                name='True Anomaly Labels',
                marker=dict(color='orange', size=8)
            ))

        # 4) Predicted anomalies
        anomaly_pred_indices = [i for i in range(len(anomaly_preds)) if anomaly_preds[i] == 1]
        if anomaly_pred_indices:
            fig.add_trace(go.Scatter(
                x=anomaly_pred_indices,
                y=[test_data[i] for i in anomaly_pred_indices],
                mode='markers',
                name='Anomaly Predictions',
                marker=dict(color='green', size=8)
            ))

        fig.update_layout(
            title='Test Data & Wavelet-AE Anomaly Detection',
            xaxis_title='Time Steps',
            yaxis_title='Value',
            legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
            template='plotly',
            width=plot_width,
            height=600
        )
        fig.show()

    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses['train'], label='Training Loss')
        plt.plot(self.losses['valid'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Wavelet LSTM Autoencoder Training Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_channels(self, size=800):
        """
        For each channel:
          - Plot the average wavelet sub-band value over time in BLUE (left Y-axis).
          - Plot the reconstruction error in RED (right Y-axis).
          - Plot a horizontal line for the channel's threshold in GREEN (right Y-axis),
            using a dashed line style.
        """
        if self.channel_errors is None or self.channel_values is None:
            raise ValueError("You must call evaluate() before plotting channels.")

        time_axis = np.arange(len(self.test_data))
        num_channels = self.wavelet_channels

        # Create a subplot for each channel with secondary y-axis
        fig = make_subplots(
            rows=num_channels, cols=1,
            shared_xaxes=True,
            subplot_titles=[f"Channel {c}" for c in range(num_channels)],
            specs=[[{"secondary_y": True}] for _ in range(num_channels)]
        )

        for c in range(num_channels):
            # 1) Wavelet sub-band amplitude (BLUE, left y-axis)
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=self.channel_values[:, c],
                    name=f"Channel {c} Value",
                    mode='lines',
                    line=dict(color='blue')
                ),
                row=c+1, col=1, secondary_y=False
            )

            # 2) Reconstruction error (RED, right y-axis)
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=self.channel_errors[:, c],
                    name=f"Channel {c} Error",
                    mode='lines',
                    line=dict(color='red')
                ),
                row=c+1, col=1, secondary_y=True
            )

            # 3) Threshold line (GREEN, dashed, right y-axis)
            # We'll just draw a 2-point line from min to max of time_axis.
            fig.add_trace(
                go.Scatter(
                    x=[time_axis[0], time_axis[-1]],
                    y=[self.thresholds[c], self.thresholds[c]],
                    name=f"Channel {c} Threshold",
                    mode='lines',
                    line=dict(color='green', dash='dash')
                ),
                row=c+1, col=1, secondary_y=True
            )

            # Label the axes
            fig.update_yaxes(
                title_text="Amplitude (Blue)",
                row=c+1, col=1,
                secondary_y=False,
                showgrid=True,
                zeroline=True,
                linecolor='blue'
            )
            fig.update_yaxes(
                title_text="Error (Red/Green)",
                row=c+1, col=1,
                secondary_y=True,
                showgrid=False,
                zeroline=True,
                linecolor='red'
            )

        plot_width = max(size, len(test_data) // 10)
        # Layout settings
        fig.update_layout(
            height=300 * num_channels,
            width=plot_width,
            title_text="Channel-Wise Wavelet Values & Reconstruction Errors",
            title_x=0.5,  # center the title
            legend_title_text="Legend",
            hovermode='x unified'
        )

        fig.update_xaxes(title_text="Time Steps", row=num_channels, col=1)

        fig.show()


# EXAMPLE USAGE:
# # Read all categories of anomalies
# ts = "/content/31_train.npy"

# # Find test and labels file for the time-serie
# te_file = ts.replace("_train.npy", "_test.npy")
# lb_file = ts.replace("_train.npy", "_labels.npy")

# # Load the files into np array
# X_train = np.load(ts)
# X_test = np.load(te_file)
# Y_test = np.load(lb_file)
# print(len(X_train))

# # Suppose we have some 1D data arrays:
# train_data = X_train
# test_data  = X_test
# labels     = Y_test



# # Initialize the model with window_size=128, features=2 for [mag, phase]
# model = WaveletLSTMAutoencoder(
#     train_data=train_data,
#     test_data=test_data,
#     labels=labels,
#     timesteps=512,
#     step_size=1,
#     wavelet='db4',
#     level=6,
#     lstm_units=64,
#     latent_dim=32,
#     threshold_sigma=2.0,
#     seed=42
# )

# # Train and evaluate
# model.train(epochs=1, batch_size=32, patience=30, loss='mse')
# model.evaluate()
# model.plot_results(1000)
# model.plot_losses()
# model.plot_channels(1000)
