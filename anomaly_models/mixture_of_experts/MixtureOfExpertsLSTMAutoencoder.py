# ==============================
# 1) Imports and Metric Functions
# ==============================
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import custom_object_scope
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, precision_score
import glob, os, sys
import kaleido
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm, trange
import shutil
import math


# ---------- Metrics ----------
def pointwise_precision(y_true, y_pred):
    """
    Timepoint-wise precision: fraction of detected anomalies that are correct.
    """
    return precision_score(y_true, y_pred, zero_division=0)


def make_event(y_true, y_pred):
    """
    Converts binary sequences (y_true and y_pred) into a list of (start, end) event tuples.
    """
    y_true_starts = np.argwhere(np.diff(y_true.flatten(), prepend=0) == 1).flatten()
    y_true_ends = np.argwhere(np.diff(y_true.flatten(), append=0) == -1).flatten()
    y_true_events = list(zip(y_true_starts, y_true_ends))

    y_pred_starts = np.argwhere(np.diff(y_pred, prepend=0) == 1).flatten()
    y_pred_ends = np.argwhere(np.diff(y_pred, append=0) == -1).flatten()
    y_pred_events = list(zip(y_pred_starts, y_pred_ends))

    return y_true_events, y_pred_events


def event_wise_recall(y_true_events, y_pred_events):
    """
    Event-based recall. We consider an event 'detected' if the predicted event
    overlaps with the true event in any way.
    """
    detected_events = 0
    for true_event in y_true_events:
        true_start, true_end = true_event
        for pred_event in y_pred_events:
            pred_start, pred_end = pred_event
            if pred_end >= true_start and pred_start <= true_end:
                detected_events += 1
                break
    return detected_events / len(y_true_events) if y_true_events else 0


def composite_f_score(y_true, y_pred):
    """
    Combines timepoint precision and event-wise recall into a single F-score.
    """
    prt = pointwise_precision(y_true, y_pred)
    y_true_events, y_pred_events = make_event(y_true, y_pred)
    rece = event_wise_recall(y_true_events, y_pred_events)
    if prt + rece == 0:
        return 0
    return 2 * (prt * rece) / (prt + rece)


def custom_auc_with_perfect_point(y_true, anomaly_scores, threshold_steps=100, plot=False):
    """
    Generates thresholds, computes precision (timepoint) and recall (event-wise)
    pairs, checks for a perfect point, and computes the AUC (area under the curve)
    on the PR plane.
    """
    percentiles = np.linspace(np.min(anomaly_scores), np.max(anomaly_scores) + 1e-7, threshold_steps)
    precision_list = []
    recall_list = []
    perfect_point_found = False

    for threshold in percentiles:
        y_pred = (anomaly_scores >= threshold).astype(int)
        prt = pointwise_precision(y_true, y_pred)

        y_true_events, y_pred_events = make_event(y_true, y_pred)
        rece = event_wise_recall(y_true_events, y_pred_events)

        precision_list.append(prt)
        recall_list.append(rece)

        if prt == 1 and rece == 1:
            perfect_point_found = True
            break

    # Compute AUC (precision vs recall)
    custom_area = auc(recall_list, precision_list)

    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(recall_list, precision_list, marker='o', label=f"AUC = {custom_area:.4f}")
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True)
        plt.legend(loc="best")
        plt.show()

    return custom_area, perfect_point_found


def compute_auc_pr(y_true, anomaly_scores):
    """
    Compute AUC-PR for time-series anomaly detection.
    """
    try:
        precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
        auc_pr = auc(recall, precision)
    except ValueError:
        print("AUC-PR computation failed: Ensure both classes (0 and 1) are present in y_true.")
        auc_pr = np.nan
    return auc_pr


def compute_auc_roc(y_true, anomaly_scores):
    """
    Compute AUC-ROC for time-series anomaly detection.
    """
    try:
        auc_roc = roc_auc_score(y_true, anomaly_scores)
    except ValueError:
        print("AUC-ROC computation failed: Ensure both classes (0 and 1) are present in y_true.")
        auc_roc = np.nan
    return auc_roc


# ========== Utils ==========
def create_windows(data, window_size: int, step_size: int = 1):
    """
    Given a 2D array data of shape (N, features), create overlapping windows
    of shape (window_size, features). Returns array of shape (M, window_size, features).
    If data is shorter than window_size, returns None.
    """
    if data is None or data.shape[0] < window_size:
        return None
    windows = []
    N = data.shape[0]
    for i in range(0, N - window_size + 1, step_size):
        window = data[i: i + window_size]
        windows.append(window)
    # Return as float32 to avoid dtype mismatch
    return np.stack(windows, axis=0).astype(np.float32)


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ==============================
# 2) Original LSTMAutoencoder Class
# ==============================

class LSTMAutoencoder:
    def __init__(self,
                 train_data,
                 test_data,
                 labels,
                 timesteps: int = 128,
                 features: int = 1,
                 latent_dim: int = 32,
                 lstm_units: int = 64,
                 step_size: int = 1,
                 threshold_sigma=2.0,
                 seed: int = 0):

        self.train_data = train_data.astype(np.float32) if train_data is not None else None
        self.test_data = test_data.astype(np.float32) if test_data is not None else None
        self.labels = labels.astype(int) if labels is not None else None

        self.timesteps = timesteps
        self.features = features
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.step_size = step_size
        self.threshold_sigma = threshold_sigma

        # Prepare windowed data
        self.train_data_window = create_windows(self.train_data, timesteps, step_size)
        self.test_data_window = create_windows(self.test_data, timesteps, 1)

        # Model placeholders
        self.model = None
        self.threshold = 0

        # Arrays to hold predictions
        if self.test_data_window is not None:
            self.predictions_windows = np.zeros(len(self.test_data_window))
        self.anomaly_preds = np.zeros(len(self.test_data)) if self.test_data is not None else None
        self.anomaly_errors = np.zeros(len(self.test_data)) if self.test_data is not None else None
        self.predictions = np.zeros(len(self.test_data)) if self.test_data is not None else None

        self.losses = {'train': [], 'valid': []}
        self.name = 'LSTMAutoencoder'  # A name attribute for the class

        set_seed(seed)
        self._build_model()

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

    def compute_threshold(self):
        if self.train_data_window is None:
            print("No training windows to compute threshold.")
            self.threshold = 9999999
            return
        rec = self.model.predict(self.train_data_window, verbose=0)
        mse = np.mean(np.square(self.train_data_window - rec), axis=(1, 2))
        self.threshold = np.mean(mse) + self.threshold_sigma * np.std(mse)

    def train(self,
              batch_size=32,
              epochs=50,
              optimizer='adam',
              loss='mse',
              patience=10,
              shuffle: bool = False,
              seed: int = 42):
        set_seed(seed)

        # Custom max-diff loss function
        def max_diff_loss(y_true, y_pred):
            # return the average of (max(|x - recon|) across timesteps)
            # so it’s a single scalar
            return tf.reduce_mean(tf.reduce_max(tf.abs(y_true - y_pred), axis=[1, 2]))

        # Determine which loss function to use
        loss_function = 'mse' if loss == 'mse' else max_diff_loss

        # Compile the model
        self.model.compile(optimizer=optimizer, loss=loss_function)

        if self.train_data_window is None:
            print("No training windows found. Skipping training.")
            return

        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )

        # Train the model
        history = self.model.fit(
            self.train_data_window, self.train_data_window,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_split=0.1,
            epochs=epochs,
            verbose=1,
            callbacks=[early_stopping]
        )

        self.losses['train'] = [float(l) for l in history.history['loss']]
        self.losses['valid'] = [float(l) for l in history.history['val_loss']]

    def evaluate(self, batch_size=32, loss='mse'):
        """
        Evaluate the model on self.test_data_window.
        Sets self.anomaly_preds, self.anomaly_errors, self.predictions.
        """
        if self.test_data_window is None or len(self.test_data_window) == 0:
            print("No test windows available for evaluation.")
            return

        length = self.test_data.shape[0]
        self.compute_threshold()

        # Generate predictions for the test data windows
        self.predictions_windows = self.model.predict(self.test_data_window, batch_size=batch_size)

        # Compute reconstruction errors
        if loss == 'mse':
            errors = np.mean(np.square(self.test_data_window - self.predictions_windows), axis=(1, 2))
        else:
            errors = np.max(np.abs(self.test_data_window - self.predictions_windows), axis=(1, 2))

        # Expand window errors to match original time steps
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

        # Compute predictions (averaged across windows)
        counts = np.zeros(length)
        self.predictions = np.zeros(length)
        for i in range(M):
            for j in range(self.timesteps):
                timestep_index = i + j
                if timestep_index < length:
                    self.predictions[timestep_index] += self.predictions_windows[i, j]
                    counts[timestep_index] += 1

        # Avoid division by zero
        for i in range(length):
            if counts[i] > 0:
                self.predictions[i] /= counts[i]

        self.predictions = np.nan_to_num(self.predictions)

    def get_latent(self, x):
        """
        Returns latent representation from the encoder part of the model.
        """
        encoder_model = models.Model(inputs=self.model.input,
                                     outputs=self.model.get_layer('latent').output)
        latent_representations = encoder_model.predict(x)
        return latent_representations

    def save_model(self, model_path: str = "model.h5"):
        """
        Save the Keras model to disk.
        """
        if self.model is not None:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save.")

    def load_model(self, model_path: str, train_path: str, test_path: str, label_path: str):
        """
        Load the Keras model from the specified file paths and set
        self.train_data, self.test_data, and self.labels accordingly.
        """
        self.model = models.load_model(model_path, compile=False)
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

        # Load data
        self.train_data = np.load(train_path).astype(np.float32)
        self.test_data = np.load(test_path).astype(np.float32)
        self.labels = np.load(label_path).astype(int)

        # Recreate windows
        self.train_data_window = create_windows(self.train_data, self.timesteps, self.step_size)
        self.test_data_window = create_windows(self.test_data, self.timesteps, 1)

        print(f"Loaded model from {model_path} and data from {train_path}, {test_path}, {label_path}.")

    def plot_results(self, save_path=None, file_format='html', size=800):
        """
        Plot test data, predictions, anomaly errors, and highlight
        labeled anomalies and predicted anomalies.
        """
        if self.test_data is None or self.labels is None:
            print("No test data or labels to plot.")
            return

        # Flatten arrays
        test_data = self.test_data.ravel()
        anomaly_preds = self.anomaly_preds
        anomaly_errors = self.anomaly_errors
        predictions = self.predictions
        labels = self.labels.ravel()

        if not (len(test_data) == len(labels) == len(anomaly_preds) == len(anomaly_errors) == len(predictions)):
            raise ValueError("All input arrays must have the same length.")

        plot_width = max(size, len(test_data) // 10)

        fig = go.Figure()
        # Test Data
        fig.add_trace(go.Scatter(x=list(range(len(test_data))),
                                 y=test_data,
                                 mode='lines',
                                 name='Test Data',
                                 line=dict(color='blue')))
        # Predictions
        fig.add_trace(go.Scatter(x=list(range(len(predictions))),
                                 y=predictions,
                                 mode='lines',
                                 name='Predictions',
                                 line=dict(color='purple')))
        # Anomaly Errors
        fig.add_trace(go.Scatter(x=list(range(len(anomaly_errors))),
                                 y=anomaly_errors,
                                 mode='lines',
                                 name='Anomaly Errors',
                                 line=dict(color='red')))

        # Labeled anomalies
        label_indices = [i for i in range(len(labels)) if labels[i] == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[test_data[i] for i in label_indices],
                                     mode='markers',
                                     name='Labels on Test Data',
                                     marker=dict(color='orange', size=10)))

        # Predicted anomalies
        anomaly_pred_indices = [i for i in range(len(anomaly_preds)) if anomaly_preds[i] == 1]
        if anomaly_pred_indices:
            fig.add_trace(go.Scatter(x=anomaly_pred_indices,
                                     y=[predictions[i] for i in anomaly_pred_indices],
                                     mode='markers',
                                     name='Anomaly Predictions',
                                     marker=dict(color='green', size=10)))

        fig.update_layout(title='Test Data, Predictions, and Anomalies',
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
                          template='plotly',
                          width=plot_width)

        # Optionally save the figure
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if file_format.lower() == 'html':
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, format=file_format)
            print(f"Plot saved to: {save_path}")

        fig.show()

    def plot_losses(self, save_path=None):
        """
        Plot training and validation losses.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses['train'], label='Training Loss')
        plt.plot(self.losses['valid'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')

        plt.show()


# ==============================
# 3) MixtureOfExpertsLSTMAutoencoder Class (With Plotting for Expert1, Expert2, and MOE Combined)
# ==============================

class MixtureOfExpertsLSTMAutoencoder:
    """
    Two LSTMAutoencoder experts (expert1, expert2) with:
      - identical architecture
      - separate trainable weights and losses
      - dynamic gating threshold (for expert1) to decide if a window is passed on to expert2
      - separate dynamic threshold for expert2, computed from the windows that actually pass
      - dedicated plot functions for:
          1) Expert1 alone
          2) Expert2 alone
          3) Final MoE combination
    """

    def __init__(self,
                 train_data,
                 test_data,
                 labels,
                 timesteps=128,
                 features=1,
                 latent_dim=32,
                 lstm_units=64,
                 step_size=1,
                 threshold_sigma=2.0,
                 seed=0,
                 loss='mse'):
        self.train_data = train_data.astype(np.float32) if train_data is not None else None
        self.test_data = test_data.astype(np.float32) if test_data is not None else None
        self.labels = labels.astype(int) if labels is not None else None
        self.timesteps = timesteps
        self.features = features
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.step_size = step_size
        self.threshold_sigma = threshold_sigma
        self.seed = seed
        self.loss_str = loss  # store 'mse' or 'max_diff_loss'

        # Prepare windowed data
        self.train_windows = create_windows(self.train_data, timesteps, step_size)
        self.test_windows = create_windows(self.test_data, timesteps, 1)

        # Build two separate sub-models
        self.expert1 = self._build_expert()
        self.expert2 = self._build_expert()

        # Dynamic thresholds
        self.dynamic_threshold_e1 = 0.0
        self.dynamic_threshold_e2 = 0.0

        # Final time-series outputs after gating
        self.final_anomaly_score = None
        self.anomaly_preds = None

        # If we want to store pure "expert1 only" or "expert2 only" reconstructions:
        self.e1_time_scores = None
        self.e1_time_preds = None
        self.e2_time_scores = None
        self.e2_time_preds = None

        set_seed(seed)

    def _build_expert(self):
        inputs = tf.keras.Input(shape=(self.timesteps, self.features))
        x = layers.LSTM(self.lstm_units, return_sequences=True)(inputs)
        x = layers.LSTM(self.latent_dim, return_sequences=False, name='latent')(x)
        x = layers.RepeatVector(self.timesteps)(x)
        x = layers.LSTM(self.latent_dim, return_sequences=True)(x)
        x = layers.LSTM(self.lstm_units, return_sequences=True)(x)
        outputs = layers.TimeDistributed(layers.Dense(self.features))(x)
        return models.Model(inputs, outputs)

    def _get_loss_fn(self):
        """
        Returns MSE or max_diff_loss function for reconstruction.
        """
        if self.loss_str == 'mse':
            return tf.keras.losses.MeanSquaredError()
        else:
            def max_diff_loss(y_true, y_pred):
                return tf.reduce_mean(tf.reduce_max(tf.abs(y_true - y_pred), axis=[1, 2]))

            return max_diff_loss

    def train(self, epochs=50, batch_size=32, patience=10, optimizer='adam'):
        """
        Custom training loop:
          - Each epoch:
              1) For each batch, get expert1 recon error.
              2) If error > dynamic_threshold_e1, pass subset to expert2.
              3) Update both experts accordingly.
              4) Recompute dynamic_threshold_e1 after each epoch from training set.
        """
        if self.train_windows is None:
            print("No training windows available. Cannot train.")
            return

        dataset = tf.data.Dataset.from_tensor_slices(self.train_windows).batch(batch_size, drop_remainder=False)

        loss_fn = self._get_loss_fn()
        opt1 = tf.keras.optimizers.get(optimizer)
        opt2 = tf.keras.optimizers.get(optimizer)

        best_loss = np.inf
        patience_counter = 0

        # Initialize threshold so that in epoch 0, we pass everything
        self.dynamic_threshold_e1 = -999999

        for epoch in range(epochs):
            epoch_losses_e1 = []
            epoch_losses_e2 = []

            for x_batch in dataset:
                x_batch = tf.cast(x_batch, tf.float32)
                with tf.GradientTape(persistent=True) as tape:
                    # Expert1 forward
                    recon1 = self.expert1(x_batch, training=True)
                    if self.loss_str == 'mse':
                        e1_errors = tf.reduce_mean(tf.square(x_batch - recon1), axis=[1, 2])
                    else:
                        e1_errors = tf.reduce_max(tf.abs(x_batch - recon1), axis=[1, 2])
                    loss_e1 = loss_fn(x_batch, recon1)

                    # Gating: pass only windows with e1_errors > threshold
                    pass_mask = tf.greater(e1_errors, self.dynamic_threshold_e1)
                    x_pass_e2 = tf.boolean_mask(x_batch, pass_mask)

                    if tf.shape(x_pass_e2)[0] > 0:
                        recon2 = self.expert2(x_pass_e2, training=True)
                        loss_e2 = loss_fn(x_pass_e2, recon2)
                    else:
                        loss_e2 = 0.0

                # Backprop
                grads1 = tape.gradient(loss_e1, self.expert1.trainable_weights)
                opt1.apply_gradients(zip(grads1, self.expert1.trainable_weights))

                if tf.shape(x_pass_e2)[0] > 0:
                    grads2 = tape.gradient(loss_e2, self.expert2.trainable_weights)
                    opt2.apply_gradients(zip(grads2, self.expert2.trainable_weights))

                del tape

                epoch_losses_e1.append(loss_e1.numpy())
                if tf.shape(x_pass_e2)[0] > 0:
                    epoch_losses_e2.append(loss_e2.numpy())
                else:
                    epoch_losses_e2.append(0.0)

            # Update dynamic_threshold_e1 after each epoch
            # Forward pass entire train set with expert1
            e1_all = []
            for x_batch_all in dataset:
                recon_all = self.expert1(x_batch_all, training=False)
                if self.loss_str == 'mse':
                    batch_e1 = tf.reduce_mean(tf.square(x_batch_all - recon_all), axis=[1, 2])
                else:
                    batch_e1 = tf.reduce_max(tf.abs(x_batch_all - recon_all), axis=[1, 2])
                e1_all.extend(batch_e1.numpy())
            e1_all = np.array(e1_all)
            mean_e1 = np.mean(e1_all)
            std_e1 = np.std(e1_all)
            self.dynamic_threshold_e1 = mean_e1 + self.threshold_sigma * std_e1

            avg_e1 = np.mean(epoch_losses_e1)
            avg_e2 = np.mean(epoch_losses_e2)
            combined_loss = avg_e1 + avg_e2

            print(f"Epoch {epoch + 1}/{epochs} "
                  f"- E1 Loss: {avg_e1:.4f} "
                  f"- E2 Loss: {avg_e2:.4f} "
                  f"- dynamic_threshold_e1: {self.dynamic_threshold_e1:.4f}")

            # Early stopping
            if combined_loss < best_loss:
                best_loss = combined_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Compute dynamic threshold for expert2
        self._compute_expert2_threshold(dataset)

    def _compute_expert2_threshold(self, dataset):
        """
        Gathers all training windows that pass e1 threshold, then gets e2 errors,
        sets dynamic_threshold_e2 = mean + sigma * std
        """
        e2_list = []
        for x_batch in dataset:
            recon1 = self.expert1(x_batch, training=False)
            if self.loss_str == 'mse':
                e1_errors = tf.reduce_mean(tf.square(x_batch - recon1), axis=[1, 2])
            else:
                e1_errors = tf.reduce_max(tf.abs(x_batch - recon1), axis=[1, 2])
            pass_mask = tf.greater(e1_errors, self.dynamic_threshold_e1)
            x_pass_e2 = tf.boolean_mask(x_batch, pass_mask)

            if tf.shape(x_pass_e2)[0] > 0:
                recon2 = self.expert2(x_pass_e2, training=False)
                if self.loss_str == 'mse':
                    e2_err = tf.reduce_mean(tf.square(x_pass_e2 - recon2), axis=[1, 2])
                else:
                    e2_err = tf.reduce_max(tf.abs(x_pass_e2 - recon2), axis=[1, 2])
                e2_list.extend(e2_err.numpy())

        if len(e2_list) == 0:
            self.dynamic_threshold_e2 = 999999.0
        else:
            mean_e2 = np.mean(e2_list)
            std_e2 = np.std(e2_list)
            self.dynamic_threshold_e2 = mean_e2 + self.threshold_sigma * std_e2

    def evaluate(self, batch_size=32):
        """
        Final gating-based evaluation:
          1) If e1_error <= threshold => normal (use e1)
             else => pass to e2 => if e2_error > e2_threshold => anomaly, else normal.
          2) Expand window-level decisions to time steps -> self.final_anomaly_score, self.anomaly_preds
        """
        if self.test_windows is None:
            print("No test windows for evaluation.")
            return

        # We store final error for each window, final binary label for each window
        window_scores = []
        window_labels = []

        ds_test = tf.data.Dataset.from_tensor_slices(self.test_windows).batch(batch_size)
        for x_batch in ds_test:
            recon1 = self.expert1(x_batch, training=False)
            if self.loss_str == 'mse':
                e1_err = tf.reduce_mean(tf.square(x_batch - recon1), axis=[1, 2])
            else:
                e1_err = tf.reduce_max(tf.abs(x_batch - recon1), axis=[1, 2])

            pass_mask = tf.greater(e1_err, self.dynamic_threshold_e1)
            x_e2 = tf.boolean_mask(x_batch, pass_mask)

            if tf.shape(x_e2)[0] > 0:
                recon2 = self.expert2(x_e2, training=False)
                if self.loss_str == 'mse':
                    e2_err = tf.reduce_mean(tf.square(x_e2 - recon2), axis=[1, 2])
                else:
                    e2_err = tf.reduce_max(tf.abs(x_e2 - recon2), axis=[1, 2])

                # reinsert into correct positions
                idx_e2 = 0
                e2_full = []
                pass_mask_np = pass_mask.numpy()
                for pm in pass_mask_np:
                    if pm:
                        e2_full.append(e2_err[idx_e2].numpy())
                        idx_e2 += 1
                    else:
                        e2_full.append(0.0)
            else:
                e2_full = [0.0] * len(pass_mask)

            # Build final error + label
            for i in range(len(e1_err)):
                if not pass_mask[i]:
                    # e1 says normal => final error = e1_err
                    window_scores.append(e1_err[i].numpy())
                    window_labels.append(0)
                else:
                    # pass to e2
                    # if e2_full[i] > dynamic_threshold_e2 => anomaly
                    if e2_full[i] > self.dynamic_threshold_e2:
                        window_labels.append(1)
                    else:
                        window_labels.append(0)
                    window_scores.append(e2_full[i])

        # Expand window_scores, window_labels to time steps
        length = self.test_data.shape[0]
        M = len(self.test_windows)
        time_scores = np.zeros(length)
        counts = np.zeros(length)

        for i in range(M):
            start = i
            end = i + self.timesteps - 1
            time_scores[start:end + 1] += window_scores[i]
            counts[start:end + 1] += 1

        counts[counts == 0] = 1
        time_scores /= counts

        time_preds = np.zeros(length, dtype=int)
        for i in range(M):
            if window_labels[i] == 1:
                start = i
                end = i + self.timesteps - 1
                time_preds[start:end + 1] = 1

        self.final_anomaly_score = time_scores
        self.anomaly_preds = time_preds

    # -------------------------------------------------------------------------
    # Plot Expert1 alone (all windows -> expert1) ignoring gating
    # -------------------------------------------------------------------------
    def plot_expert1_results(self, save_path=None, file_format='html', size=800):
        """
        Evaluate expert1 alone on all test windows (ignoring gating) using
        the final threshold_e1. Plots the resulting reconstructions, errors, anomalies.
        """
        if self.test_windows is None or self.labels is None:
            print("No test windows or labels to plot for expert1.")
            return

        # Recompute threshold_e1 if not computed
        if self.dynamic_threshold_e1 == 0.0:
            print("Warning: dynamic_threshold_e1 not set. Expert1 may not be trained or threshold not updated.")
            self.dynamic_threshold_e1 = 999999.0

        # 1) reconstruct all test windows with expert1
        e1_recon = self.expert1.predict(self.test_windows)
        # 2) compute errors
        if self.loss_str == 'mse':
            window_errors = np.mean(np.square(self.test_windows - e1_recon), axis=(1, 2))
        else:
            window_errors = np.max(np.abs(self.test_windows - e1_recon), axis=(1, 2))

        # 3) expand to time steps
        length = self.test_data.shape[0]
        M = len(window_errors)
        time_scores = np.zeros(length)
        counts = np.zeros(length)
        for i in range(M):
            start = i
            end = i + self.timesteps - 1
            time_scores[start:end + 1] += window_errors[i]
            counts[start:end + 1] += 1
        counts[counts == 0] = 1
        time_scores /= counts

        # 4) get anomaly preds
        time_preds = (time_scores > self.dynamic_threshold_e1).astype(int)

        # 5) reconstruct predictions (averaging across windows) just like LSTMAE
        time_predictions = np.zeros(length)
        counts_pred = np.zeros(length)
        for i in range(M):
            for j in range(self.timesteps):
                idx = i + j
                if idx < length:
                    time_predictions[idx] += e1_recon[i, j]
                    counts_pred[idx] += 1
        for i in range(length):
            if counts_pred[i] > 0:
                time_predictions[i] /= counts_pred[i]

        # 6) plot
        test_data = self.test_data.ravel()
        labels = self.labels.ravel()

        plot_width = max(size, len(test_data) // 10)
        fig = go.Figure()
        # Test Data
        fig.add_trace(go.Scatter(x=list(range(len(test_data))),
                                 y=test_data,
                                 mode='lines',
                                 name='Test Data',
                                 line=dict(color='blue')))
        # Expert1 Predictions
        fig.add_trace(go.Scatter(x=list(range(len(time_predictions))),
                                 y=time_predictions,
                                 mode='lines',
                                 name='Expert1 Predictions',
                                 line=dict(color='purple')))
        # E1 Errors
        fig.add_trace(go.Scatter(x=list(range(len(time_scores))),
                                 y=time_scores,
                                 mode='lines',
                                 name='Expert1 Errors',
                                 line=dict(color='red')))

        # Labeled anomalies
        label_indices = [i for i in range(len(labels)) if labels[i] == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[test_data[i] for i in label_indices],
                                     mode='markers',
                                     name='Labels',
                                     marker=dict(color='orange', size=10)))

        # Pred anomalies
        anomaly_indices = [i for i in range(len(time_preds)) if time_preds[i] == 1]
        if anomaly_indices:
            fig.add_trace(go.Scatter(x=anomaly_indices,
                                     y=[time_predictions[i] for i in anomaly_indices],
                                     mode='markers',
                                     name='Anomaly Preds (E1)',
                                     marker=dict(color='green', size=10)))

        fig.update_layout(title='Expert1 Alone Results',
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'),
                          template='plotly',
                          width=plot_width)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if file_format.lower() == 'html':
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, format=file_format)
            print(f"[Expert1] Plot saved to: {save_path}")

        fig.show()

    # -------------------------------------------------------------------------
    # Plot Expert2 alone (all windows -> expert2), ignoring gating
    # -------------------------------------------------------------------------
    def plot_expert2_results(self, save_path=None, file_format='html', size=800):
        """
        Evaluate expert2 alone on all test windows (ignoring gating) using
        the final threshold_e2. Plots the resulting reconstructions, errors, anomalies.
        """
        if self.test_windows is None or self.labels is None:
            print("No test windows or labels to plot for expert2.")
            return

        if self.dynamic_threshold_e2 == 0.0:
            print("Warning: dynamic_threshold_e2 not set. Expert2 may not be trained or threshold not updated.")
            self.dynamic_threshold_e2 = 999999.0

        # 1) reconstruct all test windows with expert2
        e2_recon = self.expert2.predict(self.test_windows)
        # 2) compute errors
        if self.loss_str == 'mse':
            window_errors = np.mean(np.square(self.test_windows - e2_recon), axis=(1, 2))
        else:
            window_errors = np.max(np.abs(self.test_windows - e2_recon), axis=(1, 2))

        # 3) expand to time steps
        length = self.test_data.shape[0]
        M = len(window_errors)
        time_scores = np.zeros(length)
        counts = np.zeros(length)
        for i in range(M):
            start = i
            end = i + self.timesteps - 1
            time_scores[start:end + 1] += window_errors[i]
            counts[start:end + 1] += 1
        counts[counts == 0] = 1
        time_scores /= counts

        # 4) get anomaly preds
        time_preds = (time_scores > self.dynamic_threshold_e2).astype(int)

        # 5) reconstruct predictions
        time_predictions = np.zeros(length)
        counts_pred = np.zeros(length)
        for i in range(M):
            for j in range(self.timesteps):
                idx = i + j
                if idx < length:
                    time_predictions[idx] += e2_recon[i, j]
                    counts_pred[idx] += 1
        for i in range(length):
            if counts_pred[i] > 0:
                time_predictions[i] /= counts_pred[i]

        # 6) plot
        test_data = self.test_data.ravel()
        labels = self.labels.ravel()

        plot_width = max(size, len(test_data) // 10)
        fig = go.Figure()
        # Test Data
        fig.add_trace(go.Scatter(x=list(range(len(test_data))),
                                 y=test_data,
                                 mode='lines',
                                 name='Test Data',
                                 line=dict(color='blue')))
        # Expert2 Predictions
        fig.add_trace(go.Scatter(x=list(range(len(time_predictions))),
                                 y=time_predictions,
                                 mode='lines',
                                 name='Expert2 Predictions',
                                 line=dict(color='purple')))
        # E2 Errors
        fig.add_trace(go.Scatter(x=list(range(len(time_scores))),
                                 y=time_scores,
                                 mode='lines',
                                 name='Expert2 Errors',
                                 line=dict(color='red')))

        # Labeled anomalies
        label_indices = [i for i in range(len(labels)) if labels[i] == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[test_data[i] for i in label_indices],
                                     mode='markers',
                                     name='Labels',
                                     marker=dict(color='orange', size=10)))

        # Pred anomalies
        anomaly_indices = [i for i in range(len(time_preds)) if time_preds[i] == 1]
        if anomaly_indices:
            fig.add_trace(go.Scatter(x=anomaly_indices,
                                     y=[time_predictions[i] for i in anomaly_indices],
                                     mode='markers',
                                     name='Anomaly Preds (E2)',
                                     marker=dict(color='green', size=10)))

        fig.update_layout(title='Expert2 Alone Results',
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'),
                          template='plotly',
                          width=plot_width)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if file_format.lower() == 'html':
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, format=file_format)
            print(f"[Expert2] Plot saved to: {save_path}")

        fig.show()

    # -------------------------------------------------------------------------
    # Plot final MoE gating results (the combined approach from self.evaluate())
    # -------------------------------------------------------------------------
    def plot_moe_final_results(self, save_path=None, file_format='html', size=800):
        """
        Plots the final gating-based reconstruction and anomalies after calling self.evaluate().
        This uses self.final_anomaly_score and self.anomaly_preds.
        For each window:
            if e1_err <= threshold => use e1 recon
            else => use e2 recon
        Expanded to time steps, we show final anomalies and reconstruction.
        """
        if self.final_anomaly_score is None or self.anomaly_preds is None:
            print("Please run .evaluate() first to get final anomaly scores & preds.")
            return

        if self.test_windows is None or self.labels is None:
            print("No test windows or labels to plot for MOE final.")
            return

        # Let's reconstruct the final "MoE reconstruction" at the window level
        # For each window i, if e1_err <= e1 threshold => e1 recon
        # else => e2 recon. Then expand to time steps.

        # We'll replicate the gating logic from evaluate, storing the actual reconstruction.
        ds_test = tf.data.Dataset.from_tensor_slices(self.test_windows).batch(32)
        window_recons = []
        for x_batch in ds_test:
            recon1 = self.expert1(x_batch, training=False)
            if self.loss_str == 'mse':
                e1_err = tf.reduce_mean(tf.square(x_batch - recon1), axis=[1, 2])
            else:
                e1_err = tf.reduce_max(tf.abs(x_batch - recon1), axis=[1, 2])

            pass_mask = tf.greater(e1_err, self.dynamic_threshold_e1)
            x_e2 = tf.boolean_mask(x_batch, pass_mask)
            recon2 = None
            if tf.shape(x_e2)[0] > 0:
                recon2 = self.expert2(x_e2, training=False)

            # we must re-insert recon2 into the correct windows
            idx_e2 = 0
            combined_batch_recons = []
            for i in range(len(x_batch)):
                if not pass_mask[i]:
                    # Use recon1
                    combined_batch_recons.append(recon1[i].numpy())
                else:
                    # Use recon2
                    if recon2 is not None:
                        combined_batch_recons.append(recon2[idx_e2].numpy())
                        idx_e2 += 1
                    else:
                        # fallback
                        combined_batch_recons.append(recon1[i].numpy())
            window_recons.extend(combined_batch_recons)

        # Now expand to time steps
        length = self.test_data.shape[0]
        M = len(self.test_windows)
        final_reconstruction_ts = np.zeros((length, self.features))
        counts = np.zeros(length)
        for i in range(M):
            start = i
            end = i + self.timesteps - 1
            recon_window = window_recons[i]  # shape (timesteps, features)
            for j in range(self.timesteps):
                idx = start + j
                if idx < length:
                    final_reconstruction_ts[idx] += recon_window[j]
                    counts[idx] += 1
        for i in range(length):
            if counts[i] > 0:
                final_reconstruction_ts[i] /= counts[i]

        # flatten for plotting if univariate
        if self.features == 1:
            final_reconstruction_ts = final_reconstruction_ts.ravel()

        time_scores = self.final_anomaly_score
        time_preds = self.anomaly_preds
        test_data = self.test_data.ravel()
        labels = self.labels.ravel()

        plot_width = max(size, len(test_data) // 10)
        fig = go.Figure()
        # Test Data
        fig.add_trace(go.Scatter(x=list(range(len(test_data))),
                                 y=test_data,
                                 mode='lines',
                                 name='Test Data',
                                 line=dict(color='blue')))
        # MOE Combined Predictions
        fig.add_trace(go.Scatter(x=list(range(len(final_reconstruction_ts))),
                                 y=final_reconstruction_ts,
                                 mode='lines',
                                 name='MoE Combined Predictions',
                                 line=dict(color='purple')))
        # MoE final anomaly errors
        fig.add_trace(go.Scatter(x=list(range(len(time_scores))),
                                 y=time_scores,
                                 mode='lines',
                                 name='MoE Anomaly Errors',
                                 line=dict(color='red')))

        # Labeled anomalies
        label_indices = [i for i in range(len(labels)) if labels[i] == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[test_data[i] for i in label_indices],
                                     mode='markers',
                                     name='Labels',
                                     marker=dict(color='orange', size=10)))

        # Predicted anomalies
        anomaly_indices = [i for i in range(len(time_preds)) if time_preds[i] == 1]
        if anomaly_indices:
            fig.add_trace(go.Scatter(x=anomaly_indices,
                                     y=[final_reconstruction_ts[i] for i in anomaly_indices],
                                     mode='markers',
                                     name='Anomaly Preds (MoE)',
                                     marker=dict(color='green', size=10)))

        fig.update_layout(title='MoE Final Results (Gating)',
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'),
                          template='plotly',
                          width=plot_width)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if file_format.lower() == 'html':
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, format=file_format)
            print(f"[MoE Final] Plot saved to: {save_path}")

        fig.show()

    # -------------------------------------------------------------------------
    # Save/Load sub-models
    # -------------------------------------------------------------------------
    def save_models(self, dir_path="moe_models"):
        """
        Saves the expert1 and expert2 models in H5 format.
        """
        os.makedirs(dir_path, exist_ok=True)
        self.expert1.save(os.path.join(dir_path, "expert1.h5"))
        self.expert2.save(os.path.join(dir_path, "expert2.h5"))
        print(f"Saved MoE experts to {dir_path}/expert1.h5 and {dir_path}/expert2.h5")

    def load_models(self, dir_path, train_path=None, test_path=None, label_path=None):
        """
        Loads expert1 and expert2 from H5 files. Also re-loads data if paths given.
        """
        expert1_path = os.path.join(dir_path, "expert1.h5")
        expert2_path = os.path.join(dir_path, "expert2.h5")
        self.expert1 = models.load_model(expert1_path, compile=False)
        self.expert2 = models.load_model(expert2_path, compile=False)
        self.expert1.compile(optimizer='adam', loss='mean_squared_error')
        self.expert2.compile(optimizer='adam', loss='mean_squared_error')

        if train_path and test_path and label_path:
            self.train_data = np.load(train_path).astype(np.float32)
            self.test_data = np.load(test_path).astype(np.float32)
            self.labels = np.load(label_path).astype(int)
            self.train_windows = create_windows(self.train_data, self.timesteps, self.step_size)
            self.test_windows = create_windows(self.test_data, self.timesteps, 1)
        print(f"Loaded MoE experts from {expert1_path} and {expert2_path}.")


# ==============================
# 4) Universal Evaluation Function (extended for MoE)
# ==============================

def evaluate_model_and_save_results(
        model_class,
        model_path,
        results_csv_path,
        # For LSTMAutoencoder or StationaryLSTMAutoencoder:
        train_path=None,
        test_path=None,
        label_path=None,
        loss_type='mse',
        # For TransformerAE (PyTorch):
        test_loader=None,
        train_data=None,
        test_data=None,
        labels=None,
        # For MixtureOfExpertsLSTMAutoencoder:
        moe_dir_path=None
):
    """
    Evaluates one of these classes:
      - LSTMAutoencoder (Keras)
      - StationaryLSTMAutoencoder (Keras)
      - TransformerAE (PyTorch)
      - MixtureOfExpertsLSTMAutoencoder
    and saves results (AUC metrics, F1, etc.) to CSV.
    """
    # CASE 1: LSTMAutoencoder
    if model_class.__name__ == "LSTMAutoencoder":
        if not (train_path and test_path and label_path):
            raise ValueError("For LSTMAutoencoder, provide train_path, test_path, label_path.")

        dummy_model = model_class(
            train_data=np.array([]),
            test_data=np.array([]),
            labels=np.array([])
        )
        dummy_model.load_model(model_path, train_path, test_path, label_path)

        # Evaluate using the specified loss_type
        dummy_model.evaluate(loss=loss_type)

        y_true = dummy_model.labels.flatten()
        anomaly_scores = dummy_model.anomaly_errors
        y_pred = dummy_model.anomaly_preds

    # CASE 2: StationaryLSTMAutoencoder (if used)
    elif model_class.__name__ == "StationaryLSTMAutoencoder":
        raise NotImplementedError("Extend for StationaryLSTMAutoencoder if needed.")

    # CASE 3: TransformerAE (PyTorch-based)
    elif model_class.__name__ == "TransformerAE":
        if test_loader is None:
            raise ValueError("For TransformerAE, please provide test_loader for evaluation.")
        model = model_class.load(model_path)
        results = model.predict(test_loader, train=False)
        y_true = results['anomalies']
        anomaly_scores = results['errors']
        if 'predictions' in results:
            y_pred = np.array(results['predictions'])
        else:
            threshold = np.mean(anomaly_scores) + 3.0 * np.std(anomaly_scores)
            y_pred = (anomaly_scores > threshold).astype(int)

    # CASE 4: MixtureOfExpertsLSTMAutoencoder
    elif model_class.__name__ == "MixtureOfExpertsLSTMAutoencoder":
        if not moe_dir_path or not (train_path and test_path and label_path):
            raise ValueError("For MixtureOfExpertsLSTMAutoencoder, provide moe_dir_path and data paths.")

        dummy_moe = model_class(
            train_data=np.array([]),
            test_data=np.array([]),
            labels=np.array([]),
        )
        dummy_moe.load_models(moe_dir_path, train_path, test_path, label_path)
        dummy_moe.evaluate(batch_size=32)

        y_true = dummy_moe.labels.flatten()
        anomaly_scores = dummy_moe.final_anomaly_score
        y_pred = dummy_moe.anomaly_preds

    else:
        raise ValueError("Unsupported model class. "
                         "Must be LSTMAutoencoder, StationaryLSTMAutoencoder, "
                         "TransformerAE, or MixtureOfExpertsLSTMAutoencoder.")

    # Compute metrics
    custom_auc_val, perfect_point_found = custom_auc_with_perfect_point(y_true, anomaly_scores)
    auc_pr_val = compute_auc_pr(y_true, anomaly_scores)
    auc_roc_val = compute_auc_roc(y_true, anomaly_scores)
    composite_f1_val = composite_f_score(y_true, y_pred)

    results_dict = {
        "ModelPath": model_path,
        "AUC_Custom": custom_auc_val,
        "PerfectPointFound": perfect_point_found,
        "AUC_PR": auc_pr_val,
        "AUC_ROC": auc_roc_val,
        "CompositeF1": composite_f1_val
    }

    df = pd.DataFrame([results_dict])
    df.to_csv(results_csv_path, index=False)
    print(f"Results saved to {results_csv_path}")
    print("RESULTS:")
    print(df)
