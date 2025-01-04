import numpy as np

# Import from your anomaly_models.AE
from anomaly_models.AE import (
    impute_nans,
    create_windows,
    build_lstm_autoencoder,
    build_lstm_vae,
    build_lstm_dae2,
    train_autoencoder,
    infer_anomalies
)

# Import metrics
from metrics.event_recall import event_wise_recall
from metrics.timepoint_precision import pointwise_precision
from metrics.f_composite import composite_f_score
from metrics.auc_roc import compute_auc_roc
from metrics.auc_pr import compute_auc_pr
from metrics.auc_event import custom_auc_score


def train_and_evaluate_single_ts(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    Y_test: np.ndarray,
    window_size: int = 128,
    epochs: int = 10,
    batch_size: int = 64,
    latent_dim: int = 32,
    lstm_units: int = 64,
    threshold_sigma: float = 2.0
):
    """
    Train AE, VAE, DAE on ONE time-series (train part), then infer on the test part
    and compute metrics (precision, recall, f1, AUC-ROC, AUC-PR, custom AUC).
    Now also returns anomaly_preds and anomaly_scores for each model in 'metrics'.

    Parameters
    ----------
    X_train_raw : np.ndarray
        1D array or 2D shape (N,1) for the training portion of the time-series.
    X_test_raw : np.ndarray
        1D array or 2D shape (M,1) for the test portion of the same time-series.
    Y_test : np.ndarray
        1D array of shape (M,) with binary labels for anomalies.
    window_size : int
        Window size for the LSTM input.
    epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    latent_dim : int
        Dimension of the latent space.
    lstm_units : int
        Number of units in LSTM layers.
    threshold_sigma : float
        Number of std-dev above mean to set as threshold for anomaly detection.

    Returns
    -------
    dict
        {
          "thresholds": {"AE": val, "VAE": val, "DAE": val},
          "metrics": {
               "AE": {
                   "prt": ...,
                   "rece": ...,
                   "fc1": ...,
                   "auc_roc": ...,
                   "auc_pr": ...,
                   "custom_auc": ...,
                   "anomaly_preds": np.ndarray([...]),
                   "anomaly_scores": np.ndarray([...])
               },
               "VAE": {...},
               "DAE": {...}
          }
        }
    """

    # 1) Impute and shape check
    X_train_raw = impute_nans(X_train_raw)
    X_test_raw  = impute_nans(X_test_raw)
    if X_train_raw is None or X_test_raw is None:
        raise ValueError("Train or Test data is empty or all NaNs.")

    # Ensure both are 2D shape (N,1) for consistency
    if len(X_train_raw.shape) == 1:
        X_train_raw = np.expand_dims(X_train_raw, axis=-1)
    if len(X_test_raw.shape) == 1:
        X_test_raw = np.expand_dims(X_test_raw, axis=-1)
    if len(Y_test.shape) > 1:
        Y_test = Y_test.flatten()

    # 2) Compute mean/std from train
    mean_val = np.mean(X_train_raw)
    std_val  = np.std(X_train_raw) + 1e-7

    # 3) Normalize
    X_train_norm = (X_train_raw - mean_val) / std_val
    X_test_norm  = (X_test_raw - mean_val) / std_val

    # 4) Create windows for training
    train_windows = create_windows(X_train_norm, window_size)
    if train_windows is None:
        raise ValueError("Training data too short to form windows.")

    # Reshape to (batch_size, timesteps, features=1) if needed
    if len(train_windows.shape) == 2:
        X_train_windows = np.expand_dims(train_windows, axis=-1)
    else:
        X_train_windows = train_windows

    # 5) Build and train AE, VAE, DAE
    model_ae  = build_lstm_autoencoder(window_size, 1, latent_dim, lstm_units)
    _, model_ae  = train_autoencoder(model_ae,  X_train_windows, epochs=epochs, batch_size=batch_size)

    model_vae = build_lstm_vae(window_size, 1, latent_dim, lstm_units)
    _, model_vae = train_autoencoder(model_vae, X_train_windows, epochs=epochs, batch_size=batch_size)

    model_dae = build_lstm_dae2(window_size, 1, latent_dim, lstm_units)
    _, model_dae = train_autoencoder(model_dae, X_train_windows, epochs=epochs, batch_size=batch_size)

    # 6) Compute thresholds for each model from training reconstruction errors
    from anomaly_models.AE import compute_reconstruction_error

    def get_threshold(model, train_data):
        rec = model.predict(train_data, verbose=0)
        mse = np.mean(np.square(train_data - rec), axis=(1,2))
        return np.mean(mse) + threshold_sigma * np.std(mse)

    threshold_ae  = get_threshold(model_ae,  X_train_windows)
    threshold_vae = get_threshold(model_vae, X_train_windows)
    threshold_dae = get_threshold(model_dae, X_train_windows)

    thresholds_dict = {"AE": threshold_ae, "VAE": threshold_vae, "DAE": threshold_dae}

    # 7) Inference: define a helper to run inference + compute metrics on the test
    def evaluate_model(model, threshold, X_test_norm, Y_test, window_size):
        anomaly_preds, anomaly_scores = infer_anomalies(model, X_test_norm, threshold, window_size)

        # Identify events in y_true and y_pred
        y_true_starts = np.argwhere(np.diff(Y_test, prepend=0) == 1).flatten()
        y_true_ends   = np.argwhere(np.diff(Y_test, append=0) == -1).flatten()
        y_true_events = list(zip(y_true_starts, y_true_ends))

        y_pred_starts = np.argwhere(np.diff(anomaly_preds, prepend=0) == 1).flatten()
        y_pred_ends   = np.argwhere(np.diff(anomaly_preds, append=0) == -1).flatten()
        y_pred_events = list(zip(y_pred_starts, y_pred_ends))

        prt  = pointwise_precision(Y_test, anomaly_preds)
        rece = event_wise_recall(y_true_events, y_pred_events)
        fc1  = composite_f_score(Y_test, anomaly_preds)
        auc_roc_val = compute_auc_roc(Y_test, anomaly_scores)
        auc_pr_val  = compute_auc_pr(Y_test, anomaly_scores)
        custom_auc_val = custom_auc_score(Y_test, anomaly_scores, threshold_steps=100, plot=False)

        # ALSO STORE the raw anomaly_preds and anomaly_scores for later plotting
        return {
            "prt":         prt,
            "rece":        rece,
            "fc1":         fc1,
            "auc_roc":     auc_roc_val,
            "auc_pr":      auc_pr_val,
            "custom_auc":  custom_auc_val,
            "anomaly_preds":  anomaly_preds,
            "anomaly_scores": anomaly_scores
        }

    # Evaluate AE, VAE, DAE on the test set
    metrics_dict = {}
    metrics_dict["AE"]  = evaluate_model(model_ae,  threshold_ae,  X_test_norm, Y_test, window_size)
    metrics_dict["VAE"] = evaluate_model(model_vae, threshold_vae, X_test_norm, Y_test, window_size)
    metrics_dict["DAE"] = evaluate_model(model_dae, threshold_dae, X_test_norm, Y_test, window_size)

    # 8) Return everything, including anomaly_preds / anomaly_scores
    return {
        "thresholds": thresholds_dict,
        "metrics": metrics_dict
    }
