import numpy as np
import pandas as pd

# Imports for AE, VAE, DAE
from anomaly_models.AE import (
    impute_nans,
    create_windows,
    build_lstm_autoencoder,
    build_lstm_vae,
    build_lstm_dae2,
    train_autoencoder,
    infer_anomalies,
    compute_reconstruction_error
)

# Import your KNN model
from anomaly_models.KNN import TimeSeriesAnomalyDetectorKNN

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
    threshold_sigma: float = 2.0,
    knn_k: int = 10,
    knn_metric: str = 'cosine'
):
    """
    Train 3 LSTM-based models (AE, VAE, DAE) AND a KNN model on a single time-series
    (train part), then infer on the test part and compute metrics
    (precision, recall, f1, AUC-ROC, AUC-PR, custom AUC).

    This function now also returns 'anomaly_preds' and 'anomaly_scores' for each model
    in the 'metrics' section, so no re-inference is needed for visualization.

    Parameters
    ----------
    X_train_raw : np.ndarray
        1D array or 2D (N,1) for training portion of the time-series.
    X_test_raw : np.ndarray
        1D array or 2D (M,1) for the test portion of the time-series.
    Y_test : np.ndarray
        1D array of shape (M,) with binary labels for anomalies.
    window_size : int
        Window size for the LSTM input (and for KNN).
    epochs : int
        Number of training epochs (AE, VAE, DAE).
    batch_size : int
        Training batch size (AE, VAE, DAE).
    latent_dim : int
        Dimension of the latent space (AE, VAE, DAE).
    lstm_units : int
        Number of units in LSTM layers (AE, VAE, DAE).
    threshold_sigma : float
        Number of std-dev above mean to set as threshold for anomaly detection
        for AE, VAE, DAE.
    knn_k : int
        'k' parameter for the KNN model.
    knn_metric : str
        Metric for the KNN model, e.g. 'cosine' or 'mahalanobis'.

    Returns
    -------
    dict
        {
          "thresholds": {"AE": val, "VAE": val, "DAE": val, "KNN": val},
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
               "DAE": {...},
               "KNN": {...}
          }
        }
    """

    # 1) Impute and shape check
    X_train_raw = impute_nans(X_train_raw)
    X_test_raw  = impute_nans(X_test_raw)
    if X_train_raw is None or X_test_raw is None:
        raise ValueError("Train or Test data is empty or all NaNs.")

    # Ensure both are 2D shape (N,1)
    if len(X_train_raw.shape) == 1:
        X_train_raw = np.expand_dims(X_train_raw, axis=-1)
    if len(X_test_raw.shape) == 1:
        X_test_raw = np.expand_dims(X_test_raw, axis=-1)
    if len(Y_test.shape) > 1:
        Y_test = Y_test.flatten()

    # For simplicity, not normalizing here (or do so if you prefer)
    X_train_norm = X_train_raw
    X_test_norm  = X_test_raw

    #############################
    # PART 1: Train AE, VAE, DAE
    #############################

    # 2) Create windows for training
    train_windows = create_windows(X_train_norm, window_size)
    if train_windows is None:
        raise ValueError("Training data too short to form windows.")

    # Reshape for model input if needed
    if len(train_windows.shape) == 2:
        X_train_windows = np.expand_dims(train_windows, axis=-1)
    else:
        X_train_windows = train_windows

    # 3) Build + Train AE
    model_ae = build_lstm_autoencoder(window_size, 1, latent_dim, lstm_units)
    _, model_ae = train_autoencoder(
        model_ae, X_train_windows,
        epochs=epochs,
        batch_size=batch_size
    )

    # 4) Build + Train VAE
    model_vae = build_lstm_vae(window_size, 1, latent_dim, lstm_units)
    _, model_vae = train_autoencoder(
        model_vae, X_train_windows,
        epochs=epochs,
        batch_size=batch_size
    )

    # 5) Build + Train DAE
    model_dae = build_lstm_dae2(window_size, 1, latent_dim, lstm_units)
    _, model_dae = train_autoencoder(
        model_dae, X_train_windows,
        epochs=epochs,
        batch_size=batch_size
    )

    # 6) Compute thresholds for AE, VAE, DAE from training reconstruction errors
    def get_threshold(model, train_data):
        rec = model.predict(train_data, verbose=0)
        mse = np.mean(np.square(train_data - rec), axis=(1,2))
        return np.mean(mse) + threshold_sigma * np.std(mse)

    threshold_ae  = get_threshold(model_ae,  X_train_windows)
    threshold_vae = get_threshold(model_vae, X_train_windows)
    threshold_dae = get_threshold(model_dae, X_train_windows)

    #############################
    # PART 2: Train KNN
    #############################
    # KNN uses raw sequences or data frames for train/test
    # We'll pass X_train_raw, X_test_raw as DataFrames for the KNN constructor
    # so that we can do KNN's train_func/test_func
    from pandas import DataFrame

    # Flatten them to 1D for KNN's transform_to_matrix usage
    flattened_train = X_train_raw.flatten()
    flattened_test  = X_test_raw.flatten()

    # Convert to pandas Series or DataFrame
    df_train = DataFrame(flattened_train)
    df_test  = DataFrame(flattened_test)

    # Create KNN object
    knn_model = TimeSeriesAnomalyDetectorKNN(
        k=knn_k,
        train=df_train,
        test=df_test,
        metric=knn_metric,
        window_length=window_size
    )

    # KNN threshold: we replicate the idea of a threshold based on 'training' anomaly scores
    # We'll call calculate_anomaly_threshold from your KNN class, which uses
    # self.train_func(...) and self.test_func(...) on the training data to produce training anomaly scores
    # By default, it picks quantile=0.95, but you can also pass your own if you like
    threshold_knn = knn_model.calculate_anomaly_threshold(quantile=0.95)

    # Then for the test set, we get the anomaly scores
    knn_scores_series = knn_model.calc_anomaly()  # This returns a pd.Series
    # Convert to NumPy array
    anomaly_scores_knn = knn_scores_series.values
    # Then anomaly_preds based on threshold
    anomaly_preds_knn = (anomaly_scores_knn > threshold_knn).astype(int)

    #############################
    # PART 3: Evaluate + Store Metrics
    #############################
    # Helper: Evaluate AE/VAE/DAE on test
    def evaluate_model(model, threshold, X_test_norm, Y_test):
        # Reuse 'infer_anomalies'
        anomaly_preds, anomaly_scores = infer_anomalies(model, X_test_norm, threshold, window_size)

        # Identify events in y_true / y_pred
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

    # Evaluate AE, VAE, DAE
    metrics_ae  = evaluate_model(model_ae,  threshold_ae,  X_test_norm, Y_test)
    metrics_vae = evaluate_model(model_vae, threshold_vae, X_test_norm, Y_test)
    metrics_dae = evaluate_model(model_dae, threshold_dae, X_test_norm, Y_test)

    # Evaluate KNN
    # We already have 'anomaly_preds_knn' and 'anomaly_scores_knn'
    # We just need to compute the metrics in the same style.
    # We'll replicate the logic inline:
    def evaluate_knn(Y_test, anomaly_preds, anomaly_scores):
        # Identify events in y_true / y_pred
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

    metrics_knn = evaluate_knn(Y_test, anomaly_preds_knn, anomaly_scores_knn)

    #############################
    # PART 4: Combine All Results
    #############################

    thresholds_dict = {
        "AE":  threshold_ae,
        "VAE": threshold_vae,
        "DAE": threshold_dae,
        "KNN": threshold_knn
    }

    metrics_dict = {
        "AE":  metrics_ae,
        "VAE": metrics_vae,
        "DAE": metrics_dae,
        "KNN": metrics_knn
    }

    return {
        "thresholds": thresholds_dict,
        "metrics": metrics_dict
    }
