from metrics.event_recall import event_wise_recall
from metrics.timepoint_precision import pointwise_precision
from metrics.f_composite import composite_f_score
from metrics.auc_roc import compute_auc_roc
from metrics.auc_pr import compute_auc_pr
from metrics.auc_event import custom_auc_with_perfect_point
import numpy as np


def get_all_metrics(anomaly_preds, anomaly_scores, Y_test):
    """
    Evaluate the model given the anomaly predictions and scores.
    
    Parameters:
        - anomaly_preds: 1D array of predicted anomalies (0/1).
        - anomaly_scores: 1D array of continuous anomaly scores.
        - threshold: threshold for classification (not used in current implementation).
        - Y_test: Ground truth labels for anomalies.
    
    Returns:
        Various evaluation metrics: precision, recall, F1-score, AUC-ROC, AUC-PR, etc.
    """
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
    custom_auc_val = custom_auc_with_perfect_point(Y_test, anomaly_scores, threshold_steps=100, plot=False)
    
    return {
        'pointwise_precision': prt,
        'event_wise_recall': rece,
        'composite_f_score': fc1,
        'auc_roc': auc_roc_val,
        'auc_pr': auc_pr_val,
        'custom_auc': custom_auc_val
    }
