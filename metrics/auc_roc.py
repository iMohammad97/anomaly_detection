import numpy as np
from sklearn.metrics import roc_auc_score

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
