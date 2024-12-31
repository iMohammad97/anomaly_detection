import numpy as np
from sklearn.metrics import precision_recall_curve, auc

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
