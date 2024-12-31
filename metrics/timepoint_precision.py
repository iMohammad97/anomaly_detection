import numpy as np
from sklearn.metrics import precision_score

def pointwise_precision(y_true, y_pred):
    """
    Timepoint-wise precision: fraction of detected anomalies that are correct.
    """
    return precision_score(y_true, y_pred, zero_division=0)
