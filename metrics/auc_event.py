import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from .timepoint_precision import pointwise_precision
from .event_recall import event_wise_recall, make_event



def custom_auc_with_perfect_point(y_true, anomaly_scores, threshold_steps=100, plot=False):
    # Generate thresholds using percentiles
    percentiles = np.linspace(np.min(anomaly_scores), np.max(anomaly_scores) + 1e-7, threshold_steps)
    precision_list = []
    recall_list = []
    perfect_point_found = False

    for threshold in percentiles:
        # Convert anomaly scores to binary predictions based on threshold
        y_pred = (anomaly_scores >= threshold).astype(int)
        
        # Calculate pointwise precision
        prt = pointwise_precision(y_true, y_pred)
        
        # Calculate event-wise recall
        y_true_events, y_pred_events = make_event(y_true, y_pred)
        rece = event_wise_recall(y_true_events, y_pred_events)
        
        # Append to lists
        precision_list.append(prt)
        recall_list.append(rece)

        # Check if both precision and recall are 1
        if prt == 1 and rece == 1:
            perfect_point_found = True
            break

    # Compute AUC using precision-recall pairs
    custom_auc = auc(recall_list, precision_list)

    # Plot precision-recall curve if requested
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(recall_list, precision_list, marker='o', label=f"AUC = {custom_auc:.4f}")
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True)
        plt.legend(loc="best")
        plt.show()

    return custom_auc, perfect_point_found
