from metrics.timepoint_precision import pointwise_precision
from metrics.event_recall import event_wise_recall

def composite_f_score(y_true, y_pred, y_true_events, y_pred_events):
    """
    Combines timepoint precision and event-wise recall into a single F-score.
    """
    prt = pointwise_precision(y_true, y_pred)
    rece = event_wise_recall(y_true_events, y_pred_events)
    if prt + rece == 0:
        return 0
    return 2 * (prt * rece) / (prt + rece)
