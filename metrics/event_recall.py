import numpy as np


def make_event(y_true, y_pred):
    
    y_true_starts = np.argwhere(np.diff(y_true.flatten(), prepend=0) == 1).flatten()
    y_true_ends   = np.argwhere(np.diff(y_true.flatten(), append=0) == -1).flatten()
    y_true_events = list(zip(y_true_starts, y_true_ends))

    y_pred_starts = np.argwhere(np.diff(y_pred, prepend=0) == 1).flatten()
    y_pred_ends   = np.argwhere(np.diff(y_pred, append=0) == -1).flatten()
    y_pred_events = list(zip(y_pred_starts, y_pred_ends))

    return y_true_events,y_pred_events





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
