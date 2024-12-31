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
