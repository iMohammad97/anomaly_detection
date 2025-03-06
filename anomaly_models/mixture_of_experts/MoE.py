###############################################################################
# anomaly_detection/anomaly_models/mixture_of_experts/MoE.py
###############################################################################
import numpy as np
import tensorflow as tf
import os

# Example: import our metrics from the metrics folder
# from anomaly_detection.metrics.timepoint_precision import pointwise_precision
# from anomaly_detection.metrics.event_recall import event_wise_recall
# from anomaly_detection.metrics.auc_pr import compute_auc_pr
# etc.

###############################################################################
# 1) CREATE THE MOE
###############################################################################
def create_moe(
    ExpertClass,
    train_data,
    test_data,
    labels,
    timesteps=128,
    features=1,
    threshold_sigma=2.0,
    step_size=1,
    loss_type='mse',
    seed=42,
    **expert_kwargs
):
    """
    Creates a dictionary that encapsulates everything needed for a 2-expert MoE:
      - 'expert1' and 'expert2': two instances of ExpertClass
      - 'train_data', 'test_data', 'labels': references to the datasets
      - 'threshold_e1', 'threshold_e2': dynamic gating thresholds
      - 'loss_type': 'mse' or 'max_diff_loss'
      - plus any extra fields we need for plotting or gating logic.

    :param ExpertClass: The class to instantiate for each expert (e.g. LSTMAutoencoder).
    :param train_data:  Numpy array of shape (N, features).
    :param test_data:   Numpy array of shape (M, features).
    :param labels:      Numpy array of shape (M,) for final evaluation/plotting.
    :param timesteps:   Window size.
    :param features:    Dimensionality of each time step (default=1).
    :param threshold_sigma: Controls dynamic threshold = mean + sigma*std for gating.
    :param step_size:   Step size for window creation during training.
    :param loss_type:   'mse' or 'max_diff_loss'.
    :param seed:        Random seed.
    :param expert_kwargs: Additional keyword args to pass to the ExpertClass constructor.

    :return: A dictionary named 'moe' with fields like:
             moe['expert1'], moe['expert2'], moe['train_data'], moe['test_data'],
             moe['threshold_e1'], moe['threshold_e2'], moe['loss_type'], etc.
    """
    # Convert data to float32
    X_train = train_data.astype(np.float32)
    X_test  = test_data.astype(np.float32)
    Y_test  = labels.astype(int)

    # Instantiate two experts with the same architecture/params
    # We pass in extra **expert_kwargs if we want to configure the experts differently
    expert1 = ExpertClass(
        train_data=X_train,
        test_data=X_test,
        labels=Y_test,
        timesteps=timesteps,
        features=features,
        step_size=step_size,
        seed=seed,
        loss_type=loss_type,   # Some experts might want to know their loss upfront
        **expert_kwargs
    )
    expert2 = ExpertClass(
        train_data=X_train,
        test_data=X_test,
        labels=Y_test,
        timesteps=timesteps,
        features=features,
        step_size=step_size,
        seed=seed+123,         # slightly different seed if we want
        loss_type=loss_type,
        **expert_kwargs
    )

    # Build a dictionary for our MoE
    moe = {
        'expert1': expert1,
        'expert2': expert2,
        'train_data': X_train,
        'test_data': X_test,
        'labels': Y_test,
        'threshold_e1': 0.0,
        'threshold_e2': 0.0,
        'timesteps': timesteps,
        'features': features,
        'loss_type': loss_type,
        'threshold_sigma': threshold_sigma,
        'step_size': step_size,
        'seed': seed,
        # any other fields we want...
    }
    return moe


###############################################################################
# 2) TRAINING FUNCTION
###############################################################################
def train_moe(moe, epochs=50, batch_size=32, patience=10, optimizer='adam'):
    """
    Custom training loop for the mixture-of-experts gating architecture.
      1) For each batch, compute expert1 errors, threshold gating to pass to expert2.
      2) Recompute dynamic threshold for expert1 each epoch.
      3) Then compute threshold for expert2 from the windows that actually passed.

    :param moe: The dictionary from create_moe(...).
    :param epochs: number of epochs
    :param batch_size: mini-batch size
    :param patience: early stopping
    :param optimizer: e.g. 'adam'

    The experts in 'moe' must each have:
      - a .model or forward pass
      - a .trainable_weights
    """
    from tensorflow.keras.optimizers import get as get_optimizer
    import math

    expert1 = moe['expert1']
    expert2 = moe['expert2']
    X_train = moe['train_data']
    threshold_sigma = moe['threshold_sigma']
    loss_type = moe['loss_type']

    # Build a tf.data.Dataset for training windows if our ExpertClass doesn't do it automatically
    # If our ExpertClass has .train_data_window, use that. We do an example here:
    train_windows = expert1.train_data_window
    if train_windows is None:
        print("No training windows found. Skipping training.")
        return

    dataset = tf.data.Dataset.from_tensor_slices(train_windows).batch(batch_size)

    # Decide on a custom loss function
    def loss_fn_mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred), axis=[1,2])

    def loss_fn_max_diff(y_true, y_pred):
        return tf.reduce_max(tf.abs(y_true - y_pred), axis=[1,2])

    opt_fn = get_optimizer(optimizer)
    opt1 = opt_fn
    opt2 = get_optimizer(optimizer)

    best_loss = math.inf
    patience_counter = 0

    # Initialize threshold so that in epoch 0, everything passes
    moe['threshold_e1'] = -999999

    for epoch in range(epochs):
        epoch_losses_e1 = []
        epoch_losses_e2 = []

        for x_batch in dataset:
            x_batch = tf.cast(x_batch, tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                # Expert1 forward
                recon1 = expert1.model(x_batch, training=True)
                if loss_type == 'mse':
                    e1_errors = loss_fn_mse(x_batch, recon1)  # shape (batch,)
                    loss_e1 = tf.reduce_mean(e1_errors)
                else:
                    e1_errors = loss_fn_max_diff(x_batch, recon1)  # shape (batch,)
                    loss_e1 = tf.reduce_mean(e1_errors)

                # gating
                pass_mask = tf.greater(e1_errors, moe['threshold_e1'])
                x_pass_e2 = tf.boolean_mask(x_batch, pass_mask)
                if tf.shape(x_pass_e2)[0] > 0:
                    recon2 = expert2.model(x_pass_e2, training=True)
                    if loss_type == 'mse':
                        e2_err = loss_fn_mse(x_pass_e2, recon2)
                        loss_e2 = tf.reduce_mean(e2_err)
                    else:
                        e2_err = loss_fn_max_diff(x_pass_e2, recon2)
                        loss_e2 = tf.reduce_mean(e2_err)
                else:
                    loss_e2 = 0.0

            grads1 = tape.gradient(loss_e1, expert1.model.trainable_weights)
            opt1.apply_gradients(zip(grads1, expert1.model.trainable_weights))

            if tf.shape(x_pass_e2)[0] > 0:
                grads2 = tape.gradient(loss_e2, expert2.model.trainable_weights)
                opt2.apply_gradients(zip(grads2, expert2.model.trainable_weights))

            del tape
            epoch_losses_e1.append(loss_e1.numpy())
            if tf.shape(x_pass_e2)[0] > 0:
                epoch_losses_e2.append(loss_e2.numpy())
            else:
                epoch_losses_e2.append(0.0)

        # End of epoch: compute new threshold for Expert1
        # We can do a forward pass again on entire train set, just like our code
        e1_all_err = []
        for x_batch_all in dataset:
            recon_all = expert1.model(x_batch_all, training=False)
            if loss_type == 'mse':
                batch_e1 = loss_fn_mse(x_batch_all, recon_all)  # shape (batch,)
            else:
                batch_e1 = loss_fn_max_diff(x_batch_all, recon_all)
            e1_all_err.extend(batch_e1.numpy())
        e1_all_err = np.array(e1_all_err)
        mean_e1 = np.mean(e1_all_err)
        std_e1  = np.std(e1_all_err)
        moe['threshold_e1'] = mean_e1 + threshold_sigma * std_e1

        # Compute average epoch losses
        avg_e1 = np.mean(epoch_losses_e1)
        avg_e2 = np.mean(epoch_losses_e2)
        combined_loss = avg_e1 + avg_e2

        print(f"[Epoch {epoch+1}/{epochs}] E1 Loss={avg_e1:.4f}, E2 Loss={avg_e2:.4f}, threshold_e1={moe['threshold_e1']:.4f}")

        # Early stopping
        if combined_loss < best_loss:
            best_loss = combined_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Finally compute threshold_e2
    _compute_threshold_e2(moe, dataset)


def _compute_threshold_e2(moe, dataset):
    """Gather windows that pass e1 threshold, measure e2 errors, set threshold_e2."""
    import numpy as np

    expert1 = moe['expert1']
    expert2 = moe['expert2']
    threshold_sigma = moe['threshold_sigma']
    loss_type = moe['loss_type']

    def loss_fn_mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred), axis=[1,2])

    def loss_fn_max_diff(y_true, y_pred):
        return tf.reduce_max(tf.abs(y_true - y_pred), axis=[1,2])

    e2_errs = []
    for x_batch in dataset:
        recon1 = expert1.model(x_batch, training=False)
        if loss_type == 'mse':
            e1_err = loss_fn_mse(x_batch, recon1)
        else:
            e1_err = loss_fn_max_diff(x_batch, recon1)

        pass_mask = tf.greater(e1_err, moe['threshold_e1'])
        x_pass_e2 = tf.boolean_mask(x_batch, pass_mask)
        if tf.shape(x_pass_e2)[0] > 0:
            recon2 = expert2.model(x_pass_e2, training=False)
            if loss_type == 'mse':
                e2 = loss_fn_mse(x_pass_e2, recon2)
            else:
                e2 = loss_fn_max_diff(x_pass_e2, recon2)
            e2_errs.extend(e2.numpy())

    if len(e2_errs) == 0:
        moe['threshold_e2'] = 9999999.0
    else:
        mean_e2 = np.mean(e2_errs)
        std_e2  = np.std(e2_errs)
        moe['threshold_e2'] = mean_e2 + threshold_sigma * std_e2


###############################################################################
# 3) EVALUATION FUNCTION (FINAL GATING)
###############################################################################
def evaluate_moe(moe, batch_size=32):
    """
    Runs final gating-based evaluation on the test set:
      - If e1_err <= threshold_e1 => normal
        else pass to e2 => if e2_err > threshold_e2 => anomaly
    Expands the window-level decisions to time steps and
    attaches them to moe['final_scores'] and moe['final_preds'].

    :param moe: The dictionary from create_moe(...).
    """
    import numpy as np

    expert1 = moe['expert1']
    expert2 = moe['expert2']
    X_test  = moe['test_data']
    threshold_e1 = moe['threshold_e1']
    threshold_e2 = moe['threshold_e2']
    timesteps = moe['timesteps']
    loss_type = moe['loss_type']

    # Build dataset from test windows
    test_windows = expert1.test_data_window  # or moe['expert1'].test_data_window
    if test_windows is None:
        print("No test windows to evaluate.")
        moe['final_scores'] = None
        moe['final_preds']  = None
        return

    ds_test = tf.data.Dataset.from_tensor_slices(test_windows).batch(batch_size)

    # define losses again
    def loss_fn_mse(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred), axis=[1,2])

    def loss_fn_max_diff(y_true, y_pred):
        return tf.reduce_max(tf.abs(y_true - y_pred), axis=[1,2])

    window_scores = []
    window_labels = []

    for x_batch in ds_test:
        recon1 = expert1.model(x_batch, training=False)
        if loss_type == 'mse':
            e1_err = loss_fn_mse(x_batch, recon1)
        else:
            e1_err = loss_fn_max_diff(x_batch, recon1)

        pass_mask = tf.greater(e1_err, threshold_e1)
        x_pass_e2 = tf.boolean_mask(x_batch, pass_mask)

        # get e2 errors
        if tf.shape(x_pass_e2)[0] > 0:
            recon2 = expert2.model(x_pass_e2, training=False)
            if loss_type == 'mse':
                e2_err = loss_fn_mse(x_pass_e2, recon2)
            else:
                e2_err = loss_fn_max_diff(x_pass_e2, recon2)

            # put them back
            pass_mask_np = pass_mask.numpy()
            e2_full = []
            idx_e2 = 0
            for pm in pass_mask_np:
                if pm:
                    e2_full.append(e2_err[idx_e2].numpy())
                    idx_e2 += 1
                else:
                    e2_full.append(0.0)
        else:
            e2_full = [0.0]*len(pass_mask)

        # final error + label
        for i in range(len(e1_err)):
            if not pass_mask[i]:  # e1_err <= threshold
                window_scores.append(e1_err[i].numpy())
                window_labels.append(0)
            else:
                # pass to e2
                if e2_full[i] > threshold_e2:
                    window_labels.append(1)
                else:
                    window_labels.append(0)
                window_scores.append(e2_full[i])

    # expand to time steps
    length = X_test.shape[0]
    M = len(window_scores)
    time_scores = np.zeros(length)
    counts = np.zeros(length)
    for i in range(M):
        start = i
        end   = i + timesteps - 1
        time_scores[start:end+1] += window_scores[i]
        counts[start:end+1] += 1

    counts[counts == 0] = 1
    time_scores /= counts
    time_preds = np.zeros(length, dtype=int)
    for i in range(M):
        if window_labels[i] == 1:
            start = i
            end   = i + timesteps - 1
            time_preds[start:end+1] = 1

    moe['final_scores'] = time_scores
    moe['final_preds']  = time_preds


###############################################################################
# 4) PLOTTING FUNCTIONS (EXPERT1, EXPERT2 ALONE, AND FINAL)
###############################################################################
# For brevity, we show only one example. We can replicate for expert2 and final.

def plot_expert1(moe, save_path=None):
    """
    Example function that feeds all test windows to expert1 (ignoring gating),
    uses threshold_e1, and plots the final time-series reconstruction and anomalies.
    """
    import plotly.graph_objects as go
    import os

    # Implementation is just like the "expert1 alone" approach we used before.
    # We'll skip for brevity.
    # ...
    print("[plot_expert1] Not fully implemented in this snippet - fill in if needed.")


def plot_final_moe(moe, save_path=None):
    """
    Plots the final gating results from moe['final_scores'], moe['final_preds'].
    """
    import plotly.graph_objects as go
    import numpy as np

    X_test = moe['test_data']
    labels = moe['labels']
    final_scores = moe.get('final_scores', None)
    final_preds  = moe.get('final_preds', None)
    timesteps    = moe['timesteps']

    if final_scores is None or final_preds is None:
        print("Please call evaluate_moe(...) first.")
        return

    # build some "combined" reconstruction if desired
    # or just plot final_scores + preds + test data
    # Example:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(X_test))),
        y=X_test.squeeze(),
        mode='lines',
        name='Test Data'
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(final_scores))),
        y=final_scores,
        mode='lines',
        name='MoE Anomaly Scores'
    ))
    # highlight anomalies
    anomaly_idxs = [i for i in range(len(final_preds)) if final_preds[i] == 1]
    fig.add_trace(go.Scatter(
        x=anomaly_idxs,
        y=[X_test[i] for i in anomaly_idxs],
        mode='markers',
        name='MoE Anomalies'
    ))

    # add label anomalies
    label_idxs = [i for i in range(len(labels)) if labels[i] == 1]
    fig.add_trace(go.Scatter(
        x=label_idxs,
        y=[X_test[i] for i in label_idxs],
        mode='markers',
        name='True Anomalies'
    ))

    fig.update_layout(title='MoE Final Gating Results')

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"Saved final MoE plot to {save_path}")

    fig.show()
