###############################################################################
# File: anomaly_detection/anomaly_models/mixture_of_experts/MoE.py
###############################################################################
import numpy as np
import tensorflow as tf
import os
import plotly.graph_objects as go
import math

###############################################################################
# MixtureOfExperts Class
###############################################################################
class MixtureOfExperts:
    """
    A class-based 2-expert Mixture of Experts with dynamic threshold gating:
      1) expert1 sees all windows
      2) if expert1 reconstruction error > threshold_e1, pass to expert2
      3) if expert2 reconstruction error > threshold_e2 => anomaly

    This class expects each "expert" to be instantiated from an ExpertClass
    we provide (e.g. LSTMAutoencoder). The ExpertClass must define:
      - .model (a tf.keras Model or similar)
      - .train_data_window, .test_data_window
      - and any needed properties for the shape, etc.

    By default, we consider 'mse' or 'max_diff_loss' for gating.
    """

    def __init__(
        self,
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
        # Additional keyword arguments passed to the ExpertClass constructor
        **expert_kwargs
    ):
        """
        :param ExpertClass: The class to instantiate for each expert (e.g. LSTMAutoencoder).
        :param train_data:  (N, features) array, float
        :param test_data:   (M, features) array, float
        :param labels:      (M,) or (M,1) array of integers
        :param timesteps:   int, window size
        :param features:    int, dimension of each time step
        :param threshold_sigma: float, controls dynamic threshold = mean + sigma * std
        :param step_size:   int, step size for sliding windows
        :param loss_type:   'mse' or 'max_diff_loss'
        :param seed:        random seed
        :param expert_kwargs: extra args passed to each ExpertClass
        """
        self.train_data = train_data.astype(np.float32)
        self.test_data  = test_data.astype(np.float32)
        self.labels     = labels.astype(int) if labels is not None else None

        self.timesteps        = timesteps
        self.features         = features
        self.threshold_sigma  = threshold_sigma
        self.step_size        = step_size
        self.loss_type        = loss_type
        self.seed             = seed
        self.expert_kwargs    = expert_kwargs

        # Instantiate two experts with identical config
        # Each ExpertClass should handle window creation or have .train_data_window, .test_data_window
        self.expert1 = ExpertClass(
            train_data=self.train_data,
            test_data=self.test_data,
            labels=self.labels,
            timesteps=self.timesteps,
            features=self.features,
            step_size=self.step_size,
            seed=self.seed,
            loss_type=self.loss_type,     # if our expert needs this
            **self.expert_kwargs
        )

        self.expert2 = ExpertClass(
            train_data=self.train_data,
            test_data=self.test_data,
            labels=self.labels,
            timesteps=self.timesteps,
            features=self.features,
            step_size=self.step_size,
            seed=self.seed+123,  # slightly different seed if we prefer
            loss_type=self.loss_type,
            **self.expert_kwargs
        )

        # Dynamic gating thresholds
        self.threshold_e1 = 0.0
        self.threshold_e2 = 0.0

        # Final gating results on test set
        self.final_scores = None
        self.final_preds  = None

    ############################################################################
    # Training with dynamic gating
    ############################################################################
    def train(self, epochs=50, batch_size=32, patience=10, optimizer='adam'):
        """
        Custom training loop:
          - For each batch in expert1.train_data_window:
              1) forward pass expert1 -> e1 errors
              2) if e1_error > threshold_e1 => pass sub-batch to expert2
              3) separate gradient updates
          - end of epoch, recompute threshold_e1 from entire train set
          - final step: compute threshold_e2 from windows that actually pass

        :param epochs:    number of epochs
        :param batch_size: mini-batch size
        :param patience:   early stopping
        :param optimizer:  e.g. 'adam'
        """
        # Quick references
        import tensorflow as tf
        from tensorflow.keras.optimizers import get as get_optimizer

        if self.expert1.train_data_window is None:
            print("No training windows found in expert1; skipping training.")
            return

        train_windows = self.expert1.train_data_window
        dataset = tf.data.Dataset.from_tensor_slices(train_windows).batch(batch_size)

        # We define local loss functions
        def loss_fn_mse(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_true - y_pred), axis=[1,2])  # shape (batch,)

        def loss_fn_max_diff(y_true, y_pred):
            return tf.reduce_max(tf.abs(y_true - y_pred), axis=[1,2])     # shape (batch,)

        opt1 = get_optimizer(optimizer)
        opt2 = get_optimizer(optimizer)

        best_loss = math.inf
        patience_counter = 0

        # Start with a negative threshold so everything passes to expert2 in epoch 0
        self.threshold_e1 = -999999

        for epoch in range(epochs):
            epoch_losses_e1 = []
            epoch_losses_e2 = []

            for x_batch in dataset:
                x_batch = tf.cast(x_batch, tf.float32)

                with tf.GradientTape(persistent=True) as tape:
                    # forward pass expert1
                    recon1 = self.expert1.model(x_batch, training=True)
                    if self.loss_type == 'mse':
                        e1_errors = loss_fn_mse(x_batch, recon1)
                        loss_e1   = tf.reduce_mean(e1_errors)
                    else:
                        e1_errors = loss_fn_max_diff(x_batch, recon1)
                        loss_e1   = tf.reduce_mean(e1_errors)

                    # gating => pass sub-batch to expert2
                    pass_mask  = tf.greater(e1_errors, self.threshold_e1)
                    x_pass_e2  = tf.boolean_mask(x_batch, pass_mask)

                    if tf.shape(x_pass_e2)[0] > 0:
                        recon2 = self.expert2.model(x_pass_e2, training=True)
                        if self.loss_type == 'mse':
                            e2_err  = loss_fn_mse(x_pass_e2, recon2)
                            loss_e2 = tf.reduce_mean(e2_err)
                        else:
                            e2_err  = loss_fn_max_diff(x_pass_e2, recon2)
                            loss_e2 = tf.reduce_mean(e2_err)
                    else:
                        loss_e2 = 0.0

                # separate grads
                grads1 = tape.gradient(loss_e1, self.expert1.model.trainable_weights)
                opt1.apply_gradients(zip(grads1, self.expert1.model.trainable_weights))

                if tf.shape(x_pass_e2)[0] > 0:
                    grads2 = tape.gradient(loss_e2, self.expert2.model.trainable_weights)
                    opt2.apply_gradients(zip(grads2, self.expert2.model.trainable_weights))

                del tape
                epoch_losses_e1.append(loss_e1.numpy())
                if tf.shape(x_pass_e2)[0] > 0:
                    epoch_losses_e2.append(loss_e2.numpy())
                else:
                    epoch_losses_e2.append(0.0)

            # End of epoch => recompute threshold_e1 from entire train set
            e1_all_err = []
            for x_batch_all in dataset:
                recon_all = self.expert1.model(x_batch_all, training=False)
                if self.loss_type == 'mse':
                    batch_e1 = loss_fn_mse(x_batch_all, recon_all)
                else:
                    batch_e1 = loss_fn_max_diff(x_batch_all, recon_all)
                e1_all_err.extend(batch_e1.numpy())
            e1_all_err = np.array(e1_all_err)
            mean_e1 = np.mean(e1_all_err)
            std_e1  = np.std(e1_all_err)
            self.threshold_e1 = mean_e1 + self.threshold_sigma * std_e1

            avg_e1 = np.mean(epoch_losses_e1)
            avg_e2 = np.mean(epoch_losses_e2)
            combined_loss = avg_e1 + avg_e2

            print(f"[Epoch {epoch+1}/{epochs}] E1_Loss={avg_e1:.4f}, E2_Loss={avg_e2:.4f}, threshold_e1={self.threshold_e1:.4f}")

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
        self._compute_threshold_e2()

    def _compute_threshold_e2(self):
        """
        Gather windows that pass e1 threshold, measure e2 errors, set threshold_e2.
        """
        if self.expert1.train_data_window is None:
            self.threshold_e2 = 9999999
            return

        dataset = tf.data.Dataset.from_tensor_slices(self.expert1.train_data_window).batch(32)

        def loss_fn_mse(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_true - y_pred), axis=[1,2])

        def loss_fn_max_diff(y_true, y_pred):
            return tf.reduce_max(tf.abs(y_true - y_pred), axis=[1,2])

        e2_errs = []
        for x_batch in dataset:
            recon1 = self.expert1.model(x_batch, training=False)
            if self.loss_type == 'mse':
                e1_err = loss_fn_mse(x_batch, recon1)
            else:
                e1_err = loss_fn_max_diff(x_batch, recon1)

            pass_mask  = tf.greater(e1_err, self.threshold_e1)
            x_pass_e2  = tf.boolean_mask(x_batch, pass_mask)
            if tf.shape(x_pass_e2)[0] > 0:
                recon2 = self.expert2.model(x_pass_e2, training=False)
                if self.loss_type == 'mse':
                    e2 = loss_fn_mse(x_pass_e2, recon2)
                else:
                    e2 = loss_fn_max_diff(x_pass_e2, recon2)
                e2_errs.extend(e2.numpy())

        if len(e2_errs) == 0:
            self.threshold_e2 = 9999999
        else:
            mean_e2 = np.mean(e2_errs)
            std_e2  = np.std(e2_errs)
            self.threshold_e2 = mean_e2 + self.threshold_sigma * std_e2

    ############################################################################
    # Evaluation (final gating) on test set
    ############################################################################
    def evaluate(self, batch_size=32):
        """
        Final gating-based evaluation:
          - If e1_err <= threshold_e1 => normal
          - else => pass to e2 => if e2_err > threshold_e2 => anomaly
        Expand window-level decisions to time steps => self.final_scores, self.final_preds.
        """
        if self.expert1.test_data_window is None:
            print("No test windows to evaluate.")
            self.final_scores = None
            self.final_preds  = None
            return

        test_windows = self.expert1.test_data_window
        ds_test = tf.data.Dataset.from_tensor_slices(test_windows).batch(batch_size)

        def loss_fn_mse(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_true - y_pred), axis=[1,2])

        def loss_fn_max_diff(y_true, y_pred):
            return tf.reduce_max(tf.abs(y_true - y_pred), axis=[1,2])

        window_scores = []
        window_labels = []

        for x_batch in ds_test:
            recon1 = self.expert1.model(x_batch, training=False)
            if self.loss_type == 'mse':
                e1_err = loss_fn_mse(x_batch, recon1)
            else:
                e1_err = loss_fn_max_diff(x_batch, recon1)

            pass_mask  = tf.greater(e1_err, self.threshold_e1)
            x_pass_e2  = tf.boolean_mask(x_batch, pass_mask)
            if tf.shape(x_pass_e2)[0] > 0:
                recon2 = self.expert2.model(x_pass_e2, training=False)
                if self.loss_type == 'mse':
                    e2_err = loss_fn_mse(x_pass_e2, recon2)
                else:
                    e2_err = loss_fn_max_diff(x_pass_e2, recon2)

                pass_mask_np = pass_mask.numpy()
                e2_full = []
                idx2 = 0
                for pm in pass_mask_np:
                    if pm:
                        e2_full.append(e2_err[idx2].numpy())
                        idx2 += 1
                    else:
                        e2_full.append(0.0)
            else:
                e2_full = [0.0]*len(pass_mask)

            # build final error + label
            for i in range(len(e1_err)):
                if not pass_mask[i]:
                    # e1_err <= threshold => normal
                    window_scores.append(e1_err[i].numpy())
                    window_labels.append(0)
                else:
                    # pass to e2
                    if e2_full[i] > self.threshold_e2:
                        window_labels.append(1)
                    else:
                        window_labels.append(0)
                    window_scores.append(e2_full[i])

        # expand to time steps
        length = self.test_data.shape[0]
        M = len(window_scores)
        time_scores = np.zeros(length)
        counts = np.zeros(length)

        for i in range(M):
            start = i
            end   = i + self.timesteps - 1
            time_scores[start:end+1] += window_scores[i]
            counts[start:end+1] += 1

        counts[counts == 0] = 1
        time_scores /= counts

        time_preds = np.zeros(length, dtype=int)
        for i in range(M):
            if window_labels[i] == 1:
                start = i
                end   = i + self.timesteps - 1
                time_preds[start:end+1] = 1

        self.final_scores = time_scores
        self.final_preds  = time_preds

    ############################################################################
    # Plot Expert1 alone (no gating)
    ############################################################################
    def plot_expert1(self, save_path=None, file_format='html', size=800):
        """
        Feeds ALL test windows to expert1 alone (ignoring gating).
        Uses self.threshold_e1 as the anomaly cutoff at the time-step level.
        """
        if self.expert1.test_data_window is None or self.labels is None:
            print("No test windows or labels to plot.")
            return

        # 1) Reconstruct with expert1
        e1_recon = self.expert1.model.predict(self.expert1.test_data_window)
        # 2) compute window errors
        if self.loss_type == 'mse':
            window_errors = np.mean(np.square(self.expert1.test_data_window - e1_recon), axis=(1,2))
        else:
            window_errors = np.max(np.abs(self.expert1.test_data_window - e1_recon), axis=(1,2))

        # expand to time steps
        length = self.test_data.shape[0]
        M = len(window_errors)
        time_scores = np.zeros(length)
        counts      = np.zeros(length)
        for i in range(M):
            start = i
            end   = i + self.timesteps - 1
            time_scores[start:end+1] += window_errors[i]
            counts[start:end+1] += 1
        counts[counts==0] = 1
        time_scores /= counts

        time_preds = (time_scores > self.threshold_e1).astype(int)

        # reconstruct predictions (averaged) just like the LSTMAE approach
        time_recon = np.zeros((length, self.features))
        counts_pred = np.zeros(length)
        for i in range(M):
            for j in range(self.timesteps):
                idx = i+j
                if idx < length:
                    time_recon[idx] += e1_recon[i, j]
                    counts_pred[idx] += 1
        for i in range(length):
            if counts_pred[i] > 0:
                time_recon[i] /= counts_pred[i]

        # flatten if univariate
        if self.features == 1:
            time_recon = time_recon.ravel()

        # plot
        test_data = self.test_data.ravel()
        labels    = self.labels.ravel()
        plot_width = max(size, len(test_data)//10)

        fig = go.Figure()
        # Test Data
        fig.add_trace(go.Scatter(x=list(range(len(test_data))),
                                 y=test_data,
                                 mode='lines',
                                 name='Test Data',
                                 line=dict(color='blue')))
        # Expert1 Recon
        fig.add_trace(go.Scatter(x=list(range(len(time_recon))),
                                 y=time_recon,
                                 mode='lines',
                                 name='Expert1 Predictions',
                                 line=dict(color='purple')))
        # E1 Errors
        fig.add_trace(go.Scatter(x=list(range(len(time_scores))),
                                 y=time_scores,
                                 mode='lines',
                                 name='Expert1 Errors',
                                 line=dict(color='red')))

        # Labeled anomalies
        label_indices = [i for i in range(len(labels)) if labels[i] == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[test_data[i] for i in label_indices],
                                     mode='markers',
                                     name='True Anomalies',
                                     marker=dict(color='orange', size=10)))

        # Pred anomalies
        pred_indices = [i for i in range(len(time_preds)) if time_preds[i] == 1]
        if pred_indices:
            fig.add_trace(go.Scatter(x=pred_indices,
                                     y=[time_recon[i] for i in pred_indices],
                                     mode='markers',
                                     name='Expert1 Anomalies',
                                     marker=dict(color='green', size=10)))

        fig.update_layout(title='Expert1 Alone Results',
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'),
                          template='plotly',
                          width=plot_width)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if file_format.lower() == 'html':
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, format=file_format)
            print(f"[Expert1] Plot saved to {save_path}")

        fig.show()

    ############################################################################
    # Plot Expert2 alone (no gating)
    ############################################################################
    def plot_expert2(self, save_path=None, file_format='html', size=800):
        """
        Feeds ALL test windows to expert2 alone (ignoring gating).
        Uses self.threshold_e2 as the anomaly cutoff at the time-step level.
        """
        if self.expert2.test_data_window is None or self.labels is None:
            print("No test windows or labels to plot.")
            return

        e2_recon = self.expert2.model.predict(self.expert2.test_data_window)
        if self.loss_type == 'mse':
            window_errors = np.mean(np.square(self.expert2.test_data_window - e2_recon), axis=(1,2))
        else:
            window_errors = np.max(np.abs(self.expert2.test_data_window - e2_recon), axis=(1,2))

        length = self.test_data.shape[0]
        M = len(window_errors)
        time_scores = np.zeros(length)
        counts      = np.zeros(length)
        for i in range(M):
            start = i
            end   = i + self.timesteps - 1
            time_scores[start:end+1] += window_errors[i]
            counts[start:end+1] += 1
        counts[counts==0] = 1
        time_scores /= counts

        time_preds = (time_scores > self.threshold_e2).astype(int)

        # reconstruct predictions
        time_recon = np.zeros((length, self.features))
        counts_pred = np.zeros(length)
        for i in range(M):
            for j in range(self.timesteps):
                idx = i + j
                if idx < length:
                    time_recon[idx] += e2_recon[i, j]
                    counts_pred[idx] += 1
        for i in range(length):
            if counts_pred[i] > 0:
                time_recon[i] /= counts_pred[i]

        if self.features == 1:
            time_recon = time_recon.ravel()

        test_data = self.test_data.ravel()
        labels    = self.labels.ravel()
        plot_width = max(size, len(test_data)//10)

        fig = go.Figure()
        # Test Data
        fig.add_trace(go.Scatter(x=list(range(len(test_data))),
                                 y=test_data,
                                 mode='lines',
                                 name='Test Data',
                                 line=dict(color='blue')))
        # Expert2 Recon
        fig.add_trace(go.Scatter(x=list(range(len(time_recon))),
                                 y=time_recon,
                                 mode='lines',
                                 name='Expert2 Predictions',
                                 line=dict(color='purple')))
        # E2 Errors
        fig.add_trace(go.Scatter(x=list(range(len(time_scores))),
                                 y=time_scores,
                                 mode='lines',
                                 name='Expert2 Errors',
                                 line=dict(color='red')))

        # Labeled anomalies
        label_indices = [i for i in range(len(labels)) if labels[i] == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[test_data[i] for i in label_indices],
                                     mode='markers',
                                     name='True Anomalies',
                                     marker=dict(color='orange', size=10)))

        # Pred anomalies
        pred_indices = [i for i in range(len(time_preds)) if time_preds[i] == 1]
        if pred_indices:
            fig.add_trace(go.Scatter(x=pred_indices,
                                     y=[time_recon[i] for i in pred_indices],
                                     mode='markers',
                                     name='Expert2 Anomalies',
                                     marker=dict(color='green', size=10)))

        fig.update_layout(title='Expert2 Alone Results',
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'),
                          template='plotly',
                          width=plot_width)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if file_format.lower() == 'html':
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, format=file_format)
            print(f"[Expert2] Plot saved to {save_path}")

        fig.show()

    ############################################################################
    # Plot final MoE gating results
    ############################################################################
    def plot_final_moe(self, save_path=None, file_format='html', size=800):
        """
        Plots the final gating-based reconstruction after calling self.evaluate().
        We show self.final_scores (time-level) and self.final_preds alongside the true data.
        """
        if self.final_scores is None or self.final_preds is None:
            print("Please run .evaluate() first.")
            return
        if self.labels is None:
            print("No labels provided for final plot.")
            return

        time_scores = self.final_scores
        time_preds  = self.final_preds
        test_data   = self.test_data.ravel()
        labels      = self.labels.ravel()

        plot_width = max(size, len(test_data)//10)

        fig = go.Figure()
        # Test Data
        fig.add_trace(go.Scatter(
            x=list(range(len(test_data))),
            y=test_data,
            mode='lines',
            name='Test Data',
            line=dict(color='blue')
        ))
        # MoE final anomaly scores
        fig.add_trace(go.Scatter(
            x=list(range(len(time_scores))),
            y=time_scores,
            mode='lines',
            name='MoE Anomaly Scores',
            line=dict(color='red')
        ))
        # Labeled anomalies
        label_idxs = [i for i in range(len(labels)) if labels[i] == 1]
        if label_idxs:
            fig.add_trace(go.Scatter(
                x=label_idxs,
                y=[test_data[i] for i in label_idxs],
                mode='markers',
                name='True Anomalies',
                marker=dict(color='orange', size=10)
            ))
        # Predicted anomalies
        anomaly_idxs = [i for i in range(len(time_preds)) if time_preds[i] == 1]
        if anomaly_idxs:
            fig.add_trace(go.Scatter(
                x=anomaly_idxs,
                y=[test_data[i] for i in anomaly_idxs],
                mode='markers',
                name='MoE Anomalies',
                marker=dict(color='green', size=10)
            ))

        fig.update_layout(title='MoE Final Gating Results',
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'),
                          template='plotly',
                          width=plot_width)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            if file_format.lower() == 'html':
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, format=file_format)
            print(f"[MoE Final] Plot saved to {save_path}")

        fig.show()

    ############################################################################
    # Save/Load experts
    ############################################################################
    def save_models(self, dir_path="moe_models"):
        """
        Saves the expert1 and expert2 Keras models in H5 format.
        """
        os.makedirs(dir_path, exist_ok=True)
        self.expert1.model.save(os.path.join(dir_path, "expert1.h5"))
        self.expert2.model.save(os.path.join(dir_path, "expert2.h5"))
        print(f"Saved MoE experts to {dir_path}/expert1.h5 and expert2.h5")

    def load_models(self, dir_path, train_data=None, test_data=None, label_data=None):
        """
        Loads expert1 and expert2 from H5 files in dir_path.
        Optionally, reset train/test data if provided.
        """
        from tensorflow.keras.models import load_model

        expert1_path = os.path.join(dir_path, "expert1.h5")
        expert2_path = os.path.join(dir_path, "expert2.h5")

        self.expert1.model = load_model(expert1_path, compile=False)
        self.expert2.model = load_model(expert2_path, compile=False)
        self.expert1.model.compile(optimizer='adam', loss='mse')  # or as needed
        self.expert2.model.compile(optimizer='adam', loss='mse')

        if train_data is not None:
            self.train_data = train_data
        if test_data is not None:
            self.test_data  = test_data
        if label_data is not None:
            self.labels     = label_data

        print(f"Loaded experts from {expert1_path}, {expert2_path}.")
