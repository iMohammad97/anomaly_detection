import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm.notebook import trange, tqdm
import plotly.graph_objects as go
import math


class TorchMoE:
    """
    A 2-expert Mixture-of-Experts for Torch models with dynamic threshold gating.

    The class instantiates two copies (expert1 and expert2) of a given ExpertClass
    (e.g. Twin). During training, every batch is fed through expert1;
    if a window’s reconstruction error exceeds a dynamic threshold (threshold_e1),
    that window is also passed to expert2 and updated separately.

    At the end of training, a separate threshold_e2 is computed from those windows
    that passed expert1. In evaluation, each window’s final error is chosen as:
       - expert1 error if not passed, or
       - expert2 error if passed.
    A window is predicted anomalous if its final error exceeds threshold_e2.

    The class provides plotting methods for:
      • Expert1 alone (ignoring gating),
      • Expert2 alone,
      • The final MoE (gated) results.

    It also supports saving and loading the experts’ state.
    """

    def __init__(
            self,
            ExpertClass,
            window_size=256,
            device='cpu',
            threshold_sigma=2.0,
            seed=0,
            **expert_kwargs
    ):
        """
        :param ExpertClass: Torch model class (e.g. Twin)
        :param window_size: Sliding window size.
        :param device: 'cpu' or 'cuda'
        :param threshold_sigma: Dynamic threshold = mean + sigma * std.
        :param seed: Random seed.
        :param expert_kwargs: Extra parameters passed to ExpertClass.
        """
        torch.manual_seed(seed)
        self.seed = seed
        self.device = device
        self.window_size = window_size
        self.threshold_sigma = threshold_sigma

        # Instantiate two experts (they are assumed to have attributes:
        # .model, .train_data_window, .test_data_window, .optimizer, and methods forward() and stationary_loss())
        self.expert1 = ExpertClass(window_size=self.window_size, device=self.device, seed=self.seed, **expert_kwargs)
        self.expert2 = ExpertClass(window_size=self.window_size, device=self.device, seed=self.seed + 123,
                                   **expert_kwargs)

        # Gating thresholds (will be updated during training)
        self.threshold_e1 = 0.0
        self.threshold_e2 = 0.0

        # Final evaluation results (per window)
        self.final_errors = None
        self.final_preds = None

        # For evaluation/plotting, these will be computed from the data loader;
        # they are expected to be computed per window (not per time step)
        self.loss_name = None

    ###########################################################################
    # TRAINING
    ###########################################################################
    def train(self, train_loader, n_epochs=50, loss_name='MaxDiff'):
        """
        Custom training loop with gating:
          - For each batch from train_loader:
              * Forward pass through expert1.
              * Compute reconstruction error (e1_err) using the loss function from select_loss.
              * If e1_err > threshold_e1, pass those windows to expert2.
              * Compute a "stationary loss" (from each expert’s stationary_loss() method)
                and update expert1 and expert2 separately.
          - End of each epoch, update threshold_e1 = mean(e1_errors) + sigma * std(e1_errors).
          - After training, compute threshold_e2 from windows that passed expert1.
        :param train_loader: DataLoader returning (data, anomaly) pairs.
        :param n_epochs: Number of epochs.
        :param loss_name: Loss key string (e.g. "MaxDiff" or "MSE").
        """
        self.loss_name = loss_name
        # For the first epoch, set threshold_e1 very low so that all windows pass to expert2.
        self.threshold_e1 = -9999999.0

        # Get the reconstruction loss function from expert1; note: for MaxDiff it returns a lambda (do not call .to(device))
        recon_loss_func = self.expert1.select_loss(loss_name)

        best_combined = math.inf
        patience = 10
        patience_counter = 0

        # Assume expert1.train_data_window exists (set by your ExpertClass)
        # Here we use the provided train_loader (which yields batches of windows).
        dataset = train_loader  # train_loader is assumed to yield (data_batch, anomaly)

        for epoch_i in trange(n_epochs, desc="MoE Training"):
            e1_batch_losses = []
            e2_batch_losses = []
            e1_errors_all = []

            for data_batch, _ in tqdm(dataset, leave=False):
                data_batch = data_batch.to(self.device)

                # ----- Expert1 forward pass -----
                self.expert1.optimizer.zero_grad()
                latent1, recon1 = self.expert1.forward(data_batch)

                # Compute reconstruction error per sample:
                if loss_name == 'MaxDiff':
                    e1_err = (recon1 - data_batch).abs().max(dim=2)[0].max(dim=1)[0]
                    loss_e1 = torch.mean(e1_err)
                else:
                    e1_err = torch.mean((recon1 - data_batch) ** 2, dim=(1, 2))
                    loss_e1 = torch.mean(e1_err)

                # Get stationary loss from expert1
                sl, _, _ = self.expert1.stationary_loss(latent1, per_batch=False)
                loss_e1_total = loss_e1 + sl

                loss_e1_total.backward()
                self.expert1.optimizer.step()
                e1_batch_losses.append(loss_e1_total.item())
                e1_errors_all.extend(e1_err.detach().cpu().numpy())

                # ----- Gating: Pass sub-batch to Expert2 if e1_err > threshold_e1 -----
                pass_mask = (e1_err > self.threshold_e1).detach().cpu().numpy()
                pass_indices = np.where(pass_mask)[0]
                if len(pass_indices) > 0:
                    sub_batch = data_batch[pass_indices]
                    self.expert2.optimizer.zero_grad()
                    latent2, recon2 = self.expert2.forward(sub_batch)
                    if loss_name == 'MaxDiff':
                        e2_err = (recon2 - sub_batch).abs().max(dim=2)[0].max(dim=1)[0]
                        loss_e2 = torch.mean(e2_err)
                    else:
                        e2_err = torch.mean((recon2 - sub_batch) ** 2, dim=(1, 2))
                        loss_e2 = torch.mean(e2_err)
                    sl2, _, _ = self.expert2.stationary_loss(latent2, per_batch=False)
                    loss_e2_total = loss_e2 + sl2

                    loss_e2_total.backward()
                    self.expert2.optimizer.step()
                    e2_batch_losses.append(loss_e2_total.item())
                else:
                    e2_batch_losses.append(0.0)

            # End of epoch: update threshold_e1 from all e1 errors
            e1_errors_all = np.array(e1_errors_all)
            if len(e1_errors_all) > 0:
                self.threshold_e1 = e1_errors_all.mean() + self.threshold_sigma * e1_errors_all.std()
            else:
                self.threshold_e1 = 9999999.0

            epoch_e1 = np.mean(e1_batch_losses)
            epoch_e2 = np.mean(e2_batch_losses)
            combined = epoch_e1 + epoch_e2

            print(
                f"[Epoch {epoch_i + 1}/{n_epochs}] e1_loss={epoch_e1:.4f}, e2_loss={epoch_e2:.4f}, threshold_e1={self.threshold_e1:.4f}")

            if combined < best_combined:
                best_combined = combined
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Finally, compute threshold_e2 using the training set:
        self._compute_threshold_e2(train_loader, recon_loss_func)

    def _compute_threshold_e2(self, train_loader, recon_loss_func):
        """
        Computes threshold_e2 from all training windows that pass expert1's threshold.
        """
        e2_error_list = []
        for data_batch, _ in train_loader:
            data_batch = data_batch.to(self.device)
            latent1, recon1 = self.expert1.forward(data_batch)
            if self.loss_name == 'MaxDiff':
                e1_err = (recon1 - data_batch).abs().max(dim=2)[0].max(dim=1)[0]
            else:
                e1_err = torch.mean((recon1 - data_batch) ** 2, dim=(1, 2))
            pass_mask = (e1_err > self.threshold_e1).cpu().numpy()
            pass_indices = np.where(pass_mask)[0]
            if len(pass_indices) == 0:
                continue
            sub_batch = data_batch[pass_indices]
            latent2, recon2 = self.expert2.forward(sub_batch)
            if self.loss_name == 'MaxDiff':
                e2_sub_vec = (recon2 - sub_batch).abs().max(dim=2)[0].max(dim=1)[0]
            else:
                e2_sub_vec = torch.mean((recon2 - sub_batch) ** 2, dim=(1, 2))
            e2_error_list.extend(e2_sub_vec.detach().cpu().numpy())
        if len(e2_error_list) == 0:
            self.threshold_e2 = 9999999.0
        else:
            arr = np.array(e2_error_list)
            self.threshold_e2 = arr.mean() + self.threshold_sigma * arr.std()
        print(f"[Gating] threshold_e2 = {self.threshold_e2:.4f}")

    ###########################################################################
    # EVALUATION
    ###########################################################################
    def evaluate(self, data_loader, window_coef=0.2):
        """
        Evaluates the final gating on the dataset from data_loader.
        For each window:
          - Compute expert1 reconstruction error.
          - If error > threshold_e1, compute expert2 error.
          - Final error = expert1 error (if not passed) or expert2 error (if passed).
          - Final prediction: 1 if final error > threshold_e2, else 0.
        Also collects:
          - inputs: last time-step of each window from data.
          - outputs: last time-step of expert1 reconstruction.
          - anomalies: last time-step label of each window.
        Returns a dict with keys: 'inputs', 'outputs', 'errors', 'predictions', 'anomalies'.
        """
        self.eval()
        inputs_list = []
        outputs_list = []
        errors_list = []
        pred_list = []
        anomaly_list = []

        # Define loss function for reconstruction error per window:
        if self.loss_name == 'MaxDiff':
            def loss_fn(x, y):
                return torch.max(torch.abs(x - y), dim=2)[0].max(dim=1)[0]
        else:
            def loss_fn(x, y):
                return torch.mean((x - y) ** 2, dim=(1, 2))

        with torch.no_grad():
            for data_batch, anomalies in data_loader:
                data_batch = data_batch.to(self.device)
                latent, recon1 = self.expert1.forward(data_batch)
                if self.loss_name == 'MaxDiff':
                    e1_err = (recon1 - data_batch).abs().max(dim=2)[0].max(dim=1)[0]
                else:
                    e1_err = torch.mean((recon1 - data_batch) ** 2, dim=(1, 2))
                pass_mask = (e1_err > self.threshold_e1)
                e2_err = torch.zeros_like(e1_err)
                if pass_mask.sum() > 0:
                    sub_batch = data_batch[pass_mask]
                    latent2, recon2 = self.expert2.forward(sub_batch)
                    if self.loss_name == 'MaxDiff':
                        e2_err_sub = (recon2 - sub_batch).abs().max(dim=2)[0].max(dim=1)[0]
                    else:
                        e2_err_sub = torch.mean((recon2 - sub_batch) ** 2, dim=(1, 2))
                    e2_err[pass_mask] = e2_err_sub

                final_err = torch.where(pass_mask, e2_err, e1_err)
                final_pred = (final_err > self.threshold_e2).long()

                # For plotting, take the last time-step of each window (assume data shape (B, window_size, features))
                inputs_list.append(data_batch.cpu().numpy()[:, -1, :])
                outputs_list.append(recon1.cpu().numpy()[:, -1, :])
                errors_list.append(final_err.cpu().numpy())
                pred_list.append(final_pred.cpu().numpy())
                # Process anomalies: assume anomalies is (B, window_size) and take last time-step.
                a_np = anomalies.cpu().numpy()
                if a_np.ndim == 2 and a_np.shape[1] == self.window_size:
                    a_np = a_np[:, -1]
                anomaly_list.append(a_np)

        results = {}
        results['inputs'] = np.concatenate(inputs_list, axis=0).squeeze()
        results['outputs'] = np.concatenate(outputs_list, axis=0).squeeze()
        results['errors'] = np.concatenate(errors_list, axis=0).ravel()
        results['predictions'] = np.concatenate(pred_list, axis=0).ravel()
        results['anomalies'] = np.concatenate(anomaly_list, axis=0).ravel()
        self.final_errors = results['errors']
        self.final_preds = results['predictions']
        return results

    ###########################################################################
    # PLOTTING METHODS
    ###########################################################################
    def plot_expert1(self, data_loader, train=False, plot_width=800):
        """
        Plots Expert1's reconstruction results on data from data_loader (ignoring gating).
        Uses the last time-step of each window.
        """
        with torch.no_grad():
            inputs_list = []
            outputs_list = []
            errors_list = []
            anomaly_list = []
            for data_batch, anomalies in data_loader:
                data_batch = data_batch.to(self.device)
                latent, recon = self.expert1.forward(data_batch)
                if self.loss_name == 'MaxDiff':
                    err_vec = (recon - data_batch).abs().max(dim=2)[0].max(dim=1)[0].cpu().numpy()
                else:
                    err_vec = torch.mean((recon - data_batch) ** 2, dim=(1, 2)).cpu().numpy()
                inputs_list.append(data_batch.cpu().numpy()[:, -1, :])
                outputs_list.append(recon.cpu().numpy()[:, -1, :])
                errors_list.append(err_vec)
                a_np = anomalies.cpu().numpy()
                if a_np.ndim == 2 and a_np.shape[1] == self.window_size:
                    a_np = a_np[:, -1]
                anomaly_list.append(a_np)

        inputs = np.concatenate(inputs_list, axis=0).squeeze()
        outputs = np.concatenate(outputs_list, axis=0).squeeze()
        errors = np.concatenate(errors_list, axis=0).ravel()
        anomalies = np.concatenate(anomaly_list, axis=0).ravel()

        if train and self.threshold_e1 < 0:
            self.threshold_e1 = errors.mean() + 3 * errors.std()
        preds = [1 if e > self.threshold_e1 else 0 for e in errors]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(inputs))),
                                 y=inputs,
                                 mode='lines',
                                 name='Data',
                                 line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=list(range(len(outputs))),
                                 y=outputs,
                                 mode='lines',
                                 name='Expert1 Recon',
                                 line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=list(range(len(errors))),
                                 y=errors,
                                 mode='lines',
                                 name='Expert1 Errors',
                                 line=dict(color='red')))
        label_indices = [i for i in range(len(anomalies)) if anomalies[i] == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[inputs[i] for i in label_indices],
                                     mode='markers',
                                     name='Labels',
                                     marker=dict(color='orange', size=7)))
        if (not train) and (self.threshold_e1 is not None):
            fig.add_hline(y=self.threshold_e1, line_dash='dash', name='Threshold E1')
            pred_indices = [i for i in range(len(preds)) if preds[i] == 1]
            if pred_indices:
                fig.add_trace(go.Scatter(x=pred_indices,
                                         y=[inputs[i] for i in pred_indices],
                                         mode='markers',
                                         name='Predicted Anomalies',
                                         marker=dict(color='black', size=7, symbol='x')))
        fig.update_layout(title='Expert1 Alone Results',
                          xaxis_title='Window Index',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'),
                          template='plotly',
                          width=plot_width)
        fig.show()

    def plot_expert2(self, data_loader, train=False, plot_width=800):
        """
        Plots Expert2's reconstruction results on data from data_loader (ignoring gating).
        Uses the last time-step of each window.
        """
        with torch.no_grad():
            inputs_list = []
            outputs_list = []
            errors_list = []
            anomaly_list = []
            for data_batch, anomalies in data_loader:
                data_batch = data_batch.to(self.device)
                latent, recon = self.expert2.forward(data_batch)
                if self.loss_name == 'MaxDiff':
                    err_vec = (recon - data_batch).abs().max(dim=2)[0].max(dim=1)[0].cpu().numpy()
                else:
                    err_vec = torch.mean((recon - data_batch) ** 2, dim=(1, 2)).cpu().numpy()
                inputs_list.append(data_batch.cpu().numpy()[:, -1, :])
                outputs_list.append(recon.cpu().numpy()[:, -1, :])
                errors_list.append(err_vec)
                a_np = anomalies.cpu().numpy()
                if a_np.ndim == 2 and a_np.shape[1] == self.window_size:
                    a_np = a_np[:, -1]
                anomaly_list.append(a_np)

        inputs = np.concatenate(inputs_list, axis=0).squeeze()
        outputs = np.concatenate(outputs_list, axis=0).squeeze()
        errors = np.concatenate(errors_list, axis=0).ravel()
        anomalies = np.concatenate(anomaly_list, axis=0).ravel()

        if train and self.threshold_e2 < 0.1:
            self.threshold_e2 = errors.mean() + 3 * errors.std()
        preds = [1 if e > self.threshold_e2 else 0 for e in errors]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(inputs))),
                                 y=inputs,
                                 mode='lines',
                                 name='Data',
                                 line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=list(range(len(outputs))),
                                 y=outputs,
                                 mode='lines',
                                 name='Expert2 Recon',
                                 line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=list(range(len(errors))),
                                 y=errors,
                                 mode='lines',
                                 name='Expert2 Errors',
                                 line=dict(color='red')))
        label_indices = [i for i in range(len(anomalies)) if anomalies[i] == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[inputs[i] for i in label_indices],
                                     mode='markers',
                                     name='Labels',
                                     marker=dict(color='orange', size=7)))
        if (not train) and (self.threshold_e2 is not None):
            fig.add_hline(y=self.threshold_e2, line_dash='dash', name='Threshold E2')
            pred_indices = [i for i in range(len(preds)) if preds[i] == 1]
            if pred_indices:
                fig.add_trace(go.Scatter(x=pred_indices,
                                         y=[inputs[i] for i in pred_indices],
                                         mode='markers',
                                         name='Predicted Anomalies',
                                         marker=dict(color='black', size=7, symbol='x')))
        fig.update_layout(title='Expert2 Alone Results',
                          xaxis_title='Window Index',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'),
                          template='plotly',
                          width=plot_width)
        fig.show()

    def plot_final_moe(self, data_loader, plot_width=800):
        """
        Plots final MoE gating results using evaluate() on data_loader.
        Ensures that anomalies are reduced to one label per window.
        """
        results = self.evaluate(data_loader)
        inputs = np.array(results['inputs']).squeeze()
        outputs = np.array(results['outputs']).squeeze()
        errors = np.array(results['errors']).ravel()
        preds = np.array(results['predictions']).ravel()
        anomalies = np.array(results['anomalies'])

        # Ensure anomalies is 1D: if anomalies is 2D with shape (N, window_size), take last column.
        if anomalies.ndim == 2 and anomalies.shape[1] == self.window_size:
            anomalies = anomalies[:, -1]
        anomalies = anomalies.ravel()

        # Build indices for true and predicted anomalies
        label_indices = [i for i in range(len(anomalies)) if anomalies[i] == 1]
        pred_indices = [i for i in range(len(preds)) if preds[i] == 1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(inputs))),
            y=inputs,
            mode='lines',
            name='Data',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(outputs))),
            y=outputs,
            mode='lines',
            name='MoE Recon (Expert1)',
            line=dict(color='purple')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(errors))),
            y=errors,
            mode='lines',
            name='MoE Final Errors',
            line=dict(color='red')
        ))
        if label_indices:
            fig.add_trace(go.Scatter(
                x=label_indices,
                y=[inputs[i] for i in label_indices],
                mode='markers',
                name='True Anomalies',
                marker=dict(color='orange', size=7)
            ))
        if pred_indices:
            fig.add_trace(go.Scatter(
                x=pred_indices,
                y=[inputs[i] for i in pred_indices],
                mode='markers',
                name='MoE Predicted Anomalies',
                marker=dict(color='black', size=7, symbol='x')
            ))
        fig.update_layout(
            title='Final MoE Gating Results',
            xaxis_title='Window Index',
            yaxis_title='Value',
            legend=dict(x=0, y=1, orientation='h'),
            template='plotly',
            width=plot_width
        )
        fig.show()

    ###########################################################################
    # SAVE / LOAD
    ###########################################################################
    def save(self, dir_path='torch_moe'):
        os.makedirs(dir_path, exist_ok=True)
        e1_path = os.path.join(dir_path, 'expert1.pth')
        e2_path = os.path.join(dir_path, 'expert2.pth')
        torch.save(self.expert1.state_dict(), e1_path)
        torch.save(self.expert2.state_dict(), e2_path)
        meta_path = os.path.join(dir_path, 'moe_meta.npz')
        np.savez(meta_path,
                 threshold_e1=self.threshold_e1,
                 threshold_e2=self.threshold_e2,
                 device=self.device,
                 window_size=self.window_size,
                 threshold_sigma=self.threshold_sigma,
                 seed=self.seed,
                 loss_name=self.loss_name)
        print(f"MoE saved to {dir_path}")

    def load(self, dir_path='torch_moe'):
        e1_path = os.path.join(dir_path, 'expert1.pth')
        e2_path = os.path.join(dir_path, 'expert2.pth')
        meta_path = os.path.join(dir_path, 'moe_meta.npz')
        self.expert1.load_state_dict(torch.load(e1_path))
        self.expert2.load_state_dict(torch.load(e2_path))
        meta = np.load(meta_path)
        self.threshold_e1 = float(meta['threshold_e1'])
        self.threshold_e2 = float(meta['threshold_e2'])
        self.device = str(meta['device'])
        self.window_size = int(meta['window_size'])
        self.threshold_sigma = float(meta['threshold_sigma'])
        self.seed = int(meta['seed'])
        self.loss_name = str(meta['loss_name'])
        print(f"MoE loaded from {dir_path}. thresholds=({self.threshold_e1:.4f}, {self.threshold_e2:.4f})")
