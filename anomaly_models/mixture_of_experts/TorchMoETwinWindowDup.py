import torch
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm, trange
import plotly.graph_objects as go
import os
import math


class TorchMoETwinDup:
    """
    A Torch-based Mixture of Experts (MoE) for the Twin model.

    - Expert1 processes all windows.
    - For each window, if Expert1's reconstruction error is above a dynamic threshold,
      that window is duplicated 'duplication_factor' times and passed to Expert2.
    - Each expert is an instance of the Twin model.
    - The class provides training with dynamic threshold updates and plotting functions.
    """

    def __init__(self,
                 TwinClass,
                 window_size=256,
                 device='cpu',
                 n_features=1,
                 latent_dim=32,
                 seed=0,
                 threshold_sigma=3.0,
                 duplication_factor=2,
                 **twin_kwargs):
        """
        :param TwinClass: The model class (e.g. Twin) to use as expert.
        :param window_size: Window size.
        :param device: 'cpu' or 'cuda'.
        :param n_features: Number of features (default 1).
        :param latent_dim: Latent dimension.
        :param seed: Random seed.
        :param threshold_sigma: Multiplier for the dynamic threshold.
        :param duplication_factor: Number of times to duplicate windows that pass gating.
        :param twin_kwargs: Additional keyword arguments for Twin.
        """
        self.device = device
        self.window_size = window_size
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.threshold_sigma = threshold_sigma
        self.duplication_factor = duplication_factor
        self.seed = seed

        # Instantiate two experts
        self.expert1 = TwinClass(n_features=n_features,
                                 window_size=window_size,
                                 latent_dim=latent_dim,
                                 device=device,
                                 seed=seed,
                                 **twin_kwargs)
        self.expert2 = TwinClass(n_features=n_features,
                                 window_size=window_size,
                                 latent_dim=latent_dim,
                                 device=device,
                                 seed=seed + 123,
                                 **twin_kwargs)

        # Dynamic thresholds (initialized low so that initially all windows pass)
        self.threshold_e1 = -999999.0
        self.threshold_e2 = None

        # For logging training progress
        self.losses_e1 = []
        self.losses_e2 = []
        self.thresholds_e1 = []
        self.thresholds_e2 = None

    ################################################################################
    # Training function with window-level gating and duplication
    ################################################################################
    def train(self, train_loader, n_epochs=50, seed=42, loss_name1='MSE_R2', loss_name2='MaxDiff'):
        """
        Trains the MoE by processing each window in the batch. For each window, if the
        reconstruction error from expert1 is above the dynamic threshold (threshold_e1),
        the window is duplicated duplication_factor times and passed to expert2.

        The reconstruction loss is defined as:
            recon = loss1(x, d) + loss2(x[:, -latent_dim:], d[:, -latent_dim:])
        plus the stationary loss (mean + std penalty).

        :param train_loader: DataLoader for training windows.
        :param n_epochs: Number of epochs.
        :param seed: Random seed.
        :param loss_name1: Name of the first loss function.
        :param loss_name2: Name of the second loss function.
        """
        torch.manual_seed(seed)
        self.expert1.train()
        self.expert2.train()

        # Get reconstruction loss functions from each expert
        recon_loss1_e1 = self.expert1.select_loss(loss_name1)
        recon_loss2_e1 = self.expert1.select_loss(loss_name2)
        recon_loss1_e2 = self.expert2.select_loss(loss_name1)
        recon_loss2_e2 = self.expert2.select_loss(loss_name2)

        # We'll use a standard MSE loss (reduction='none') for gating computations
        mse_full = torch.nn.MSELoss(reduction='none').to(self.device)
        window_coef = 0.2  # as in Twin

        best_loss = math.inf
        patience_counter = 0

        for epoch in trange(n_epochs, desc="MoE Training"):
            e1_errors_all = []
            e2_errors_all = []

            e1_recon_list = []
            e1_mean_list = []
            e1_std_list = []
            e2_recon_list = []
            e2_mean_list = []
            e2_std_list = []

            for d, a in tqdm(train_loader, leave=False):
                d = d.to(self.device)
                # Zero gradients
                self.expert1.optimizer.zero_grad()
                self.expert2.optimizer.zero_grad()

                ############################################
                # Expert1 forward pass (all windows)
                ############################################
                latent1, x1 = self.expert1.forward(d)
                recon1 = recon_loss1_e1(x1, d) + recon_loss2_e1(x1[:, -self.latent_dim:], d[:, -self.latent_dim:])
                _, mean1, std1 = self.expert1.stationary_loss(latent1)
                total_loss_e1 = recon1 + mean1 + std1
                total_loss_e1.backward()
                self.expert1.optimizer.step()

                # For gating, compute per-window error using MSELoss with reduction='none'
                full_e1 = mse_full(x1, d)  # shape: [batch, window_size, n_features]
                # For each window, we take the last time step's error plus window_coef times the mean error.
                # Squeeze last dimension if needed.
                e1_last = full_e1[:, -1].squeeze(-1)  # shape: [batch]
                e1_mean = full_e1.mean(dim=(1, 2))  # shape: [batch]
                gating_error_1 = e1_last + window_coef * e1_mean  # shape: [batch]
                e1_errors_all.extend(gating_error_1.detach().cpu().numpy())

                e1_recon_list.append(recon1.item())
                e1_mean_list.append(mean1.item())
                e1_std_list.append(std1.item())

                ############################################
                # Gating: for each window in batch that exceeds threshold_e1,
                # duplicate it duplication_factor times and pass to expert2.
                ############################################
                # Ensure gating_error_1 is 1D:
                gating_error_1 = gating_error_1.squeeze()
                pass_mask = gating_error_1 > self.threshold_e1  # boolean mask [batch]

                if pass_mask.any():
                    d_pass = d[pass_mask]  # shape: [k, window_size, n_features]
                    # Duplicate these windows along batch dimension:
                    d_pass_dup = d_pass.repeat(self.duplication_factor, 1, 1)

                    latent2, x2 = self.expert2.forward(d_pass_dup)
                    recon2 = recon_loss1_e2(x2, d_pass_dup) + recon_loss2_e2(x2[:, -self.latent_dim:],
                                                                             d_pass_dup[:, -self.latent_dim:])
                    _, mean2, std2 = self.expert2.stationary_loss(latent2)
                    total_loss_e2 = recon2 + mean2 + std2

                    self.expert2.optimizer.zero_grad()
                    total_loss_e2.backward()
                    self.expert2.optimizer.step()

                    e2_recon_list.append(recon2.item())
                    e2_mean_list.append(mean2.item())
                    e2_std_list.append(std2.item())

                    # Compute gating error for expert2 (per window)
                    mse_full_2 = torch.nn.MSELoss(reduction='none').to(self.device)
                    full_e2 = mse_full_2(x2, d_pass_dup)
                    e2_last = full_e2[:, -1].mean(dim=1)
                    e2_mean = full_e2.mean(dim=(1, 2))
                    gating_error_2 = e2_last + window_coef * e2_mean
                    # Ensure 1D:
                    gating_error_2 = gating_error_2.view(-1)
                    e2_errors_all.extend(gating_error_2.detach().cpu().numpy())
                else:
                    e2_recon_list.append(0)
                    e2_mean_list.append(0)
                    e2_std_list.append(0)

            # End of epoch: update dynamic threshold for expert1
            e1_errors_all = np.array(e1_errors_all)
            mean_e1_val = np.mean(e1_errors_all)
            std_e1_val = np.std(e1_errors_all)
            self.threshold_e1 = mean_e1_val + self.threshold_sigma * std_e1_val
            self.thresholds_e1.append(self.threshold_e1)

            self.losses_e1.append(np.mean(e1_recon_list))
            self.losses_e2.append(np.mean(e2_recon_list))
            print(
                f"[Epoch {epoch + 1}/{n_epochs}] E1 Recon={np.mean(e1_recon_list):.4f}, E2 Recon={np.mean(e2_recon_list):.4f}, threshold_e1={self.threshold_e1:.4f}")

        # After training, compute threshold_e2 from the non-zero e2 errors
        e2_errors_all = np.array(e2_errors_all)
        e2_errors_all = e2_errors_all[e2_errors_all > 0]
        if len(e2_errors_all) == 0:
            self.threshold_e2 = 9999999.0
        else:
            mean_e2_val = np.mean(e2_errors_all)
            std_e2_val = np.std(e2_errors_all)
            self.threshold_e2 = mean_e2_val + self.threshold_sigma * std_e2_val
        self.thresholds_e2 = self.threshold_e2
        print(f"--> Final threshold_e2={self.threshold_e2:.4f}")

    ################################################################################
    # Prediction and Plotting functions follow (same as previous implementation)
    ################################################################################
    def predict_expert1(self, loader, train=False, window_coef=0.2):
        self.expert1.eval()
        results = {'inputs': [], 'anomalies': [], 'outputs': [], 'errors': [], 'means': [], 'stds': []}
        mse = torch.nn.MSELoss(reduction='none').to(self.device)
        with torch.no_grad():
            for window, anomaly in loader:
                if window.shape[0] == 1:
                    break
                results['inputs'].append(window.squeeze().T[-1].cpu().numpy())
                results['anomalies'].append(anomaly.squeeze().T[-1].cpu().numpy())
                window = window.to(self.device)
                latent, recons = self.expert1.forward(window)
                recons_ = recons.cpu().numpy().squeeze().T[-1]
                results['outputs'].append(recons_)
                rec_error = mse(window, recons)
                rec_error = rec_error[:, -1] + window_coef * torch.mean(rec_error, dim=1)
                results['errors'].append(rec_error.cpu().numpy().squeeze())
                _, mean_, std_ = self.expert1.stationary_loss(latent, per_batch=True)
                results['means'].append(mean_.cpu().numpy().squeeze())
                results['stds'].append(std_.cpu().numpy().squeeze())
        results['inputs'] = np.concatenate(results['inputs'])
        results['anomalies'] = np.concatenate(results['anomalies'])
        results['outputs'] = np.concatenate(results['outputs'])
        results['errors'] = np.concatenate(results['errors'])
        results['means'] = np.concatenate(results['means'])
        results['stds'] = np.concatenate(results['stds'])
        if (not train) and (self.threshold_e1 is not None):
            results['predictions'] = [1 if e > self.threshold_e1 else 0 for e in results['errors']]
        return results

    def plot_expert1(self, loader, train=False, plot_width=800):
        res = self.predict_expert1(loader, train=train)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(res['inputs']))),
                                 y=res['inputs'],
                                 mode='lines',
                                 name='Test Data',
                                 line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=list(range(len(res['outputs']))),
                                 y=res['outputs'],
                                 mode='lines',
                                 name='Predictions',
                                 line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=list(range(len(res['errors']))),
                                 y=res['errors'],
                                 mode='lines',
                                 name='Anomaly Errors',
                                 line=dict(color='red')))
        fig.add_trace(go.Scatter(x=list(range(len(res['means']))),
                                 y=res['means'],
                                 mode='lines',
                                 name='Mean Errors',
                                 line=dict(color='pink')))
        fig.add_trace(go.Scatter(x=list(range(len(res['stds']))),
                                 y=res['stds'],
                                 mode='lines',
                                 name='STD Errors',
                                 line=dict(color='green')))
        label_indices = [i for i, v in enumerate(res['anomalies']) if v == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[res['inputs'][i] for i in label_indices],
                                     mode='markers',
                                     name='Labels (Expert1)',
                                     marker=dict(color='orange', size=10)))
        if (not train) and (self.threshold_e1 is not None):
            fig.add_hline(y=self.threshold_e1, line_dash='dash', name='Threshold E1')
            if 'predictions' in res:
                pred_indices = [i for i, v in enumerate(res['predictions']) if v == 1]
                fig.add_trace(go.Scatter(x=pred_indices,
                                         y=[res['inputs'][i] for i in pred_indices],
                                         mode='markers',
                                         name='Predicted Anomalies (Expert1)',
                                         marker=dict(color='black', size=7, symbol='x')))
        fig.update_layout(title='Expert1 Window-level Results',
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'),
                          template='plotly',
                          width=plot_width)
        fig.show()

    def predict_expert2(self, loader, train=False, window_coef=0.2):
        self.expert2.eval()
        results = {'inputs': [], 'anomalies': [], 'outputs': [], 'errors': [], 'means': [], 'stds': []}
        mse = torch.nn.MSELoss(reduction='none').to(self.device)
        with torch.no_grad():
            for window, anomaly in loader:
                if window.shape[0] == 1:
                    break
                results['inputs'].append(window.squeeze().T[-1].cpu().numpy())
                results['anomalies'].append(anomaly.squeeze().T[-1].cpu().numpy())
                window = window.to(self.device)
                latent, recons = self.expert2.forward(window)
                recons_ = recons.cpu().numpy().squeeze().T[-1]
                results['outputs'].append(recons_)
                rec_error = mse(window, recons)
                rec_error = rec_error[:, -1] + window_coef * torch.mean(rec_error, dim=1)
                results['errors'].append(rec_error.cpu().numpy().squeeze())
                _, mean_, std_ = self.expert2.stationary_loss(latent, per_batch=True)
                results['means'].append(mean_.cpu().numpy().squeeze())
                results['stds'].append(std_.cpu().numpy().squeeze())
        results['inputs'] = np.concatenate(results['inputs'])
        results['anomalies'] = np.concatenate(results['anomalies'])
        results['outputs'] = np.concatenate(results['outputs'])
        results['errors'] = np.concatenate(results['errors'])
        results['means'] = np.concatenate(results['means'])
        results['stds'] = np.concatenate(results['stds'])
        if (not train) and (self.threshold_e2 is not None):
            results['predictions'] = [1 if e > self.threshold_e2 else 0 for e in results['errors']]
        return results

    def plot_expert2(self, loader, train=False, plot_width=800):
        res = self.predict_expert2(loader, train=train)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(res['inputs']))),
                                 y=res['inputs'],
                                 mode='lines',
                                 name='Test Data',
                                 line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=list(range(len(res['outputs']))),
                                 y=res['outputs'],
                                 mode='lines',
                                 name='Predictions',
                                 line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=list(range(len(res['errors']))),
                                 y=res['errors'],
                                 mode='lines',
                                 name='Anomaly Errors',
                                 line=dict(color='red')))
        fig.add_trace(go.Scatter(x=list(range(len(res['means']))),
                                 y=res['means'],
                                 mode='lines',
                                 name='Mean Errors',
                                 line=dict(color='pink')))
        fig.add_trace(go.Scatter(x=list(range(len(res['stds']))),
                                 y=res['stds'],
                                 mode='lines',
                                 name='STD Errors',
                                 line=dict(color='green')))
        label_indices = [i for i, v in enumerate(res['anomalies']) if v == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[res['inputs'][i] for i in label_indices],
                                     mode='markers',
                                     name='Labels (Expert2)',
                                     marker=dict(color='orange', size=10)))
        if (not train) and (self.threshold_e2 is not None):
            fig.add_hline(y=self.threshold_e2, line_dash='dash', name='Threshold E2')
            if 'predictions' in res:
                pred_indices = [i for i, v in enumerate(res['predictions']) if v == 1]
                fig.add_trace(go.Scatter(x=pred_indices,
                                         y=[res['inputs'][i] for i in pred_indices],
                                         mode='markers',
                                         name='Predicted Anomalies (Expert2)',
                                         marker=dict(color='black', size=7, symbol='x')))
        fig.update_layout(title='Expert2 Window-level Results',
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'),
                          template='plotly',
                          width=plot_width)
        fig.show()

    ################################################################################
    # Final MoE prediction and plotting with window-level gating
    ################################################################################
    def predict_final(self, loader, window_coef=0.2):
        """
        For each window in the batch:
           - Compute expert1's reconstruction error.
           - For each sample (window) individually, if error > threshold_e1, pass that window to expert2.
           - Record which windows were passed (in 'passed2').
        Returns a dictionary with predictions and an array 'passed2' indicating for each window whether it was routed to expert2.
        """
        self.expert1.eval()
        self.expert2.eval()
        results = {'inputs': [], 'anomalies': [], 'outputs': [], 'errors': [], 'means': [], 'stds': [], 'passed2': []}
        mse = torch.nn.MSELoss(reduction='none').to(self.device)

        with torch.no_grad():
            for window, anomaly in loader:
                if window.shape[0] == 1:
                    break
                batch_size = window.shape[0]
                inputs_ = window.squeeze().T[-1].cpu().numpy()
                anoms_ = anomaly.squeeze().T[-1].cpu().numpy()
                results['inputs'].append(inputs_)
                results['anomalies'].append(anoms_)

                window = window.to(self.device)
                latent1, recon1 = self.expert1.forward(window)
                rec_err1 = mse(window, recon1)
                rec_err1 = rec_err1[:, -1] + window_coef * torch.mean(rec_err1, dim=1)
                rec_err1 = rec_err1.squeeze()  # ensure shape [batch]
                rec_err1_vals = rec_err1.view(-1).cpu().numpy()  # force 1D array

                pass_mask = rec_err1 > self.threshold_e1  # shape [batch]

                d_pass = window[pass_mask]  # these windows are to be passed to expert2
                # Duplicate them duplication_factor times
                if d_pass.shape[0] > 0:
                    d_pass_dup = d_pass.repeat(self.duplication_factor, 1, 1)
                    latent2, recon2 = self.expert2.forward(d_pass_dup)
                    rec_err2 = mse(d_pass_dup, recon2)
                    rec_err2 = rec_err2[:, -1] + window_coef * torch.mean(rec_err2, dim=1)
                    rec_err2_vals = rec_err2.view(-1).cpu().numpy()
                else:
                    rec_err2_vals = np.array([])

                final_recons = []
                final_errors = []
                final_means = []
                final_stds = []
                passed2_arr = []
                e2_idx = 0
                for i in range(batch_size):
                    if pass_mask[i]:
                        # Use expert2's output (take the next duplicated sample)
                        final_recon_sample = recon2[e2_idx]
                        final_error_sample = rec_err2_vals[e2_idx]
                        latent_sample = latent2[e2_idx].unsqueeze(0)
                        _, m, s = self.expert2.stationary_loss(latent_sample, per_batch=True)
                        e2_idx += 1
                        passed2_arr.append(1)
                    else:
                        final_recon_sample = recon1[i]
                        final_error_sample = rec_err1_vals[i]
                        latent_sample = latent1[i].unsqueeze(0)
                        _, m, s = self.expert1.stationary_loss(latent_sample, per_batch=True)
                        passed2_arr.append(0)
                    final_recons.append(final_recon_sample.cpu().numpy().squeeze().T[-1])
                    final_errors.append(final_error_sample)
                    final_means.append(m.cpu().numpy().squeeze())
                    final_stds.append(s.cpu().numpy().squeeze())
                results['outputs'].append(np.array(final_recons))
                results['errors'].append(np.array(final_errors))
                results['means'].append(np.array(final_means))
                results['stds'].append(np.array(final_stds))
                results['passed2'].append(np.array(passed2_arr))

        # Flatten results
        results['inputs'] = np.concatenate(results['inputs'])
        results['anomalies'] = np.concatenate(results['anomalies'])
        results['outputs'] = np.concatenate(results['outputs'])
        results['errors'] = np.concatenate(results['errors'])
        results['means'] = np.concatenate(results['means'])
        results['stds'] = np.concatenate(results['stds'])
        results['passed2'] = np.concatenate(results['passed2'])

        if self.threshold_e2 is not None:
            final_preds = []
            for i, passed in enumerate(results['passed2']):
                if passed == 1:
                    final_preds.append(1 if results['errors'][i] > self.threshold_e2 else 0)
                else:
                    final_preds.append(1 if results['errors'][i] > self.threshold_e1 else 0)
            results['predictions'] = final_preds
        return results

    def plot_final_moe(self, loader, plot_width=800):
        """
        Plots the final MoE gating results on the test data.
        Also produces a second plot that highlights which windows were passed to expert2.
        """
        res = self.predict_final(loader)

        # First figure: standard final MoE plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(res['inputs']))),
                                 y=res['inputs'],
                                 mode='lines',
                                 name='Test Data',
                                 line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=list(range(len(res['outputs']))),
                                 y=res['outputs'],
                                 mode='lines',
                                 name='MoE Predictions',
                                 line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=list(range(len(res['errors']))),
                                 y=res['errors'],
                                 mode='lines',
                                 name='Anomaly Errors',
                                 line=dict(color='red')))
        fig.add_trace(go.Scatter(x=list(range(len(res['means']))),
                                 y=res['means'],
                                 mode='lines',
                                 name='Mean Errors',
                                 line=dict(color='pink')))
        fig.add_trace(go.Scatter(x=list(range(len(res['stds']))),
                                 y=res['stds'],
                                 mode='lines',
                                 name='STD Errors',
                                 line=dict(color='green')))
        label_indices = [i for i, v in enumerate(res['anomalies']) if v == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[res['inputs'][i] for i in label_indices],
                                     mode='markers',
                                     name='True Anomalies',
                                     marker=dict(color='orange', size=10)))
        if self.threshold_e1 is not None:
            fig.add_hline(y=self.threshold_e1, line_dash='dash', name='Threshold E1', line_color='gray')
        if self.threshold_e2 is not None:
            fig.add_hline(y=self.threshold_e2, line_dash='dash', name='Threshold E2', line_color='black')
        if 'predictions' in res:
            pred_indices = [i for i, v in enumerate(res['predictions']) if v == 1]
            fig.add_trace(go.Scatter(x=pred_indices,
                                     y=[res['inputs'][i] for i in pred_indices],
                                     mode='markers',
                                     name='MoE Predicted Anomalies',
                                     marker=dict(color='black', size=7, symbol='x')))
        fig.update_layout(title='Final MoE Gating Results',
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'),
                          template='plotly',
                          width=plot_width)
        fig.show()

        # Second figure: Highlight windows passed to expert2
        pass_e2_idxs = [i for i, v in enumerate(res['passed2']) if v == 1]
        if pass_e2_idxs:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=list(range(len(res['inputs']))),
                                      y=res['inputs'],
                                      mode='lines',
                                      name='Test Data',
                                      line=dict(color='blue')))
            fig2.add_trace(go.Scatter(x=pass_e2_idxs,
                                      y=[res['inputs'][i] for i in pass_e2_idxs],
                                      mode='markers',
                                      name='Windows Passed to Expert2',
                                      marker=dict(color='purple', size=8, symbol='diamond')))
            fig2.update_layout(title='Windows Passed to Expert2',
                               xaxis_title='Window Index',
                               yaxis_title='Value',
                               legend=dict(x=0, y=1, orientation='h'),
                               template='plotly',
                               width=plot_width)
            fig2.show()

    ################################################################################
    # Save and Load functions
    ################################################################################
    def save_models(self, dir_path='moe_models'):
        os.makedirs(dir_path, exist_ok=True)
        path_e1 = os.path.join(dir_path, 'expert1.pth')
        path_e2 = os.path.join(dir_path, 'expert2.pth')
        self.expert1.save(path_e1)
        self.expert2.save(path_e2)
        print(f"[MoE-Window] Experts saved: {path_e1}, {path_e2}")

    def load_models(self, dir_path='moe_models'):
        from torch import load
        path_e1 = os.path.join(dir_path, 'expert1.pth')
        path_e2 = os.path.join(dir_path, 'expert2.pth')
        self.expert1 = self.expert1.load(path_e1)
        self.expert2 = self.expert2.load(path_e2)
        print(f"[MoE-Window] Experts loaded from: {path_e1}, {path_e2}")
