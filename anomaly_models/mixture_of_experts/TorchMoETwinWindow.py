import torch
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm, trange
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import os

###############################################################################
# TorchMoETwinWindow Class
###############################################################################
class TorchMoETwinWindow:
    """
    A 2-expert Mixture-of-Experts for the 'Twin' model, but with per-window gating:
      - We have two Twin experts (expert1, expert2).
      - For each sample in the batch:
          1) Calculate expert1 recon error. If above threshold_e1 => pass that sample to expert2.
          2) This leads to partial sub-batches for expert2.
      - We update threshold_e1 each epoch from distribution of expert1 errors
      - We then update threshold_e2 from distribution of expert2 errors
      - We plot standard 'Twin' outputs, plus a special final plot to highlight which windows were passed to expert2
    """

    def __init__(
        self,
        TwinClass,
        window_size=256,
        device='cpu',
        data_id=0,
        threshold_sigma=3.0,
        n_features=1,
        latent_dim=32,
        seed=0,
        **twin_kwargs
    ):
        """
        :param TwinClass: The class for each expert, e.g. Twin.
        :param window_size: int, default 256
        :param device: 'cpu' or 'cuda'
        :param data_id: optional, not strictly used but kept for consistency
        :param threshold_sigma: multiplier for dynamic threshold (e.g. mean + sigma * std)
        :param n_features: default 1
        :param latent_dim: default 32
        :param seed: random seed
        :param twin_kwargs: Additional arguments to pass to the Twin constructor
        """
        self.expert1 = TwinClass(
            n_features=n_features,
            window_size=window_size,
            latent_dim=latent_dim,
            device=device,
            seed=seed,
            **twin_kwargs
        )
        self.expert2 = TwinClass(
            n_features=n_features,
            window_size=window_size,
            latent_dim=latent_dim,
            device=device,
            seed=seed+123,
            **twin_kwargs
        )

        self.threshold_e1 = -999999.0  # start super low => everything passes to e2 at epoch 0
        self.threshold_e2 = None
        self.threshold_sigma = threshold_sigma

        self.device = device
        self.window_size = window_size
        self.latent_dim  = latent_dim
        self.n_features  = n_features
        self.data_id     = data_id

        # For logging
        self.name = "TorchMoETwinWindow"
        self.losses_e1  = []
        self.losses_e2  = []
        self.thresholds_e1 = []
        self.thresholds_e2 = None

    ############################################################################
    # Train with window-level gating
    ############################################################################
    def train(
        self,
        train_loader,
        n_epochs=50,
        seed=42,
        loss_name='MaxDiff'
    ):
        """
        Window-level gating approach:
          For each batch:
             1) Expert1 forward pass for entire batch => recon error PER SAMPLE
             2) For samples whose error > threshold_e1 => pass them to expert2
             3) Two separate gradient updates
          end of epoch => threshold_e1 from distribution of e1 errors
          final => threshold_e2 from distribution of e2 errors
        """
        torch.manual_seed(seed)
        self.expert1.train()
        self.expert2.train()

        recon_loss_fn = self.expert1.select_loss(loss_name)

        # Outer loop over epochs
        for epoch_i in trange(n_epochs, desc="MoE Train (Window-level)"):
            e1_errors_all = []
            e2_errors_all = []

            e1_recon_list = []
            e1_mean_list  = []
            e1_std_list   = []
            e2_recon_list = []
            e2_mean_list  = []
            e2_std_list   = []

            for (d, a) in tqdm(train_loader, leave=False):
                d = d.to(self.device)

                # Zero grads
                self.expert1.optimizer.zero_grad()
                self.expert2.optimizer.zero_grad()

                ############################################
                # 1) Expert1 forward for entire batch
                ############################################
                latent1, x1 = self.expert1.forward(d)
                recon_errors_1 = recon_loss_fn(x1, d)  # scalar if 'MaxDiff'
                # But we want per-sample error => must replicate how Twin predict does it:
                # We'll do MSE for each sample to do gating. Or if user wants MaxDiff, we do that per sample.
                # Let's do a custom approach: we measure the last point or something like Twin's predict?

                # We'll replicate Twin's "predict" logic:
                # We'll do MSELoss(reduction='none') ourselves for gating
                # Then sum or average across dimension?
                mse_full = torch.nn.MSELoss(reduction='none').to(self.device)
                full_e1 = mse_full(x1, d)  # shape [batch, window_size, n_features]
                # Like Twin, we do last-step error + window_coef * mean
                # We'll define window_coef = 0.2 (like Twin)
                window_coef = 0.2
                # last-step error:
                e1_last = full_e1[:, -1]  # shape [batch, n_features]
                e1_last = torch.mean(e1_last, dim=1)  # shape [batch]
                # mean over entire window
                e1_mean = torch.mean(full_e1, dim=(1,2))  # shape [batch]
                gating_error_1 = e1_last + window_coef * e1_mean  # shape [batch]

                # Now also compute the stationary losses for entire batch
                # We'll define station_loss_1 as sum over batch
                total_recon1 = recon_errors_1  # a scalar
                _, mean1, std1 = self.expert1.stationary_loss(latent1)
                total_loss_e1 = total_recon1 + mean1 + std1

                # Backprop e1
                total_loss_e1.backward()
                self.expert1.optimizer.step()

                # Store e1 stats
                e1_recon_list.append(float(recon_errors_1.item()))
                e1_mean_list.append(mean1.item())
                e1_std_list.append(std1.item())

                # We'll record gating_error_1 in e1_errors_all for threshold
                # We do "mean" or "max"? We'll do each sample's gating_error
                e1_errors_all.extend(gating_error_1.detach().cpu().numpy())

                ############################################
                # 2) Gating => pass sub-batch to e2
                ############################################
                pass_mask = gating_error_1 > self.threshold_e1  # shape [batch]
                if pass_mask.any():
                    d_pass = d[pass_mask]  # shape [some, window_size, n_features]

                    # forward e2
                    latent2, x2 = self.expert2.forward(d_pass)
                    recon_errors_2 = recon_loss_fn(x2, d_pass)

                    # again for e2 gating stats
                    # We'll do the same approach with MSELoss to measure e2 error
                    mse_full_2 = torch.nn.MSELoss(reduction='none').to(self.device)
                    full_e2 = mse_full_2(x2, d_pass)
                    e2_last = full_e2[:, -1]
                    e2_last = torch.mean(e2_last, dim=1)
                    e2_mean = torch.mean(full_e2, dim=(1,2))
                    gating_error_2 = e2_last + window_coef * e2_mean

                    _, mean2, std2 = self.expert2.stationary_loss(latent2)
                    total_loss_e2 = recon_errors_2 + mean2 + std2

                    # zero e2 grad again? or do we keep?
                    # Typically we do a fresh backward pass
                    self.expert2.optimizer.zero_grad()
                    total_loss_e2.backward()
                    self.expert2.optimizer.step()

                    e2_recon_list.append(float(recon_errors_2.item()))
                    e2_mean_list.append(mean2.item())
                    e2_std_list.append(std2.item())

                    e2_errors_all.extend(gating_error_2.detach().cpu().numpy())
                else:
                    # no pass => no update for e2
                    e2_recon_list.append(0)
                    e2_mean_list.append(0)
                    e2_std_list.append(0)

            # End of epoch => compute threshold_e1
            e1_errors_all = np.array(e1_errors_all)
            mean_e1 = np.mean(e1_errors_all)
            std_e1  = np.std(e1_errors_all)
            self.threshold_e1 = mean_e1 + self.threshold_sigma * std_e1
            self.thresholds_e1.append(self.threshold_e1)

            # record average e1,e2 recon
            self.losses_e1.append(np.mean(e1_recon_list))
            self.losses_e2.append(np.mean(e2_recon_list))

            print(f"[Epoch {epoch_i+1}/{n_epochs}] E1 Recon={np.mean(e1_recon_list):.4f}, "
                  f"E2 Recon={np.mean(e2_recon_list):.4f}, threshold_e1={self.threshold_e1:.4f}")

        # After all epochs => threshold_e2
        e2_errors_all = np.array(e2_errors_all)
        e2_errors_all = e2_errors_all[e2_errors_all > 0]  # only non-zero
        if len(e2_errors_all) == 0:
            self.threshold_e2 = 9999999
        else:
            mean_e2 = np.mean(e2_errors_all)
            std_e2  = np.std(e2_errors_all)
            self.threshold_e2 = mean_e2 + self.threshold_sigma * std_e2
        self.thresholds_e2 = self.threshold_e2
        print(f"--> Final threshold_e2={self.threshold_e2:.4f}")

    ############################################################################
    # Plot each expert alone ignoring gating
    ############################################################################
    def predict_expert1(self, loader, train=False, window_coef=0.2):
        self.expert1.eval()
        results = {
            'inputs': [],
            'anomalies': [],
            'outputs': [],
            'errors': [],
            'means': [],
            'stds': []
        }
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
                rec_error = rec_error.cpu().numpy().squeeze()
                results['errors'].append(rec_error)

                _, mean_, std_ = self.expert1.stationary_loss(latent, per_batch=True)
                results['means'].append(mean_.cpu().numpy().squeeze())
                results['stds'].append(std_.cpu().numpy().squeeze())

        # flatten
        results['inputs']    = np.concatenate(results['inputs'])
        results['anomalies'] = np.concatenate(results['anomalies'])
        results['outputs']   = np.concatenate(results['outputs'])
        results['errors']    = np.concatenate(results['errors'])
        results['means']     = np.concatenate(results['means'])
        results['stds']      = np.concatenate(results['stds'])

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

        label_indices = [i for i,v in enumerate(res['anomalies']) if v == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[res['inputs'][i] for i in label_indices],
                                     mode='markers',
                                     name='Labels on Test Data',
                                     marker=dict(color='orange', size=10)))
        if (not train) and (self.threshold_e1 is not None):
            fig.add_hline(y=self.threshold_e1, line_dash='dash', name='Threshold E1')
            if 'predictions' in res:
                pred_indices = [i for i,v in enumerate(res['predictions']) if v == 1]
                fig.add_trace(go.Scatter(x=pred_indices,
                                         y=[res['inputs'][i] for i in pred_indices],
                                         mode='markers',
                                         name='Predicted Anomalies (E1)',
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
        results = {
            'inputs': [],
            'anomalies': [],
            'outputs': [],
            'errors': [],
            'means': [],
            'stds': []
        }
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
                rec_error = rec_error.cpu().numpy().squeeze()
                results['errors'].append(rec_error)

                _, mean_, std_ = self.expert2.stationary_loss(latent, per_batch=True)
                results['means'].append(mean_.cpu().numpy().squeeze())
                results['stds'].append(std_.cpu().numpy().squeeze())

        # flatten
        results['inputs']    = np.concatenate(results['inputs'])
        results['anomalies'] = np.concatenate(results['anomalies'])
        results['outputs']   = np.concatenate(results['outputs'])
        results['errors']    = np.concatenate(results['errors'])
        results['means']     = np.concatenate(results['means'])
        results['stds']      = np.concatenate(results['stds'])

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

        label_indices = [i for i,v in enumerate(res['anomalies']) if v == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[res['inputs'][i] for i in label_indices],
                                     mode='markers',
                                     name='Labels on Test Data',
                                     marker=dict(color='orange', size=10)))
        if (not train) and (self.threshold_e2 is not None):
            fig.add_hline(y=self.threshold_e2, line_dash='dash', name='Threshold E2')
            if 'predictions' in res:
                pred_indices = [i for i,v in enumerate(res['predictions']) if v == 1]
                fig.add_trace(go.Scatter(x=pred_indices,
                                         y=[res['inputs'][i] for i in pred_indices],
                                         mode='markers',
                                         name='Predicted Anomalies (E2)',
                                         marker=dict(color='black', size=7, symbol='x')))

        fig.update_layout(title='Expert2 Window-level Results',
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'),
                          template='plotly',
                          width=plot_width)
        fig.show()

    ############################################################################
    # Final Gating: If e1_error <= threshold_e1 => use e1, else => use e2
    # plus a separate "passed2" array to color those windows differently
    ############################################################################
    def predict_final(self, loader, window_coef=0.2):
        """
        For each sample in the batch => check e1 error => if > threshold_e1 => pass that sample to e2
        We'll store which windows pass to e2 in 'passed2' array (1 if passed).
        """
        self.expert1.eval()
        self.expert2.eval()
        results = {
            'inputs': [],
            'anomalies': [],
            'outputs': [],
            'errors': [],
            'means': [],
            'stds': [],
            'passed2': []
        }
        mse = torch.nn.MSELoss(reduction='none').to(self.device)

        with torch.no_grad():
            for window, anomaly in loader:
                if window.shape[0] == 1:
                    break
                batch_size = window.shape[0]
                inputs_ = window.squeeze().T[-1].cpu().numpy()
                anoms_  = anomaly.squeeze().T[-1].cpu().numpy()

                window = window.to(self.device)
                # Expert1 forward
                latent1, recon1 = self.expert1.forward(window)
                rec_err1 = mse(window, recon1)
                rec_err1 = rec_err1[:, -1] + window_coef*torch.mean(rec_err1, dim=1)
                rec_err1_vals = rec_err1.cpu().numpy().squeeze()

                rec_err1 = rec_err1.squeeze(-1)
                pass_mask = rec_err1 > self.threshold_e1  # shape [batch]
                # We'll do partial sub-batch for e2
                d_pass = window[pass_mask]
                if d_pass.shape[0] > 0:
                    latent2, recon2 = self.expert2.forward(d_pass)
                    rec_err2 = mse(d_pass, recon2)
                    rec_err2 = rec_err2[:, -1] + window_coef * torch.mean(rec_err2, dim=1)
                    rec_err2_vals = rec_err2.view(-1).cpu().numpy()  # now shape = [sub_batch_size]
                else:
                    rec_err2_vals = []

                # Build final array
                final_recons = []
                final_errors = []
                final_means  = []
                final_stds   = []
                passed2_arr  = []
                # We must re-run the stationary_loss for the chosen expert
                # We'll do it sample by sample or at least we gather them
                # We'll store latent2 for those that pass, else latent1
                # For quickness, let's store them in lists
                e2_idx = 0

                # We also need to do a "re-run" or we can do partial for means, stds
                # We'll do chunk approach. We'll define a function that gives means, std for each sample.
                # Or do we do it batch-level again? It's simpler to do one by one. It's not super efficient
                # but okay for demonstration.

                for i in range(batch_size):
                    if pass_mask[i]:
                        # use e2
                        final_recon_sample = recon2[e2_idx]
                        final_error_sample = rec_err2_vals[e2_idx]
                        # stationary loss
                        latent_samp = latent2[e2_idx].unsqueeze(0)  # shape [1, latent_dim]
                        _, m_, s_ = self.expert2.stationary_loss(latent_samp, per_batch=True)
                        m_ = m_.cpu().numpy().squeeze()
                        s_ = s_.cpu().numpy().squeeze()
                        e2_idx += 1
                        passed2_arr.append(1)
                    else:
                        # use e1
                        final_recon_sample = recon1[i]
                        final_error_sample = rec_err1_vals[i]
                        latent_samp = latent1[i].unsqueeze(0)
                        _, m_, s_ = self.expert1.stationary_loss(latent_samp, per_batch=True)
                        m_ = m_.cpu().numpy().squeeze()
                        s_ = s_.cpu().numpy().squeeze()
                        passed2_arr.append(0)

                    final_recons.append(final_recon_sample.cpu().numpy().squeeze().T[-1])
                    final_errors.append(final_error_sample)
                    final_means.append(m_)
                    final_stds.append(s_)

                # gather
                results['inputs'].append(inputs_)
                results['anomalies'].append(anoms_)
                results['outputs'].append(np.array(final_recons))
                results['errors'].append(np.array(final_errors))
                results['means'].append(np.array(final_means))
                results['stds'].append(np.array(final_stds))
                results['passed2'].append(np.array(passed2_arr))

        # Flatten
        results['inputs']    = np.concatenate(results['inputs'])
        results['anomalies'] = np.concatenate(results['anomalies'])
        results['outputs']   = np.concatenate(results['outputs'])
        results['errors']    = np.concatenate(results['errors'])
        results['means']     = np.concatenate(results['means'])
        results['stds']      = np.concatenate(results['stds'])
        results['passed2']   = np.concatenate(results['passed2'])

        # If we have threshold_e2 => produce predictions
        if self.threshold_e2 is not None:
            final_preds = []
            for i, pass_e2 in enumerate(results['passed2']):
                if pass_e2 == 1:
                    # use threshold_e2
                    if results['errors'][i] > self.threshold_e2:
                        final_preds.append(1)
                    else:
                        final_preds.append(0)
                else:
                    # threshold_e1
                    if results['errors'][i] > self.threshold_e1:
                        final_preds.append(1)
                    else:
                        final_preds.append(0)
            results['predictions'] = final_preds

        return results

    def plot_final_moe(self, loader, plot_width=800):
        """
        Plots final gating-based usage of experts (per window) + draws threshold lines
        Also we highlight the windows that got passed to expert2
        """
        res = self.predict_final(loader)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(res['inputs']))),
                                 y=res['inputs'],
                                 mode='lines',
                                 name='Test Data',
                                 line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=list(range(len(res['outputs']))),
                                 y=res['outputs'],
                                 mode='lines',
                                 name='Predictions (MoE)',
                                 line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=list(range(len(res['errors']))),
                                 y=res['errors'],
                                 mode='lines',
                                 name='Anomaly Errors (MoE)',
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

        label_indices = [i for i,v in enumerate(res['anomalies']) if v == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[res['inputs'][i] for i in label_indices],
                                     mode='markers',
                                     name='Labels (Test Data)',
                                     marker=dict(color='orange', size=10)))

        if self.threshold_e1 is not None:
            fig.add_hline(y=self.threshold_e1, line_dash='dash', name='Threshold E1', line_color='gray')
        if self.threshold_e2 is not None:
            fig.add_hline(y=self.threshold_e2, line_dash='dash', name='Threshold E2', line_color='black')

        # show predicted anomalies if we have them
        if 'predictions' in res:
            pred_indices = [i for i,v in enumerate(res['predictions']) if v == 1]
            if pred_indices:
                fig.add_trace(go.Scatter(x=pred_indices,
                                         y=[res['inputs'][i] for i in pred_indices],
                                         mode='markers',
                                         name='MoE Anomalies',
                                         marker=dict(color='black', size=7, symbol='x')))

        fig.update_layout(title='Final MoE Window-level Gating', xaxis_title='Time Steps', yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'), template='plotly', width=plot_width)
        fig.show()

        # Let's also produce a second figure to highlight which windows got passed to e2
        # We'll color them differently
        pass_e2_idxs = [i for i, v in enumerate(res['passed2']) if v == 1]
        if pass_e2_idxs:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=list(range(len(res['inputs']))),
                                      y=res['inputs'],
                                      mode='lines',
                                      name='Test Data',
                                      line=dict(color='blue')))
            # highlight windows passed to e2
            fig2.add_trace(go.Scatter(x=pass_e2_idxs,
                                      y=[res['inputs'][i] for i in pass_e2_idxs],
                                      mode='markers',
                                      name='Passed to Expert2',
                                      marker=dict(color='purple', size=8, symbol='diamond')))

            fig2.update_layout(title='Windows Passed to Expert2', xaxis_title='Window Index', yaxis_title='Value',
                               legend=dict(x=0, y=1, orientation='h'), template='plotly', width=plot_width)
            fig2.show()

    ############################################################################
    # Save/Load
    ############################################################################
    def save_models(self, dir_path='moe_models_window'):
        os.makedirs(dir_path, exist_ok=True)
        path_e1 = os.path.join(dir_path, 'expert1.pth')
        path_e2 = os.path.join(dir_path, 'expert2.pth')
        self.expert1.save(path_e1)
        self.expert2.save(path_e2)
        print(f"[MoE-Window] Experts saved: {path_e1}, {path_e2}")

    def load_models(self, dir_path='moe_models_window'):
        path_e1 = os.path.join(dir_path, 'expert1.pth')
        path_e2 = os.path.join(dir_path, 'expert2.pth')
        self.expert1 = self.expert1.load(path_e1)
        self.expert2 = self.expert2.load(path_e2)
        print(f"[MoE-Window] Experts loaded from: {path_e1}, {path_e2}")
