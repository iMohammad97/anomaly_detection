import torch
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm, trange
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import os

################################################################################
# TorchMoETwin Class
################################################################################
class TorchMoETwin:
    """
    A 2-expert Mixture of Experts for the Torch-based 'Twin' model:
      - Each expert is an instance of 'Twin'
      - We apply dynamic gating thresholds:
          1) All windows go to expert1
          2) If recon error of expert1 > threshold_e1 => pass to expert2
          3) We update thresholds after each epoch

    It preserves Twin's logic (reconstruction + stationary loss).
    """

    def __init__(self,
                 TwinClass,
                 window_size=256,
                 device='cpu',
                 data_id=0,
                 threshold_sigma=3.0,
                 # Additional Twin args:
                 n_features=1,
                 latent_dim=32,
                 seed=0,
                 **twin_kwargs
                 ):
        """
        :param TwinClass: The class of your expert, e.g. 'Twin' as given in your code.
        :param window_size: int, default 256
        :param device: 'cpu' or 'cuda'
        :param data_id: optional ID or such, not strictly necessary
        :param threshold_sigma: multiplier for dynamic threshold, default 3.0
        :param n_features: default 1
        :param latent_dim: default 32
        :param seed: random seed
        :param twin_kwargs: any additional arguments to pass to each Twin instance
        """
        # Create two experts
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
                                 seed=seed+123,
                                 **twin_kwargs)

        # Gating thresholds
        # We will update threshold_e1 after each epoch, threshold_e2 at the end
        self.threshold_e1 = -999999.0   # start super low => pass everything to expert2 in epoch 0
        self.threshold_e2 = None

        self.threshold_sigma = threshold_sigma
        self.device = device
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.n_features  = n_features
        self.data_id     = data_id

        # For logging
        self.name = "TorchMoETwin"
        self.losses_e1 = []
        self.losses_e2 = []
        self.thresholds_e1 = []
        self.thresholds_e2 = None

        # Final gating results
        self.final_threshold_e2 = None   # store final dynamic threshold e2
        self.final_inputs    = None
        self.final_anomalies = None
        self.final_predictions = None
        self.final_errors    = None
        self.final_outputs   = None
        self.final_means     = None
        self.final_stds      = None

    ############################################################################
    # 2) Train MoE with gating
    ############################################################################
    def train(self,
              train_loader,
              n_epochs=50,
              seed=42,
              loss_name='MaxDiff'):
        """
        Replicates Twin.learn logic but with gating:
          1) For each batch:
             - forward pass expert1 => recon error + stationary => loss1
             - gating => pass sub-batch to expert2 => recon2 error + stationary => loss2
          2) After epoch => recompute threshold_e1 from distribution of e1 recon errors
          3) Finally => threshold_e2 from distribution of e2 recon errors
        """
        torch.manual_seed(seed)
        self.expert1.train()
        self.expert2.train()

        recon_loss_fn = self.expert1.select_loss(loss_name)
        # We assume both experts use same type of loss

        # We'll do an outer loop over epochs, exactly like Twin
        for epoch_i in trange(n_epochs, desc="MoE Training"):
            # We'll gather errors for e1 to update threshold at epoch end
            e1_errors_all = []
            e2_errors_all = []

            # We'll track average losses for e1/e2
            e1_recon_list = []
            e1_mean_list  = []
            e1_std_list   = []
            e2_recon_list = []
            e2_mean_list  = []
            e2_std_list   = []

            for (d, a) in tqdm(train_loader, leave=False):
                d = d.to(self.device)
                # 1) Expert1 forward
                latent1, x1 = self.expert1.forward(d)
                recon1 = recon_loss_fn(x1, d)

                # stationary
                _, mean1, std1 = self.expert1.stationary_loss(latent1)

                # gating error => we'll define gating error as recon1 item (like in predict)
                gating_err1 = recon1.item()
                e1_total_loss = recon1 + mean1 + std1

                # zero grad
                self.expert1.optimizer.zero_grad()
                self.expert2.optimizer.zero_grad()
                # backward e1
                e1_total_loss.backward()
                self.expert1.optimizer.step()

                # store e1 stats
                e1_recon_list.append(recon1.item())
                e1_mean_list.append(mean1.item())
                e1_std_list.append(std1.item())
                e1_errors_all.append(gating_err1)

                # gating => if gating_err1 > threshold_e1 => pass entire batch to expert2
                # NOTE: This is a batch-level gating, i.e. either pass the entire batch or not.
                # If you want a per-example gating, you'd do more complicated indexing.
                if gating_err1 > self.threshold_e1:
                    # forward expert2
                    latent2, x2 = self.expert2.forward(d)
                    recon2 = recon_loss_fn(x2, d)
                    _, mean2, std2 = self.expert2.stationary_loss(latent2)
                    e2_total_loss = recon2 + mean2 + std2
                    e2_total_loss.backward()
                    self.expert2.optimizer.step()

                    e2_recon_list.append(recon2.item())
                    e2_mean_list.append(mean2.item())
                    e2_std_list.append(std2.item())
                    e2_errors_all.append(recon2.item())
                else:
                    # no pass => no update for expert2
                    e2_recon_list.append(0)
                    e2_mean_list.append(0)
                    e2_std_list.append(0)
                    e2_errors_all.append(0)

            # At epoch end => compute new threshold_e1 from all e1_errors
            e1_errors_all = np.array(e1_errors_all)
            mean_e1 = np.mean(e1_errors_all)
            std_e1  = np.std(e1_errors_all)
            self.threshold_e1 = mean_e1 + self.threshold_sigma * std_e1

            # log
            self.losses_e1.append(np.mean(e1_recon_list))
            self.losses_e2.append(np.mean(e2_recon_list))
            self.thresholds_e1.append(self.threshold_e1)

            print(f"[Epoch {epoch_i+1}/{n_epochs}] E1 Recon={np.mean(e1_recon_list):.4f}, "
                  f"E2 Recon={np.mean(e2_recon_list):.4f}, threshold_e1={self.threshold_e1:.4f}")

        # Finally => compute threshold_e2 from e2_errors_all (the ones that actually got passed)
        e2_errors_all = np.array([e for e in e2_errors_all if e > 0])  # only non-zero
        if len(e2_errors_all) == 0:
            self.threshold_e2 = 9999999.0
        else:
            mean_e2 = np.mean(e2_errors_all)
            std_e2  = np.std(e2_errors_all)
            self.threshold_e2 = mean_e2 + self.threshold_sigma * std_e2
        self.thresholds_e2 = self.threshold_e2
        print(f"--> Final threshold_e2={self.threshold_e2:.4f}")

    ############################################################################
    # 3) Evaluate/Plot Each Expert Alone
    ############################################################################
    def predict_expert1(self, loader, train=False, window_coef=0.2):
        """
        Exactly like 'Twin.predict', but we always use self.expert1 ignoring gating.
        Returns a dictionary of results with 'inputs', 'outputs', 'errors', 'anomalies', etc.
        """
        self.expert1.eval()
        results = {
            'inputs': [],
            'anomalies': [],
            'outputs': [],
            'errors': [],
            'means': [],
            'stds': []
        }
        # We replicate the logic from Twin.predict
        mse = torch.nn.MSELoss(reduction='none').to(self.device)
        with torch.no_grad():
            for window, anomaly in loader:
                if window.shape[0] == 1:
                    break
                results['inputs'].append(window.squeeze().T[-1].cpu().numpy())
                results['anomalies'].append(anomaly.squeeze().T[-1].cpu().numpy())

                window = window.to(self.device)
                latent, recons = self.expert1.forward(window)
                recons_ = recons.cpu().detach().numpy().squeeze().T[-1]
                results['outputs'].append(recons_)

                rec_error = mse(window, recons)
                rec_error = rec_error[:, -1] + window_coef * torch.mean(rec_error, dim=1)
                rec_error = rec_error.cpu().detach().numpy().squeeze()
                results['errors'].append(rec_error)

                _, mean_, std_ = self.expert1.stationary_loss(latent, per_batch=True)
                results['means'].append(mean_.cpu().detach().numpy().squeeze())
                results['stds'].append(std_.cpu().detach().numpy().squeeze())

        # Flatten
        results['inputs']    = np.concatenate(results['inputs'])
        results['anomalies'] = np.concatenate(results['anomalies'])
        results['outputs']   = np.concatenate(results['outputs'])
        results['errors']    = np.concatenate(results['errors'])
        results['means']     = np.concatenate(results['means'])
        results['stds']      = np.concatenate(results['stds'])

        # If train and no threshold => define e1 threshold or do nothing
        if train and self.threshold_e1 < 0:
            # e.g. set threshold e1
            pass
        elif not train and self.threshold_e1 is not None:
            # produce predictions
            results['predictions'] = [1 if e > self.threshold_e1 else 0 for e in results['errors']]
        return results

    def plot_expert1(self, loader, train=False, plot_width=800):
        """
        Plots exactly like Twin.plot_results, but uses expert1 alone ignoring gating.
        """
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
                                     name='Labels on Test Data',
                                     marker=dict(color='orange', size=10)))
        if (not train) and (self.threshold_e1 is not None):
            fig.add_hline(y=self.threshold_e1, line_dash='dash', name='Threshold E1')
            # predicted anomalies
            if 'predictions' in res:
                pred_indices = [i for i, v in enumerate(res['predictions']) if v == 1]
                fig.add_trace(go.Scatter(x=pred_indices,
                                         y=[res['inputs'][i] for i in pred_indices],
                                         mode='markers',
                                         name='Predictions (Expert1)',
                                         marker=dict(color='black', size=7, symbol='x')))

        fig.update_layout(title='Expert1 Results', xaxis_title='Time Steps', yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'), template='plotly', width=plot_width)
        fig.show()

    def predict_expert2(self, loader, train=False, window_coef=0.2):
        """
        Always use self.expert2 alone ignoring gating, replicate logic of Twin.predict.
        """
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
                recons_ = recons.cpu().detach().numpy().squeeze().T[-1]
                results['outputs'].append(recons_)

                rec_error = mse(window, recons)
                rec_error = rec_error[:, -1] + window_coef * torch.mean(rec_error, dim=1)
                rec_error = rec_error.cpu().detach().numpy().squeeze()
                results['errors'].append(rec_error)

                _, mean_, std_ = self.expert2.stationary_loss(latent, per_batch=True)
                results['means'].append(mean_.cpu().detach().numpy().squeeze())
                results['stds'].append(std_.cpu().detach().numpy().squeeze())

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
        """
        Plot exactly like Twin.plot_results but for expert2 alone ignoring gating.
        """
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
                                     name='Labels on Test Data',
                                     marker=dict(color='orange', size=10)))
        if (not train) and (self.threshold_e2 is not None):
            fig.add_hline(y=self.threshold_e2, line_dash='dash', name='Threshold E2')
            if 'predictions' in res:
                pred_indices = [i for i, v in enumerate(res['predictions']) if v == 1]
                fig.add_trace(go.Scatter(x=pred_indices,
                                         y=[res['inputs'][i] for i in pred_indices],
                                         mode='markers',
                                         name='Predictions (Expert2)',
                                         marker=dict(color='black', size=7, symbol='x')))

        fig.update_layout(title='Expert2 Results', xaxis_title='Time Steps', yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'), template='plotly', width=plot_width)
        fig.show()

    ############################################################################
    # 4) Final MoE gating evaluation
    ############################################################################
    def predict_final(self, loader, window_coef=0.2):
        """
        If e1_error <= threshold_e1 => use expert1's predictions
        else => pass entire batch to expert2 => use e2
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
            'expert_used': []  # track which expert was used
        }
        mse = torch.nn.MSELoss(reduction='none').to(self.device)

        with torch.no_grad():
            for window, anomaly in loader:
                if window.shape[0] == 1:
                    break
                # We'll do a single gating decision PER BATCH here
                # If you want gating per example inside the batch, you'd do partial indexing.
                batch_inputs = window.squeeze().T[-1].cpu().numpy()
                batch_anomalies = anomaly.squeeze().T[-1].cpu().numpy()
                results['inputs'].append(batch_inputs)
                results['anomalies'].append(batch_anomalies)

                window = window.to(self.device)
                # Step 1: compute e1 recon error
                latent1, recons1 = self.expert1.forward(window)
                rec_error1 = mse(window, recons1)
                rec_error1 = rec_error1[:, -1] + window_coef*torch.mean(rec_error1, dim=1)
                error_val1 = rec_error1.mean().item()  # average across batch or just take first?

                # gating
                if error_val1 <= self.threshold_e1:
                    # use expert1's output
                    final_recons = recons1
                    final_latent = latent1
                    used_expert = "E1"
                else:
                    # pass entire batch to e2
                    latent2, recons2 = self.expert2.forward(window)
                    rec_error2 = mse(window, recons2)
                    rec_error2 = rec_error2[:, -1] + window_coef*torch.mean(rec_error2, dim=1)
                    error_val2 = rec_error2.mean().item()
                    final_recons = recons2
                    final_latent = latent2
                    used_expert = "E2"

                # Now compute final predictions
                final_recons_ = final_recons.cpu().detach().numpy().squeeze().T[-1]
                results['outputs'].append(final_recons_)

                # We define final error as the average of the final_recons error
                final_err = mse(window, final_recons)
                final_err = final_err[:, -1] + window_coef*torch.mean(final_err, dim=1)
                final_err = final_err.cpu().detach().numpy().squeeze()
                results['errors'].append(final_err)

                _, mean_, std_ = self.expert1.stationary_loss(final_latent, per_batch=True) \
                                 if used_expert=="E1" else \
                                 self.expert2.stationary_loss(final_latent, per_batch=True)
                results['means'].append(mean_.cpu().detach().numpy().squeeze())
                results['stds'].append(std_.cpu().detach().numpy().squeeze())
                results['expert_used'].append([used_expert]*len(batch_inputs))

        # Flatten
        results['inputs']    = np.concatenate(results['inputs'])
        results['anomalies'] = np.concatenate(results['anomalies'])
        results['outputs']   = np.concatenate(results['outputs'])
        results['errors']    = np.concatenate(results['errors'])
        results['means']     = np.concatenate(results['means'])
        results['stds']      = np.concatenate(results['stds'])
        results['expert_used'] = np.concatenate(results['expert_used'])

        # If threshold_e2 is set => produce final predictions
        if self.threshold_e2 is not None:
            final_preds = []
            for i, e_used in enumerate(results['expert_used']):
                # if used E1 => threshold e1
                # if used E2 => threshold e2
                if e_used == "E1":
                    if results['errors'][i] > self.threshold_e1:
                        final_preds.append(1)
                    else:
                        final_preds.append(0)
                else:
                    if results['errors'][i] > self.threshold_e2:
                        final_preds.append(1)
                    else:
                        final_preds.append(0)
            results['predictions'] = final_preds
        return results

    def plot_final_moe(self, loader, plot_width=800):
        """
        Plots final gating-based usage of experts:
        - If e1_error <= threshold => use e1
        - else => use e2
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
                                     name='Labels on Test Data',
                                     marker=dict(color='orange', size=10)))
        if (self.threshold_e1 is not None) and (self.threshold_e2 is not None):
            fig.add_hline(y=self.threshold_e1, line_dash='dash', name='Threshold E1')
            fig.add_hline(y=self.threshold_e2, line_dash='dash', line_color='blue', name='Threshold E2')
            if 'predictions' in res:
                pred_indices = [i for i, v in enumerate(res['predictions']) if v == 1]
                fig.add_trace(go.Scatter(x=pred_indices,
                                         y=[res['inputs'][i] for i in pred_indices],
                                         mode='markers',
                                         name='Predictions (MoE)',
                                         marker=dict(color='black', size=7, symbol='x')))

        fig.update_layout(title='Final MoE Gating Results', xaxis_title='Time Steps', yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'), template='plotly', width=plot_width)
        fig.show()

    ############################################################################
    # 5) Save and Load
    ############################################################################
    def save_models(self, dir_path='moe_models'):
        os.makedirs(dir_path, exist_ok=True)
        # We'll reuse self.expert1.save(...) if it exists
        path_e1 = os.path.join(dir_path, 'expert1.pth')
        path_e2 = os.path.join(dir_path, 'expert2.pth')
        self.expert1.save(path_e1)
        self.expert2.save(path_e2)
        print(f"[MoE] Experts saved: {path_e1}, {path_e2}")

    def load_models(self, dir_path='moe_models'):
        path_e1 = os.path.join(dir_path, 'expert1.pth')
        path_e2 = os.path.join(dir_path, 'expert2.pth')
        self.expert1 = self.expert1.load(path_e1)
        self.expert2 = self.expert2.load(path_e2)
        print(f"[MoE] Experts loaded from: {path_e1}, {path_e2}")
