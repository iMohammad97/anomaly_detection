import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm.notebook import trange, tqdm
import plotly.graph_objects as go

###############################################################################
# TorchMoE class
###############################################################################
class TorchMoE:
    """
    A 2-expert Mixture-of-Experts class for Torch-based models with dynamic threshold gating:
      1) expert1 sees every batch
      2) if reconstruction error > dynamic threshold => pass sub-batch to expert2
      3) separate gradient updates for each sub-expert
      4) dynamic threshold update at end of each epoch for expert1
      5) separate threshold for expert2 from windows that actually pass

    This design specifically integrates two copies of 'Twin' (or any Torch model
    following a similar "forward" + "stationary_loss" pattern).

    USAGE:
      - Instantiate with ExpertClass=Twin (or our Torch model) and relevant hyperparams.
      - call .train(train_loader, n_epochs=..., loss_name='MaxDiff', ...)
      - call .evaluate(test_loader)
      - call .plot_expert1(test_loader), .plot_expert2(test_loader), .plot_final_moe(test_loader)
    """

    def __init__(
        self,
        ExpertClass,
        window_size=256,
        device='cpu',
        threshold_sigma=2.0,
        seed=0,
        # Additional arguments to pass to the ExpertClass
        **expert_kwargs
    ):
        """
        :param ExpertClass: A Torch model class, e.g. Twin
        :param window_size: Sliding window size
        :param device: 'cpu' or 'cuda'
        :param threshold_sigma: dynamic threshold = mean + sigma * std
        :param seed: random seed
        :param expert_kwargs: additional config passed to each expert's constructor
        """
        torch.manual_seed(seed)
        self.seed = seed
        self.device = device
        self.window_size = window_size
        self.threshold_sigma = threshold_sigma

        # Build two sub-experts
        self.expert1 = ExpertClass(
            window_size=self.window_size,
            device=self.device,
            seed=self.seed,
            **expert_kwargs
        )
        self.expert2 = ExpertClass(
            window_size=self.window_size,
            device=self.device,
            seed=self.seed + 123,
            **expert_kwargs
        )

        # Each expert will have its own optimizer, defined later
        # But the 'Twin' constructor already defines self.optimizer, etc.
        # We'll override or re-init if we want separate configs.

        # Gating thresholds
        self.threshold_e1 = 0.0
        self.threshold_e2 = 0.0

        # Final gating results
        self.final_errors = None
        self.final_preds = None

        self.loss_name = None

    ###########################################################################
    # 1) TRAIN with gating
    ###########################################################################
    def train(self, train_loader, n_epochs=50, loss_name='MaxDiff'):
        """
        Custom training loop with gating:
          - For each batch:
              1) Pass entire batch to expert1 => compute reconstruction + stationary losses
              2) If error > threshold_e1 => sub-batch also passed to expert2
              3) separate backward passes
          - End of epoch => recompute threshold_e1 with mean+ sigma*std from entire dataset
          - End => threshold_e2 from windows that pass
        :param train_loader: DataLoader returning (data, anomaly) for each batch
        :param n_epochs: number of epochs
        :param loss_name: e.g. 'MaxDiff' or 'MSE', etc. (the Twin model's select_loss keys)
        """
        self.loss_name = loss_name
        # We'll define negative threshold so in the first epoch, everything passes to e2
        self.threshold_e1 = -9999999.0

        # Expert1, Expert2 => each already has an optimizer from Twin constructor
        # We can confirm or override. We'll keep what's built in, i.e. self.expertX.optimizer
        # Then we define a reference to the reconstruction loss function:
        recon_loss_func = self.expert1.select_loss(loss_name).to(self.device)

        # We'll do multiple epochs
        best_combined = np.inf
        patience = 10
        patience_counter = 0

        for epoch_i in trange(n_epochs, desc="MoE Training"):
            # We'll store batch losses for e1, e2, so we can see how training is going
            e1_batch_losses = []
            e2_batch_losses = []

            # We'll also gather all e1 errors for threshold update
            e1_errors_all = []

            for data_batch, _ in tqdm(train_loader, leave=False):
                data_batch = data_batch.to(self.device)  # shape: (batch, window_size, features=1)

                # 1) forward pass on expert1
                self.expert1.optimizer.zero_grad()
                latent1, recon1 = self.expert1.forward(data_batch)
                # measure reconstruction error (scalar)
                # We'll do something akin to 'Twin' approach:
                #   recon = recon_loss_func(recon1, data_batch)
                recon_val = recon_loss_func(recon1, data_batch)
                if loss_name == 'MaxDiff':
                    # If it's a lambda returning max of abs error, that's a single value
                    e1_error_vec = (recon1 - data_batch).abs().max(dim=2)[0].max(dim=1)[0]
                else:
                    # For MSE or Huber, we can get elementwise, then average
                    # We'll define e1_error_vec as the per-sample error
                    # MSELoss is "mean" by default, so we do a reduce=None approach
                    # but our Twin's code lumps it. We'll do a simpler approach:
                    # We'll forcibly compute per-sample MSE:
                    e1_error_vec = torch.mean((recon1 - data_batch) ** 2, dim=(1,2))

                # stationary loss => (loss, mean_loss, std_loss) from Twin
                sl, meanL, stdL = self.expert1.stationary_loss(latent1, per_batch=False)

                # total loss for e1
                loss_e1 = recon_val + sl

                # We'll do backward
                loss_e1.backward()
                self.expert1.optimizer.step()

                e1_batch_losses.append(loss_e1.item())

                # 2) Gating => if e1_error_vec[i] > threshold => pass sub-batch[i] to expert2
                pass_mask = (e1_error_vec > self.threshold_e1).detach().cpu().numpy()
                pass_indices = np.where(pass_mask == True)[0]
                if len(pass_indices) > 0:
                    # build a sub-batch
                    sub_batch = data_batch[pass_indices]
                    self.expert2.optimizer.zero_grad()
                    latent2, recon2 = self.expert2.forward(sub_batch)

                    # measure recon error
                    recon2_val = recon_loss_func(recon2, sub_batch)
                    # stationary
                    sl2, meanL2, stdL2 = self.expert2.stationary_loss(latent2, per_batch=False)
                    loss_e2 = recon2_val + sl2

                    loss_e2.backward()
                    self.expert2.optimizer.step()

                    e2_batch_losses.append(loss_e2.item())
                else:
                    e2_batch_losses.append(0.0)

                # gather e1_error_vec for threshold update
                e1_errors_all.extend(e1_error_vec.detach().cpu().numpy())

            # end of epoch => compute new threshold for e1
            e1_errors_all = np.array(e1_errors_all)
            if len(e1_errors_all) > 0:
                mean_e1 = e1_errors_all.mean()
                std_e1 = e1_errors_all.std()
                self.threshold_e1 = mean_e1 + self.threshold_sigma * std_e1
            else:
                # fallback if no data
                self.threshold_e1 = 9999999

            # average losses
            epoch_e1 = np.mean(e1_batch_losses)
            epoch_e2 = np.mean(e2_batch_losses)
            combined = epoch_e1 + epoch_e2

            print(f"[Epoch {epoch_i+1}/{n_epochs}] e1_loss={epoch_e1:.4f}, e2_loss={epoch_e2:.4f}, threshold_e1={self.threshold_e1:.4f}")

            # simple early stopping
            if combined < best_combined:
                best_combined = combined
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Now compute threshold_e2 from the training set
        self._compute_threshold_e2(train_loader, recon_loss_func)

    def _compute_threshold_e2(self, train_loader, recon_loss_func):
        """
        Gathers windows that pass e1 threshold, measure e2 error, sets threshold_e2.
        """
        e2_error_list = []

        for data_batch, _ in train_loader:
            data_batch = data_batch.to(self.device)
            # forward pass e1
            latent1, recon1 = self.expert1.forward(data_batch)
            if self.loss_name == 'MaxDiff':
                e1_error_vec = (recon1 - data_batch).abs().max(dim=2)[0].max(dim=1)[0]
            else:
                e1_error_vec = torch.mean((recon1 - data_batch) ** 2, dim=(1,2))

            pass_mask = (e1_error_vec > self.threshold_e1).cpu().numpy()
            pass_indices = np.where(pass_mask == True)[0]
            if len(pass_indices) == 0:
                continue
            sub_batch = data_batch[pass_indices]

            # forward pass e2
            latent2, recon2 = self.expert2.forward(sub_batch)
            if self.loss_name == 'MaxDiff':
                e2_sub_vec = (recon2 - sub_batch).abs().max(dim=2)[0].max(dim=1)[0]
            else:
                e2_sub_vec = torch.mean((recon2 - sub_batch) ** 2, dim=(1,2))

            e2_error_list.extend(e2_sub_vec.cpu().numpy())

        if len(e2_error_list) == 0:
            self.threshold_e2 = 9999999
        else:
            arr = np.array(e2_error_list)
            self.threshold_e2 = arr.mean() + self.threshold_sigma * arr.std()

        print(f"[Gating] threshold_e2 = {self.threshold_e2:.4f}")

    ###########################################################################
    # 2) EVALUATE (final gating) on a dataset
    ###########################################################################
    def evaluate(self, data_loader, window_coef=0.2):
        """
        Runs final gating:
          - if e1_error <= threshold_e1 => normal
          - else => pass to e2 => if e2_error > threshold_e2 => anomaly
        We store final errors & preds in self.final_errors, self.final_preds
        """
        errors_list = []
        pred_list   = []
        all_anomalies = []

        inputs_list   = []
        rec_list      = []
        # We'll replicate the logic from each sub-expert's 'predict' approach

        with torch.no_grad():
            for data_batch, anomalies in data_loader:
                data_batch = data_batch.to(self.device)
                # e1 forward
                latent1, recon1 = self.expert1.forward(data_batch)
                if self.loss_name == 'MaxDiff':
                    e1_err_vec = (recon1 - data_batch).abs().max(dim=2)[0].max(dim=1)[0]
                else:
                    e1_err_vec = torch.mean((recon1 - data_batch) ** 2, dim=(1,2))

                # gating
                pass_mask = (e1_err_vec > self.threshold_e1)
                pass_indices = pass_mask.nonzero(as_tuple=True)[0]

                # For sub-batch that passes
                e2_errors = torch.zeros_like(e1_err_vec)
                e2_errors[pass_indices] = -1.0  # placeholder

                if len(pass_indices) > 0:
                    sub_batch = data_batch[pass_indices]
                    latent2, recon2 = self.expert2.forward(sub_batch)
                    if self.loss_name == 'MaxDiff':
                        e2_err_vec = (recon2 - sub_batch).abs().max(dim=2)[0].max(dim=1)[0]
                    else:
                        e2_err_vec = torch.mean((recon2 - sub_batch) ** 2, dim=(1,2))
                    # place them back
                    for i, idx in enumerate(pass_indices):
                        e2_errors[idx] = e2_err_vec[i]

                # final error: if pass => e2_errors, else e1_err_vec
                final_err = []
                final_pred = []
                for i in range(len(e1_err_vec)):
                    if pass_mask[i].item() == 0:
                        # didn't pass => e1
                        fe = e1_err_vec[i].item()
                        final_err.append(fe)
                        # below threshold => normal
                        final_pred.append(0)
                    else:
                        # pass => e2
                        fe = e2_errors[i].item()
                        final_err.append(fe)
                        if fe > self.threshold_e2:
                            final_pred.append(1)
                        else:
                            final_pred.append(0)

                # store
                errors_list.append(np.array(final_err))
                pred_list.append(np.array(final_pred))
                all_anomalies.append(anomalies.numpy())

                # Also store data for plotting if we want
                # We'll just store last item across features dimension
                # since Twin does that in "predict"
                inputs_list.append(data_batch.cpu().numpy()[:, -1, :])  # shape (batch, features)
                rec_list.append(recon1.cpu().numpy()[:, -1, :])         # partial rec from expert1 (but not the final if e2 used)

        self.final_errors = np.concatenate(errors_list)
        self.final_preds  = np.concatenate(pred_list)
        anomalies_np      = np.concatenate(all_anomalies)

        print("[Evaluate] Completed gating approach.")

        return {
            'inputs': np.concatenate(inputs_list),
            'outputs': np.concatenate(rec_list),  # NOTE: Not truly combined e1/e2, for demonstration
            'errors': self.final_errors,
            'predictions': self.final_preds,
            'anomalies': anomalies_np
        }

    ###########################################################################
    # 3) PLOT EXPERT1 ALONE
    ###########################################################################
    def plot_expert1(self, data_loader, train=False, plot_width=800):
        """
        Feeds all data to expert1 alone, ignoring gating, and plots the reconstruction,
        error, threshold, anomalies, etc. akin to the Twin.plot_results method.
        """
        with torch.no_grad():
            inputs_list  = []
            outputs_list = []
            errors_list  = []
            anomaly_list = []
            for data_batch, anomalies in data_loader:
                data_batch = data_batch.to(self.device)
                latent, recon = self.expert1.forward(data_batch)

                # We'll do a "MaxDiff" style if self.loss_name == 'MaxDiff', else MSE
                if self.loss_name == 'MaxDiff':
                    err_vec = (recon - data_batch).abs().max(dim=2)[0].max(dim=1)[0].cpu().numpy()
                else:
                    err_vec = torch.mean((recon - data_batch)**2, dim=(1,2)).cpu().numpy()

                # For plotting, store last step in time
                inputs_list.append(data_batch.cpu().numpy()[:, -1, :])
                outputs_list.append(recon.cpu().numpy()[:, -1, :])
                errors_list.append(err_vec)
                anomaly_list.append(anomalies.numpy())

        inputs = np.concatenate(inputs_list).squeeze()
        outputs = np.concatenate(outputs_list).squeeze()
        errors = np.concatenate(errors_list)
        anomalies = np.concatenate(anomaly_list)

        # If train => we set threshold if not exist
        if train and self.threshold_e1 < 0:
            # e.g. mean + 3 std
            self.threshold_e1 = errors.mean() + 3*errors.std()
        # If not train => we highlight anomalies
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

        # Labeled anomalies
        label_indices = [i for i in range(len(anomalies)) if anomalies[i] == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[inputs[i] for i in label_indices],
                                     mode='markers',
                                     name='Labels',
                                     marker=dict(color='orange', size=7)))

        # Pred anomalies
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
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'),
                          template='plotly',
                          width=plot_width)
        fig.show()

    ###########################################################################
    # 4) PLOT EXPERT2 ALONE
    ###########################################################################
    def plot_expert2(self, data_loader, train=False, plot_width=800):
        """
        Feeds all data to expert2 alone, ignoring gating, and plots everything similarly.
        """
        with torch.no_grad():
            inputs_list  = []
            outputs_list = []
            errors_list  = []
            anomaly_list = []
            for data_batch, anomalies in data_loader:
                data_batch = data_batch.to(self.device)
                latent, recon = self.expert2.forward(data_batch)

                if self.loss_name == 'MaxDiff':
                    err_vec = (recon - data_batch).abs().max(dim=2)[0].max(dim=1)[0].cpu().numpy()
                else:
                    err_vec = torch.mean((recon - data_batch)**2, dim=(1,2)).cpu().numpy()

                inputs_list.append(data_batch.cpu().numpy()[:, -1, :])
                outputs_list.append(recon.cpu().numpy()[:, -1, :])
                errors_list.append(err_vec)
                anomaly_list.append(anomalies.numpy())

        inputs = np.concatenate(inputs_list).squeeze()
        outputs = np.concatenate(outputs_list).squeeze()
        errors = np.concatenate(errors_list)
        anomalies = np.concatenate(anomaly_list)

        if train and self.threshold_e2 < 0.1:
            self.threshold_e2 = errors.mean() + 3*errors.std()
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
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'),
                          template='plotly',
                          width=plot_width)
        fig.show()

    ###########################################################################
    # 5) PLOT FINAL MOE GATING
    ###########################################################################
    def plot_final_moe(self, data_loader, plot_width=800):
        """
        Runs self.evaluate(...) on data_loader to get final_errors, final_preds.
        Plots the final gating-based anomaly detection vs. the true data and labels.
        """
        results = self.evaluate(data_loader)

        inputs = results['inputs'].squeeze()
        outputs= results['outputs'].squeeze()  # note: mostly expert1's recon
        errors = results['errors']
        preds  = results['predictions']
        anomalies = results['anomalies'].squeeze()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(inputs))),
                                 y=inputs,
                                 mode='lines',
                                 name='Data',
                                 line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=list(range(len(outputs))),
                                 y=outputs,
                                 mode='lines',
                                 name='MoE Recon (Expert1 partial)',
                                 line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=list(range(len(errors))),
                                 y=errors,
                                 mode='lines',
                                 name='MoE Final Errors',
                                 line=dict(color='red')))

        # Labeled anomalies
        label_indices = [i for i in range(len(anomalies)) if anomalies[i] == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[inputs[i] for i in label_indices],
                                     mode='markers',
                                     name='Labels',
                                     marker=dict(color='orange', size=7)))

        # Pred anomalies
        pred_indices = [i for i in range(len(preds)) if preds[i] == 1]
        if pred_indices:
            fig.add_trace(go.Scatter(x=pred_indices,
                                     y=[inputs[i] for i in pred_indices],
                                     mode='markers',
                                     name='MoE Anomalies',
                                     marker=dict(color='black', size=7, symbol='x')))

        fig.update_layout(title='Final MoE Gating Results',
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, orientation='h'),
                          template='plotly',
                          width=plot_width)
        fig.show()

    ###########################################################################
    # 6) SAVE AND LOAD
    ###########################################################################
    def save(self, dir_path='torch_moe'):
        os.makedirs(dir_path, exist_ok=True)
        # Save both experts
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
        self.device        = str(meta['device'])
        self.window_size   = int(meta['window_size'])
        self.threshold_sigma = float(meta['threshold_sigma'])
        self.seed          = int(meta['seed'])
        self.loss_name     = str(meta['loss_name'])

        print(f"MoE loaded from {dir_path}. thresholds=({self.threshold_e1:.4f}, {self.threshold_e2:.4f})")
