import torch
import torch.nn as nn
import torch.fft
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm, trange
import plotly.graph_objects as go
import torch.nn.functional as F

class Twin(nn.Module):
    def __init__(self, n_features: int = 1, window_size: int = 256, latent_dim: int = 32, device: str = 'cpu', seed: int = 0):
        super(Twin, self).__init__()
        torch.manual_seed(seed)
        self.name = 'Twin'
        self.lr = 0.0001
        self.device = device
        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.mean_coef = 1
        self.std_coef = 1
        lstm_units = latent_dim

        # LSTM Layers
        self.encoder_lstm1 = nn.LSTM(n_features, lstm_units, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True)
        self.encoder_lstm3 = nn.LSTM(lstm_units, latent_dim//2, batch_first=True)
        self.decoder_lstm1 = nn.LSTM(latent_dim//2, lstm_units, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True)
        self.decoder_lstm3 = nn.LSTM(lstm_units, n_features, batch_first=True)

        # Fourier Layers
        self.encoder_fc1 = nn.Linear(2 * n_features * window_size, 128)
        self.encoder_fc2 = nn.Linear(128, 64)
        self.encoder_fc3 = nn.Linear(64, latent_dim)
        self.decoder_fc1 = nn.Linear(latent_dim, 64)
        self.decoder_fc2 = nn.Linear(64, 128)
        self.decoder_fc3 = nn.Linear(128, 2 * n_features * window_size)

        self.to(device)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.losses = []
        self.recon_losses = []
        self.mean_losses = []
        self.std_losses = []

        self.threshold = None

    def freq_encode(self, x):
        # Flatten the input for linear layers
        x = x.view(x.size(0), -1)

        # Apply FFT at the beginning
        x_complex = torch.fft.fft(x, dim=1)

        # Concatenate real and imaginary parts
        x = torch.cat([x_complex.real, x_complex.imag], dim=1)

        # Encode
        x = torch.relu(self.encoder_fc1(x))
        x = torch.relu(self.encoder_fc2(x))
        latent = torch.relu(self.encoder_fc3(x))

        return latent

    def freq_decode(self, z):
        x = torch.relu(self.decoder_fc1(z))
        x = torch.relu(self.decoder_fc2(x))
        x = self.decoder_fc3(x)

        # Split into real and imaginary parts
        half_size = x.shape[1] // 2

        # Reconstruct complex tensor
        x_complex = torch.complex( x[:, :half_size], x[:, half_size:])

        # Apply IFFT at the end
        x = torch.fft.ifft(x_complex, dim=1).real

        # Reshape back to original shape
        x = x.view(x.size(0), self.window_size, self.n_features)
        return x

    def time_forward(self, x):
        # Encode
        x, _ = self.encoder_lstm1(x)
        x, _ = self.encoder_lstm2(x)
        x, _ = self.encoder_lstm3(x)
        latent = x[:, -1, :]

        # Decode
        latent_repeated = latent.unsqueeze(1).repeat(1, self.latent_dim, 1)
        x, _ = self.decoder_lstm1(latent_repeated)
        x, _ = self.decoder_lstm2(x)
        output, _ = self.decoder_lstm3(x)

        return output, latent

    def freq_forward(self, x):
        latent = self.freq_encode(x)
        # Decode
        x = self.freq_decode(latent)
        return latent, x

    def forward(self, x):
        x_lstm = x[:, -self.latent_dim:]
        _, y = self.freq_forward(x)
        z, latent = self.time_forward(x_lstm)
        y[:, -self.latent_dim:] += z
        return latent, y

    def select_loss(self, loss_name: str):
        if loss_name == "MSE":
            return nn.MSELoss(reduction='mean').to(self.device)
        elif loss_name == "Huber":
            return nn.SmoothL1Loss(reduction='mean').to(self.device)
        elif loss_name == "MaxDiff":
            return lambda inputs, target: torch.max(torch.abs(inputs - target))
        elif loss_name == "MSE_R2":
            return lambda inputs, target: (F.mse_loss(inputs, target, reduction='mean') + (1 - (1 - torch.sum((target - inputs) ** 2) / (torch.sum((target - torch.mean(target)) ** 2) + 1e-10)))) / 2
        else:
            raise ValueError("Unsupported loss function")

    def learn(self, train_loader, n_epochs: int, seed: int = 42, loss_name1: str = 'MSE_R2', loss_name2: str = 'MaxDiff'):
        torch.manual_seed(seed)
        self.train()
        recon_loss1 = self.select_loss(loss_name1)
        recon_loss2 = self.select_loss(loss_name2)
        for _ in (pbar := trange(n_epochs)):
            recons, means, stds = [], [], []
            for d, a in tqdm(train_loader, leave=False):
                d = d.to(self.device)
                latent, x = self.forward(d)
                recon = recon_loss1(x, d) + recon_loss2(x[:, -self.latent_dim:], d[:, -self.latent_dim:])
                _, mean, std = self.stationary_loss(latent)
                loss = recon + mean + std
                recons.append(recon.item()), means.append(mean.item()), stds.append(std.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            rl, ml, sl = np.mean(recons), np.mean(means), np.mean(stds)
            pbar.set_description(f'{loss_name1} + {loss_name2} Loss = {rl:.4f}, Avg Loss = {ml:.4f}, STD Loss = {sl:.4f} ')
            self.recon_losses.append(rl), self.mean_losses.append(ml)
            self.std_losses.append(sl), self.losses.append(rl + ml + sl)

    def predict(self, data, train: bool = False, window_coef: float = 0.2):
        self.eval()
        results = {}
        inputs, anomalies, outputs, rec_errors = [], [], [], []
        mean_errors, std_errors = [], []
        mse = nn.MSELoss(reduction='none').to(self.device)
        with torch.no_grad():
            for window, anomaly in data:
                if window.shape[0] == 1:
                    break
                inputs.append(window.squeeze().T[-1])
                anomalies.append(anomaly.squeeze().T[-1])
                window = window.to(self.device)
                latent, recons = self.forward(window)
                outputs.append(recons.cpu().detach().numpy().squeeze().T[-1])
                rec_error = mse(window, recons)
                rec_error = rec_error[:, -1].cpu().detach().numpy().squeeze().T + window_coef * torch.mean(rec_error, dim=1).cpu().detach().numpy().squeeze().T
                rec_errors.append(rec_error)
                _, mean, std = self.stationary_loss(latent, per_batch=True)
                mean_errors.append(mean.cpu().detach().numpy().squeeze())
                std_errors.append(std.cpu().detach().numpy().squeeze())
        results['inputs'] = np.concatenate(inputs)
        results['anomalies'] = np.concatenate(anomalies)
        results['outputs'] = np.concatenate(outputs)
        results['errors'] = np.concatenate(rec_errors)
        results['means'] = np.concatenate(mean_errors)
        results['stds'] = np.concatenate(std_errors)
        if train and self.threshold is None:
            self.threshold = np.mean(results['errors']) + 3 * np.std(results['errors'])
        elif not train and self.threshold:
            results['predictions'] = [1 if error > self.threshold else 0 for error in results['errors']]
        return results

    def plot_results(self, data, train: bool = False, plot_width: int = 800, save_path=None, file_format='html'):
        results = self.predict(data, train=train)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(results['inputs']))),
                                 y=results['inputs'],
                                 mode='lines',
                                 name='Test Data',
                                 line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=list(range(len(results['outputs']))),
                                 y=results['outputs'],
                                 mode='lines',
                                 name='Predictions',
                                 line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=list(range(len(results['errors']))),
                                 y=results['errors'],
                                 mode='lines',
                                 name='Anomaly Errors',
                                 line=dict(color='red')))
        fig.add_trace(go.Scatter(x=list(range(len(results['means']))),
                                 y=results['means'],
                                 mode='lines',
                                 name='Mean Errors',
                                 line=dict(color='pink')))
        fig.add_trace(go.Scatter(x=list(range(len(results['stds']))),
                                 y=results['stds'],
                                 mode='lines',
                                 name='STD Errors',
                                 line=dict(color='green')))
        label_indices = [i for i in range(len(results['anomalies'])) if results['anomalies'][i] == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[results['inputs'][i] for i in label_indices],
                                     mode='markers',
                                     name='Labels on Test Data',
                                     marker=dict(color='orange', size=10)))
        if self.threshold is not None and not train:
            label_indices = [i for i in range(len(results['anomalies'])) if results['predictions'][i] == 1]
            fig.add_hline(y=self.threshold, line_dash='dash', name='Threshold')
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[results['inputs'][i] for i in label_indices],
                                     mode='markers',
                                     name='Predictions on Test Data',
                                     marker=dict(color='black', size=7, symbol='x')))
        fig.update_layout(title='Test Data, Predictions, and Anomalies',
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
                          template='plotly',
                          width=plot_width)
        # Optionally save the figure
        if save_path is not None:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            if file_format.lower() == 'html':
                # Save as interactive HTML
                fig.write_html(save_path)
            else:
                # Save as static image (requires kaleido or orca)
                fig.write_image(save_path, format=file_format)

            print(f"Plot saved to: {save_path}")
        fig.show()

    def stationary_loss(self, latent, per_batch: bool = False):
        # Compute mean and std over batch and sequence dimensions
        latent_avg = torch.mean(latent, dim=1)  # Averaging over batch and sequence
        mean_loss = torch.square(latent_avg) # Enforce near-zero mean

        latent_std = torch.std(latent, dim=1)  # Compute std over batch and sequence
        std_loss = torch.abs(latent_std - 1.0) # Encourage unit variance

        if not per_batch:
            mean_loss = torch.mean(mean_loss)
            std_loss = torch.mean(std_loss)

        loss = self.mean_coef * mean_loss + self.std_coef * std_loss
        return loss, mean_loss, std_loss

    def plot_losses(self, fig_size=(10, 6), save_path=None):
        xs = np.arange(len(self.losses)) + 1
        plt.figure(figsize=fig_size)
        plt.plot(xs, self.losses, label='Total Loss')
        plt.plot(xs, self.recon_losses, label='REC Losses')
        plt.plot(xs, self.mean_losses, label='AVG Losses')
        plt.plot(xs, self.std_losses, label='STD Losses')
        plt.grid()
        plt.xticks(xs)
        plt.legend()
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
            # Save the figure
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def save(self, path: str = ''):
        """
        Save the model, optimizer state, and training history to a file.
        """
        if path == '':
            path = self.name + '_' + str(len(self.losses)).zfill(3) + '.pth'
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'config': {
                'n_features': self.n_features,
                'window_size': self.window_size,
                'latent_dim': self.latent_dim,
                'device': self.device,
                'lr': self.lr,
            }
        }, path)
        print(f'Model saved to path = {path}')

    @staticmethod
    def load(path: str):
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint['config']
        model = Twin(
            n_features=config['n_features'],
            window_size=config['window_size'],
            latent_dim=config['latent_dim'],
            device=config['device']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.losses = checkpoint['losses']

        return model
