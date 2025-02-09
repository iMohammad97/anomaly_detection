import torch
import torch.nn as nn
from tqdm.notebook import tqdm, trange
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt

class SAE(nn.Module):
    def __init__(self, n_features: int = 1, window_size: int = 256, latent_dim: int = 32, lstm_units: int = 64,
                 mean_coef: float = 1, std_coef: float = 1, device: str = 'cpu', seed: int = 0):
        super(SAE, self).__init__()
        torch.manual_seed(seed)
        self.name = 'SAE'
        self.lr = 0.0001
        self.device = device
        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.mean_coef = mean_coef
        self.std_coef = std_coef
        # self.stationary_loss = StationaryLoss(mean_coef, std_coef)

        self.encoder_lstm1 = nn.LSTM(n_features, lstm_units, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True)
        self.encoder_lstm3 = nn.LSTM(lstm_units, latent_dim, batch_first=True)

        self.decoder_lstm1 = nn.LSTM(latent_dim, lstm_units, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True)
        self.decoder_lstm3 = nn.LSTM(lstm_units, n_features, batch_first=True)

        self.to(device)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.losses = []
        self.recon_losses = []
        self.mean_losses = []
        self.std_losses = []

        self.threshold = None

    def forward(self, x):
        # Encode
        x, _ = self.encoder_lstm1(x)
        x, _ = self.encoder_lstm2(x)
        x, _ = self.encoder_lstm3(x)
        latent = x[:, -1, :]

        # Apply custom loss to the latent space
        # latent_with_loss, _ = self.stationary_loss(latent)

        # Decode
        latent_repeated = latent.unsqueeze(1).repeat(1, self.window_size, 1)
        x, _ = self.decoder_lstm1(latent_repeated)
        x, _ = self.decoder_lstm2(x)
        output, _ = self.decoder_lstm3(x)

        return output, latent

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

    def select_loss(self, loss_name: str):
        if loss_name == "MSE":
            return nn.MSELoss(reduction='mean').to(self.device)
        elif loss_name == "Huber":
            return nn.SmoothL1Loss(reduction='mean').to(self.device)
        elif loss_name == "MaxDiff":
            return lambda inputs, target: torch.max(torch.abs(inputs - target))
        else:
            raise ValueError("Unsupported loss function")

    def learn(self, train_loader, n_epochs: int, seed: int = 42, loss_name: str = 'MaxDiff'):
        torch.manual_seed(seed)
        self.train()
        recon_loss = self.select_loss(loss_name)
        for _ in (pbar := trange(n_epochs)):
            recons, means, stds = [], [], []
            for d, a in tqdm(train_loader, leave=False):
                d = d.to(self.device)
                x, latent = self.forward(d)
                recon = recon_loss(x, d)
                # mean = self.stationary_loss.mse_loss
                # std = self.stationary_loss.std_loss
                _, mean, std = self.stationary_loss(latent)
                loss = recon + mean + std
                recons.append(recon.item()), means.append(mean.item()), stds.append(std.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            rl, ml, sl = np.mean(recons), np.mean(means), np.mean(stds)
            pbar.set_description(f'Rec Loss = {rl:.4f}, Avg Loss = {ml:.4f}, STD Loss = {sl:.4f} ')
            self.recon_losses.append(rl), self.mean_losses.append(ml)
            self.std_losses.append(sl), self.losses.append(rl + ml + sl)

    def predict(self, data, train: bool = False):
        self.eval()
        results = {}
        inputs, anomalies, outputs, rec_errors = [], [], [], []
        mean_errors, std_errors = [], []
        loss = nn.MSELoss(reduction='none').to(self.device)
        with torch.no_grad():
            for window, anomaly in data:
                if window.shape[0] == 1:
                    break
                inputs.append(window.squeeze().T[-1])
                anomalies.append(anomaly.squeeze().T[-1])

                window = window.to(self.device)
                recons, latent = self.forward(window)

                outputs.append(recons.cpu().detach().numpy().squeeze().T[-1])

                rec_error = loss(window, recons).cpu().detach().numpy().squeeze().T[-1]
                rec_errors.append(rec_error)

                _, mean, std = self.stationary_loss(latent, per_batch=True)
                mean_errors.append(mean.cpu().detach().numpy().squeeze())
                std_errors.append(std.cpu().detach().numpy().squeeze())

        # Concatenate safely, preserving the batch structure
        results['inputs'] = np.concatenate(inputs)
        results['anomalies'] = np.concatenate(anomalies)
        results['outputs'] = np.concatenate(outputs)
        results['errors'] = np.concatenate(rec_errors)
        results['means'] = np.concatenate(mean_errors)
        results['stds'] = np.concatenate(std_errors)
        if train:
            self.threshold = np.mean(results['errors']) + 3 * np.std(results['errors'])
        elif not train and self.threshold is not None:
            results['predictions'] = [1 if error > self.threshold else 0 for error in results['errors']]
        return results

    def plot_results(self, data, train: bool = False, plot_width: int = 800):
        results = self.predict(data, train)
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
                                 name='Reconstruction Errors',
                                 line=dict(color='red')))
        fig.add_trace(go.Scatter(x=list(range(len(results['means']))),
                                 y=results['errors'],
                                 mode='lines',
                                 name='Mean Errors',
                                 line=dict(color='pink')))
        fig.add_trace(go.Scatter(x=list(range(len(results['stds']))),
                                 y=results['errors'],
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
            fig.add_hline(y=self.threshold, name='Threshold')
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
        fig.show()

    def plot_losses(self, fig_size=(10, 6)):
        xs = np.arange(len(self.losses)) + 1
        plt.figure(figsize=fig_size)
        plt.plot(xs, self.losses, label='Total Loss')
        plt.plot(xs, self.recon_losses, label='REC Losses')
        plt.plot(xs, self.mean_losses, label='AVG Losses')
        plt.plot(xs, self.std_losses, label='STD Losses')
        plt.grid()
        plt.xticks(xs)
        plt.legend()
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
            'recon_losses': self.recon_losses,
            'mean_losses': self.mean_losses,
            'std_losses': self.std_losses,
            'config': {
                'n_features': self.n_features,
                'window_size': self.window_size,
                'latent_dim': self.latent_dim,
                'lstm_units': self.lstm_units,
                'mean_coef': self.mean_coef,
                'std_coef': self.std_coef,
                'device': self.device,
                'lr': self.lr,
            }
        }, path)
        print(f'Model saved to path = {path}')

    @staticmethod
    def load(path: str):
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint['config']
        model = SAE(
            n_features=config['n_features'],
            window_size=config['window_size'],
            latent_dim=config['latent_dim'],
            lstm_units=config['lstm_units'],
            mean_coef=config['mean_coef'],
            std_coef=config['std_coef'],
            device=config['device']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.losses = checkpoint['losses']
        model.recon_losses = checkpoint['recon_losses']
        model.mean_losses = checkpoint['mean_losses']
        model.std_losses = checkpoint['std_losses']

        return model


class StationaryLoss(nn.Module):
    def __init__(self, mean_coef: float = 1, std_coef: float = 1):
        super(StationaryLoss, self).__init__()
        self.mean_coef = mean_coef
        self.std_coef = std_coef
        self.mse_loss = 0
        self.std_loss = 0

    def forward(self, latent):
        # Calculate the average of the latent space
        latent_avg = torch.mean(latent, dim=0)
        mse_loss = torch.mean(torch.abs(latent_avg))

        # Calculate the standard deviation of the latent space
        latent_std = torch.std(latent, dim=0)
        std_loss = torch.mean(torch.abs(latent_std - 1.0))

        # Store the losses separately for logging
        self.mse_loss = self.mean_coef * mse_loss
        self.std_loss = self.std_coef * std_loss

        # Add losses to the final loss
        loss = self.mse_loss + self.std_loss

        return latent, loss