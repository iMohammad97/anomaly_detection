import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm, trange
import numpy as np
import plotly.graph_objects as go
import torch.nn.functional as F
import os

class AE(nn.Module):
    def __init__(self, n_features: int = 1, window_size: int = 256, latent_dim: int = 32, lstm_units: int = 64, device: str = 'cpu', seed: int = 0):
        super(AE, self).__init__()
        torch.manual_seed(seed)
        self.name = 'AE'
        self.lr = 0.0001
        self.device = device
        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units

        self.encoder_lstm1 = nn.LSTM(n_features, lstm_units, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True)
        self.encoder_lstm3 = nn.LSTM(lstm_units, latent_dim, batch_first=True)

        self.decoder_lstm1 = nn.LSTM(latent_dim, lstm_units, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True)
        self.decoder_lstm3 = nn.LSTM(lstm_units, n_features, batch_first=True)

        self.to(device)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.losses = []
        self.threshold = None

    def encode(self, x):
        # Encode
        x, _ = self.encoder_lstm1(x)
        x, _ = self.encoder_lstm2(x)
        x, _ = self.encoder_lstm3(x)
        z = x[:, -1, :]
        return z

    def decode(self, z):
        # Decode
        latent_repeated = z.unsqueeze(1).repeat(1, self.window_size, 1)
        x, _ = self.decoder_lstm1(latent_repeated)
        x, _ = self.decoder_lstm2(x)
        output, _ = self.decoder_lstm3(x)
        return output

    def forward(self, x):
        latent = self.encode(x)
        output = self.decode(latent)
        return output

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

    def learn(self, train_loader, n_epochs: int, loss_name: str = "MaxDiff", seed: int = 42):
        torch.manual_seed(seed)
        self.train()
        loss_fn = self.select_loss(loss_name)
        for _ in (pbar := trange(n_epochs)):
            recons = []
            for d, a in tqdm(train_loader, leave=False):
                d = d.to(self.device)
                x = self.forward(d)
                loss = loss_fn(x, d)
                recons.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            pbar.set_description(f'{loss_name} Loss = {np.mean(recons):.4f}')
            self.losses.append(np.mean(recons))

    def predict(self, data, train: bool = False):
        self.eval()
        results = {}
        inputs, anomalies, outputs, errors = [], [], [], []
        loss = nn.MSELoss(reduction='none').to(self.device)
        with torch.no_grad():
            for window, anomaly in data:
                if window.shape[0] == 1:
                    break
                inputs.append(window.squeeze().T[-1])
                anomalies.append(anomaly.squeeze().T[-1])
                window = window.to(self.device)
                recons = self.forward(window)
                outputs.append(recons.cpu().detach().numpy().squeeze().T[-1])
                errors.append(loss(window, recons).cpu().detach().numpy().squeeze().T[-1])
        results['inputs'] = np.concatenate(inputs)
        results['anomalies'] = np.concatenate(anomalies)
        results['outputs'] = np.concatenate(outputs)
        results['errors'] = np.concatenate(errors)
        if train:
            self.threshold = np.mean(results['errors']) + 3 * np.std(results['errors'])
        elif not train and self.threshold is not None:
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

    def plot_losses(self, fig_size=(10, 6)):
        xs = np.arange(len(self.losses)) + 1
        plt.figure(figsize=fig_size)
        plt.plot(xs, self.losses, label='Total Loss')
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
            'config': {
                'n_features': self.n_features,
                'window_size': self.window_size,
                'latent_dim': self.latent_dim,
                'lstm_units': self.lstm_units,
                'device': self.device,
                'lr': self.lr,
            }
        }, path)
        print(f'Model saved to path = {path}')

    @staticmethod
    def load(path: str):
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint['config']
        model = AE(
            n_features=config['n_features'],
            window_size=config['window_size'],
            latent_dim=config['latent_dim'],
            lstm_units=config['lstm_units'],
            device=config['device']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.losses = checkpoint['losses']

        return model
