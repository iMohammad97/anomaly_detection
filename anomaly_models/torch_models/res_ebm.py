import torch
import torch.nn as nn
import torch.fft
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm, trange
import plotly.graph_objects as go

class ResidualEBM(nn.Module):
    def __init__(self, n_features: int = 1, window_size: int = 256, hidden_dim: int = 32, device: str = 'cpu', seed: int = 0):
        super(ResidualEBM, self).__init__()
        torch.manual_seed(seed)
        self.name = 'ResidualEBM'
        self.lr = 0.0001
        self.device = device
        self.n_features = n_features
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        self.energy_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * n_features * window_size, window_size // 2),
            nn.ReLU(),
            nn.Linear(window_size // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Outputs energy
        )

        self.to(device)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.energies = []

        self.threshold = None

    def forward(self, x):
        e = self.energy_net(x)
        return e

    def learn(self, train_loader, network, n_epochs: int, recon_index=None, seed: int = 42):
        torch.manual_seed(seed)
        self.train()
        network.eval()
        for _ in (pbar := trange(n_epochs)):
            energies = []
            for window, a in tqdm(train_loader, leave=False):
                window = window.to(self.device)
                with torch.no_grad():
                    if recon_index:
                        reconstructed_window = network(window)[recon_index]
                    else:
                        reconstructed_window = network(window)
                x = torch.cat([window, reconstructed_window], dim=1)
                energy = self.forward(x)
                loss = torch.mean(energy)  # Minimize energy for training samples
                energies.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            pbar.set_description(f'Energy = {np.mean(energies):.2f}')
            self.energies.append(np.mean(energies))

    def predict(self, data, network, recon_index=None, train: bool = False):
        self.eval()
        network.eval()
        results = {}
        inputs, anomalies, outputs, errors = [], [], [], []
        with torch.no_grad():
            for window, anomaly in data:
                if window.shape[0] == 1:
                    break
                inputs.append(window.squeeze().T[-1])
                anomalies.append(anomaly.squeeze().T[-1])
                window = window.to(self.device)
                if recon_index:
                    reconstructed_window = network(window)[recon_index]
                else:
                    reconstructed_window = network(window)
                x = torch.cat([window, reconstructed_window], dim=1)
                energy = self.forward(x)
                errors.append(energy.cpu().detach().numpy().squeeze())
                outputs.append(reconstructed_window.cpu().detach().numpy().squeeze().T[-1])
        results['inputs'] = np.concatenate(inputs)
        results['anomalies'] = np.concatenate(anomalies)
        results['outputs'] = np.concatenate(outputs)
        energies = np.concatenate(errors)
        energies -= np.min(energies)
        energies /= np.max(energies)
        results['energies'] = energies
        if train and self.threshold is None:
            self.threshold = np.mean(results['energies']) + 3 * np.std(results['energies'])
        elif not train and self.threshold:
            results['predictions'] = [1 if error > self.threshold else 0 for error in results['energies']]
        return results

    def plot_results(self, data, network, recon_index=None, train: bool = False, plot_width: int = 800):
        results = self.predict(data, network, recon_index, train=train)
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
        fig.add_trace(go.Scatter(x=list(range(len(results['energies']))),
                                 y=results['energies'],
                                 mode='lines',
                                 name='Energies',
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
        fig.show()

    def plot_losses(self, fig_size=(10, 6)):
        xs = np.arange(len(self.energies)) + 1
        plt.figure(figsize=fig_size)
        plt.plot(xs, self.energies, label='Energies')
        plt.grid()
        plt.xticks(xs)
        plt.legend()
        plt.show()

    def save(self, path: str = ''):
        """
        Save the model, optimizer state, and training history to a file.
        """
        if path == '':
            path = self.name + '_' + str(len(self.energies)).zfill(3) + '.pth'
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.energies,
            'config': {
                'n_features': self.n_features,
                'window_size': self.window_size,
                'hidden_dim': self.hidden_dim,
                'device': self.device,
                'lr': self.lr,
            }
        }, path)
        print(f'Model saved to path = {path}')

    @staticmethod
    def load(path: str):
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint['config']
        model = ResidualEBM(
            n_features=config['n_features'],
            window_size=config['window_size'],
            hidden_dim=config['hidden_dim'],
            device=config['device']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.energies = checkpoint['losses']

        return model
