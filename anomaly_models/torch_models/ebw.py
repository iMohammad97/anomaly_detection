import pywt
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm, trange
import plotly.graph_objects as go
from matplotlib import pyplot as plt

class EnergyBasedWavelet(nn.Module):
    def __init__(self, window_size: int , feats: int = 1, device: str = 'cpu', seed: int = 0):
        super(EnergyBasedWavelet, self).__init__()
        torch.manual_seed(seed)
        self.name = 'EnergyBasedWavelet'
        self.lr = 0.001
        self.device = device
        self.n_feats = feats
        self.n_window = window_size

        self.transforms = ['cgau1', 'cgau4', 'cgau8', 'cmor1.5-0.5', 'fbsp',
                           'gaus1', 'gaus4', 'gaus8', 'mexh', 'morl', 'shan']
        self.scales = np.arange(1, 1 + window_size)
        channels = len(self.transforms) * self.n_feats

        self.energy_net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=3, padding=1),
            nn.ReLU(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=channels * 2, out_channels=channels * 4, kernel_size=3, padding=1),
            nn.ReLU(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=channels * 4, out_channels=channels * 8, kernel_size=3, padding=1),
            nn.ReLU(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=channels * 8, out_channels=channels * 16, kernel_size=3, padding=1),
            nn.ReLU(), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(channels * window_size * len(self.scales) // 16, 1)  # Outputs energy
        )

        self.to(device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.losses = []
        self.threshold = None

    def apply_wavelet(self, window):
        window_size, dims = window.shape
        wavelet_window = np.empty(shape=(dims * len(self.transforms), len(self.scales), window_size))
        for d in range(dims):
            for i, w in enumerate(self.transforms):
                coefficients, frequencies = pywt.cwt(np.array(window[:, d]), self.scales, w)
                wavelet_window[d * len(self.transforms) + i] = coefficients
        return torch.from_numpy(wavelet_window).type(torch.float)

    def apply_wavelet_to_batch(self, window):
        batch_size, window_size, dims = window.shape
        wavelet_window = np.empty(shape=(batch_size, dims * len(self.transforms), len(self.scales), window_size))
        for b in range(batch_size):
            wavelet_window[b] = self.apply_wavelet(window[b])
        return torch.from_numpy(wavelet_window).type(torch.float)

    def forward(self, x):
        return self.energy_net(x)

    def learn(self, train_loader, n_epochs: int, seed: int = 42):
        torch.manual_seed(seed)
        wavelet_dataset = WaveletTransformDataset(train_loader, self.scales, self.transforms)
        wave_loader = DataLoader(wavelet_dataset, batch_size=train_loader.batch_size)
        self.train()
        for _ in (pbar := trange(n_epochs)):
            energy_values = []
            for data, _ in tqdm(wave_loader, leave=False):  # No labels needed
                data = data.to(self.device)
                energy = self.forward(data)
                loss = torch.mean(energy)  # Minimize energy for training samples
                energy_values.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            pbar.set_description(f'Energy Loss = {np.mean(energy_values):.4f}')
            self.losses.append(np.mean(energy_values))

    def predict(self, loader, train: bool = False):
        self.eval()
        results = {}
        inputs, anomalies, outputs = [], [], []
        with torch.no_grad():
            for window, anomaly in loader:
                if window.shape[0] == 1:
                    break
                inputs.append(window.squeeze().T[-1])
                anomalies.append(anomaly.squeeze().T[-1])
                window = self.apply_wavelet_to_batch(window).to(self.device)
                energy = self.forward(window)
                outputs.append(energy.cpu().detach().numpy().squeeze())
        results['inputs'] = np.concatenate(inputs)
        results['anomalies'] = np.concatenate(anomalies)
        outputs = np.concatenate(outputs)
        outputs -= np.min(outputs)
        outputs /= np.max(outputs)
        results['outputs'] = outputs
        if train:
            self.threshold = np.mean(results['outputs']) + 3 * np.std(results['outputs'])
        elif self.threshold and not train:
            results['predictions'] = [1 if energy > self.threshold else 0 for energy in results['outputs']]
        return results

    def plot_losses(self, fig_size=(10, 6)):
        xs = np.arange(len(self.losses)) + 1
        plt.figure(figsize=fig_size)
        plt.plot(xs, self.losses, label='Energy Loss')
        plt.grid()
        plt.xticks(xs)
        plt.legend()
        plt.show()

    def plot_results(self, data, train: bool = False, plot_width: int = 800):
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
                                 name='Energies',
                                 line=dict(color='purple')))
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
                'feats': self.n_feats,
                'window_size': self.n_window,
                'device': self.device,
            }
        }, path)
        print(f'Model saved to path = {path}')

    @staticmethod
    def load(path: str):
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint['config']
        model = EnergyBasedWavelet(
            window_size=config['window_size'],
            feats=config['feats'],
            device=config['device']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.losses = checkpoint['losses']

        return model


class WaveletTransformDataset(Dataset):
    def __init__(self, original_dataloader, scales, transforms):
        self.original_dataloader = original_dataloader
        self.scales = scales
        self.transforms = transforms
        self.data = []
        self.labels = []
        self._apply_wavelet_transform()

    def _apply_wavelet_transform(self):
        for batch, labels in tqdm(self.original_dataloader, leave=False, desc='Building the Dataset'):
            batch_size, window_size, dims = batch.shape
            for b in range(batch_size):
                self.labels.append(labels[b])
                wavelet_window = np.empty((dims * len(self.transforms), len(self.scales), window_size))
                for d in range(dims):
                    for i, w in enumerate(self.transforms):
                        signal = np.array(batch[b, :, d])
                        coefficients, _ = pywt.cwt(signal, self.scales, w)
                        wavelet_window[d * len(self.transforms) + i] = coefficients
                self.data.append(wavelet_window)
        self.data = torch.from_numpy(np.array(self.data)).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]