import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm, trange
import numpy as np
import plotly.graph_objects as go


class EBM(nn.Module):
    def __init__(self, n_features: int = 1, window_size: int = 64, hidden_dim: int = 64, device: str = 'cpu',
                 seed: int = 0):
        super(EBM, self).__init__()
        torch.manual_seed(seed)
        self.name = 'EBM'
        self.lr = 0.0001
        self.device = device
        self.n_features = n_features
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        # Define a simple feedforward network to represent the energy function
        self.energy_net = nn.Sequential(
            nn.Linear(window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Outputs energy
        )

        self.to(device)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.losses = []

    def forward(self, x):
        x = x.squeeze()
        return self.energy_net(x).squeeze()  # Output energy score

    def learn(self, train_loader, n_epochs: int, seed: int = 42):
        torch.manual_seed(seed)
        self.train()
        for _ in (pbar := trange(n_epochs)):
            energy_values = []
            for data, _ in tqdm(train_loader, leave=False):  # No labels needed
                data = data.to(self.device)
                energy = self.forward(data)
                loss = torch.mean(energy)  # Minimize energy for training samples
                energy_values.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            pbar.set_description(f'Energy Loss = {np.mean(energy_values):.4f}')
            self.losses.append(np.mean(energy_values))

    def predict(self, test_loader):
        self.eval()
        inputs, anomalies, outputs, errors = [], [], [], []
        with torch.no_grad():
            for window, anomaly in test_loader:
                inputs.append(window.squeeze().T[-1])
                anomalies.append(anomaly.squeeze().T[-1])
                window = window.to(self.device)
                energy = self.forward(window)
                outputs.append(energy.cpu().detach().numpy().squeeze())
        inputs = np.concatenate(inputs)
        anomalies = np.concatenate(anomalies)
        outputs = np.concatenate(outputs)
        outputs -= np.min(outputs)
        outputs /= np.max(outputs)
        return inputs, anomalies, outputs

    def plot_losses(self, fig_size=(10, 6)):
        xs = np.arange(len(self.losses)) + 1
        plt.figure(figsize=fig_size)
        plt.plot(xs, self.losses, label='Energy Loss')
        plt.grid()
        plt.xticks(xs)
        plt.legend()
        plt.show()

    def plot_results(self, data, plot_width: int = 800):
        inputs, anomalies, outputs = self.predict(data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(inputs))),
                                 y=inputs,
                                 mode='lines',
                                 name='Test Data',
                                 line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=list(range(len(outputs))),
                                 y=outputs,
                                 mode='lines',
                                 name='Energies',
                                 line=dict(color='purple')))
        label_indices = [i for i in range(len(anomalies)) if anomalies[i] == 1]
        if label_indices:
            fig.add_trace(go.Scatter(x=label_indices,
                                     y=[inputs[i] for i in label_indices],
                                     mode='markers',
                                     name='Labels on Test Data',
                                     marker=dict(color='orange', size=10)))
        fig.update_layout(title='Test Data, Predictions, and Anomalies',
                          xaxis_title='Time Steps',
                          yaxis_title='Value',
                          legend=dict(x=0, y=1, traceorder='normal', orientation='h'),
                          template='plotly',
                          width=plot_width)
        fig.show()

    def save(self, path: str = ''):
        if path == '':
            path = self.name + '_' + str(len(self.losses)).zfill(3) + '.pth'
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'config': {
                'n_features': self.n_features,
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
        model = EBM(
            n_features=config['n_features'],
            hidden_dim=config['hidden_dim'],
            device=config['device']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.losses = checkpoint['losses']

        return model