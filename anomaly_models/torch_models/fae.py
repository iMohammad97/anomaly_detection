import torch
import torch.nn as nn
import torch.fft
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm, trange
import plotly.graph_objects as go

class FAE(nn.Module):
    def __init__(self, n_features: int = 1, window_size: int = 256, latent_dim: int = 32, device: str = 'cpu', seed: int = 0):
        super(FAE, self).__init__()
        torch.manual_seed(seed)
        self.name = 'FAE'
        self.lr = 0.0001
        self.device = device
        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim

        self.encoder_fc1 = nn.Linear(n_features * window_size, 128)
        self.encoder_fc2 = nn.Linear(128, 64)
        self.encoder_fc3 = nn.Linear(64, latent_dim)

        self.decoder_fc1 = nn.Linear(latent_dim, 64)
        self.decoder_fc2 = nn.Linear(64, 128)
        self.decoder_fc3 = nn.Linear(128, n_features * window_size)

        self.to(device)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.losses = []

    def encode(self, x):
        # Flatten the input for linear layers
        x = x.view(x.size(0), -1)

        # Apply FFT at the beginning
        x = torch.fft.fft(x, dim=1).real

        # Encode
        x = torch.relu(self.encoder_fc1(x))
        x = torch.relu(self.encoder_fc2(x))
        latent = torch.relu(self.encoder_fc3(x))
        return latent

    def decode(self, z):
        x = torch.relu(self.decoder_fc1(z))
        x = torch.relu(self.decoder_fc2(x))
        x = self.decoder_fc3(x)

        # Apply IFFT at the end
        x = torch.fft.ifft(x, dim=1).real

        # Reshape back to original shape
        x = x.view(x.size(0), self.window_size, self.n_features)
        return x

    def forward(self, x):
        latent = self.encode(x)
        # Decode
        x = self.decode(latent)
        return latent, x

    def learn(self, train_loader, n_epochs: int, seed: int = 42):
        torch.manual_seed(seed)
        self.train()
        mse = nn.MSELoss(reduction='mean').to(self.device)
        for _ in (pbar := trange(n_epochs)):
            recons = []
            for d, a in tqdm(train_loader, leave=False):
                d = d.to(self.device)
                _, x = self.forward(d)
                loss = mse(x, d)
                recons.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            pbar.set_description(f'MSE Loss = {np.mean(recons):.4f}')
            self.losses.append(np.mean(recons))

    def predict(self, data):
        inputs, anomalies, outputs, errors = [], [], [], []
        mse = nn.MSELoss(reduction='none').to(self.device)
        for window, anomaly in data:
            inputs.append(window.squeeze().T[-1])
            anomalies.append(anomaly.squeeze().T[-1])
            window = window.to(self.device)
            _, recons = self.forward(window)
            outputs.append(recons.cpu().detach().numpy().squeeze().T[-1])
            errors.append(mse(window, recons).cpu().detach().numpy().squeeze().T[-1])
        inputs = np.concatenate(inputs)
        anomalies = np.concatenate(anomalies)
        outputs = np.concatenate(outputs)
        errors = np.concatenate(errors)
        return inputs, anomalies, outputs, errors

    def plot_results(self, data, plot_width: int = 800):
        inputs, anomalies, outputs, errors = self.predict(data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(inputs))),
                                 y=inputs,
                                 mode='lines',
                                 name='Test Data',
                                 line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=list(range(len(outputs))),
                                 y=outputs,
                                 mode='lines',
                                 name='Predictions',
                                 line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=list(range(len(errors))),
                                 y=errors,
                                 mode='lines',
                                 name='Anomaly Errors',
                                 line=dict(color='red')))
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
                'device': self.device,
                'lr': self.lr,
            }
        }, path)
        print(f'Model saved to path = {path}')

    @staticmethod
    def load(path: str):
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint['config']
        model = FAE(
            n_features=config['n_features'],
            window_size=config['window_size'],
            latent_dim=config['latent_dim'],
            device=config['device']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.losses = checkpoint['losses']

        return model
