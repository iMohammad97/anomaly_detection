import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm, trange
import numpy as np
import plotly.graph_objects as go
import math

'''
Something needs to be changed
'''


class TransformerVAE(nn.Module):
    def __init__(self, n_features: int = 1, window_size: int = 256, d_model: int = 64, nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 256, dropout: float = 0.1, device: str = 'cpu', seed: int = 0):
        super(TransformerVAE, self).__init__()
        torch.manual_seed(seed)
        self.name = 'TransformerVAE'
        self.lr = 0.0001
        self.device = device
        self.n_features = n_features
        self.window_size = window_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=window_size)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_layers
        )

        # Latent space
        self.fc_mu = nn.Linear(d_model * window_size, d_model)
        self.fc_logvar = nn.Linear(d_model * window_size, d_model)
        self.fc_latent_to_features = nn.Linear(d_model, d_model * window_size)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_layers
        )

        self.output_projection = nn.Linear(d_model, n_features)

        # Learnable start token for decoder input
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.to(device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.recons_losses = []
        self.latent_losses = []

    def encode(self, x):
        batch_size = x.size(0)
        x = self.input_projection(x).permute(1, 0, 2)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x.permute(1, 0, 2).reshape(batch_size, -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, batch_size):
        x = self.fc_latent_to_features(z).view(batch_size, self.window_size, self.d_model)
        x = x.permute(1, 0, 2)
        start_token = self.start_token.expand(1, batch_size, -1)
        tgt = torch.zeros_like(x).to(x.device)
        tgt[0] = start_token
        x = self.decoder(tgt, x).permute(1, 0, 2)
        return self.output_projection(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, x.size(0))
        return x_recon, mu, logvar

    def latent_loss(self, mu, logvar, per_batch=False):
        if per_batch:
            return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_divergence

    def select_loss(self, loss_name: str):
        if loss_name == "MSE":
            return nn.MSELoss(reduction='mean').to(self.device)
        elif loss_name == "Huber":
            return nn.SmoothL1Loss(reduction='mean').to(self.device)
        elif loss_name == "MaxDiff":
            return lambda inputs, target: torch.max(torch.abs(inputs - target))
        else:
            raise ValueError("Unsupported loss function")

    def learn(self, train_loader, n_epochs: int, loss_name: str = "MaxDiff", seed: int = 42):
        torch.manual_seed(seed)
        self.train()
        loss_fn = self.select_loss(loss_name)
        for _ in (pbar := trange(n_epochs)):
            recons = []
            kls = []
            for d, a in (p := tqdm(train_loader, leave=False)):
                d = d.to(self.device)
                x, mu, var = self.forward(d)
                recon_loss = loss_fn(x, d)
                kld_loss = self.latent_loss(mu, var)
                recons.append(recon_loss.item())
                kls.append(kld_loss.item())
                total_loss = recon_loss + kld_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                p.set_description(f'Batch {loss_name} Loss = {recons[-1]:.4f}, KL Loss = {kls[-1]:.4f}')
            pbar.set_description(f'{loss_name} Loss = {np.mean(recons):.4f}, KL Loss = {np.mean(kls):.4f}')
            self.recons_losses.append(np.mean(recons))
            self.latent_losses.append(np.mean(kls))

    def predict(self, data):
        inputs, anomalies, outputs, rec_errors, kld_errors = [], [], [], [], []
        loss = nn.MSELoss(reduction='none').to(self.device)
        for window, anomaly in data:
            inputs.append(window.squeeze().T[-1])
            anomalies.append(anomaly.squeeze().T[-1])
            window = window.to(self.device)
            recons, mu, var = self.forward(window)
            outputs.append(recons.cpu().detach().numpy().squeeze().T[-1])
            rec_errors.append(loss(window, recons).cpu().detach().numpy().squeeze().T[-1])
            kld = self.latent_loss(mu, var, per_batch=True)
            kld_errors.append(kld.cpu().detach().numpy().squeeze().T[-1])
        inputs = np.concatenate(inputs)
        anomalies = np.concatenate(anomalies)
        outputs = np.concatenate(outputs)
        rec_errors = np.concatenate(rec_errors)
        kld_errors = np.concatenate(kld_errors)
        return inputs, anomalies, outputs, rec_errors, kld_errors

    def plot_results(self, data, plot_width: int = 800):
        inputs, anomalies, outputs, rec_errors, kld_errors = self.predict(data)
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
        fig.add_trace(go.Scatter(x=list(range(len(rec_errors))),
                                 y=rec_errors,
                                 mode='lines',
                                 name='Reconstruction Errors',
                                 line=dict(color='red')))
        fig.add_trace(go.Scatter(x=list(range(len(kld_errors))),
                                 y=kld_errors,
                                 mode='lines',
                                 name='KL-Divergence Errors',
                                 line=dict(color='pink')))
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
        xs = np.arange(len(self.recons_losses)) + 1
        plt.figure(figsize=fig_size)
        plt.plot(xs, self.recons_losses, label='Reconstruction Loss')
        plt.plot(xs, self.latent_losses, label='KL-Divergence Loss')
        plt.grid()
        plt.xticks(xs)
        plt.legend()
        plt.show()

    def save(self, path: str = ''):
        """
        Save the model, optimizer state, and training history to a file.
        """
        if path == '':
            path = self.name + '_' + str(len(self.recons_losses)).zfill(3) + '.pth'
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.recons_losses,
            'config': {
                'n_features': self.n_features,
                'window_size': self.window_size,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dim_feedforward': self.dim_feedforward,
                'dropout': self.dropout,
                'device': self.device,
                'lr': self.lr,
            }
        }, path)
        print(f'Model saved to path = {path}')

    @staticmethod
    def load(path: str):
        checkpoint = torch.load(path)
        config = checkpoint['config']
        model = TransformerVAE(
            n_features=config['n_features'],
            window_size=config['window_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            device=config['device']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.recons_losses = checkpoint['losses']

        return model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(1)  # Shape: (max_len, 1, d_model)

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :].to(x.device)