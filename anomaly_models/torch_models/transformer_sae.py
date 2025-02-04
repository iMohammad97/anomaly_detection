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

class TransformerSAE(nn.Module):
    def __init__(self, n_features: int = 1, window_size: int = 256,  mean_coef: float = 1, std_coef: float = 1, d_model: int = 64, nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 256, dropout: float = 0.1, device: str = 'cpu', seed: int = 0):
        super(TransformerSAE, self).__init__()
        torch.manual_seed(seed)
        self.name = 'TransformerSAE'
        self.lr = 0.0001
        self.device = device
        self.n_features = n_features
        self.window_size = window_size
        self.mean_coef = mean_coef
        self.std_coef = std_coef
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

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_layers
        )

        self.output_projection = nn.Linear(d_model, n_features)

        # Learnable start token for decoder input
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.to(device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.recon_losses = []
        self.mean_losses = []
        self.std_losses = []

    def stationary_loss(self, latent, per_batch: bool = False):
        # Compute mean and std over batch and sequence dimensions
        latent_avg = torch.mean(latent, dim=(0, 1))  # Averaging over batch and sequence
        mean_loss = torch.square(latent_avg) # Enforce near-zero mean

        latent_std = torch.std(latent, dim=(0, 1))  # Compute std over batch and sequence
        std_loss = torch.abs(latent_std - 1.0) # Encourage unit variance

        if not per_batch:
            mean_loss = torch.mean(mean_loss)
            std_loss = torch.mean(std_loss)

        loss = self.mean_coef * mean_loss + self.std_coef * std_loss
        return loss, mean_loss, std_loss

    def forward(self, x):
        batch_size = x.size(0)

        # Input projection and positional encoding
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)

        # Encoder
        memory = self.encoder(x)

        # Decoder input: start token followed by zeros
        start_token = self.start_token.expand(1, batch_size, -1)  # (1, batch_size, d_model)
        tgt = torch.zeros_like(x).to(x.device)  # (seq_len, batch_size, d_model)
        tgt[0] = start_token  # Insert start token at the beginning

        # Decoder
        output = self.decoder(tgt, memory)
        output = output.permute(1, 0, 2)  # (batch_size, seq_len, d_model)

        # Output projection
        output = self.output_projection(output)
        return output, memory

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
            means, stds = [], []
            for d, a in (p := tqdm(train_loader, leave=False)):
                d = d.to(self.device)
                x, m = self.forward(d)
                recon = (loss_fn(x, d))
                stat, mean, std =  self.stationary_loss(m)
                recons.append(recon.item())
                means.append(mean.item()), stds.append(std.item())
                loss = recon + stat
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                p.set_description(f'{loss_name} = {recons[-1]:.4f}, Mean = {means[-1]:.4f}, STD = {stds[-1]:.4f}')
            pbar.set_description(f'{loss_name} = {np.mean(recons):.4f}, Mean = {np.mean(means):.4f}, STD = {np.mean(stds):.4f}')
            self.recon_losses.append(np.mean(recons))
            self.mean_losses.append(np.mean(means))
            self.std_losses.append(np.mean(stds))

    def predict(self, data):
        inputs, anomalies, outputs, rec_errors = [], [], [], []
        mean_errors, std_errors = [], []

        loss = nn.MSELoss(reduction='none').to(self.device)

        for window, anomaly in data:
            # Extract last time step for each batch in window
            inputs.append(window[:, -1, 0].cpu().numpy())
            anomalies.append(anomaly[:, -1, 0].cpu().numpy())

            # Forward pass
            window = window.to(self.device)
            recons, latent = self.forward(window)

            # Extract last time step for each batch after reconstruction
            outputs.append(recons.cpu().detach().numpy()[:, -1, 0])

            # Compute reconstruction error per sample (MSE loss per time step)
            rec_error = loss(window, recons).cpu().detach().numpy()[:, -1, 0]
            rec_errors.append(rec_error)

            # Compute stationary loss
            _, mean, std = self.stationary_loss(latent, per_batch=True)
            mean_errors.append(mean.cpu().detach().numpy()[:, -1, 0])
            std_errors.append(std.cpu().detach().numpy()[:, -1, 0])

        # Concatenate safely, preserving the batch structure
        inputs = np.concatenate(inputs, axis=0)
        anomalies = np.concatenate(anomalies, axis=0)
        outputs = np.concatenate(outputs, axis=0)
        rec_errors = np.concatenate(rec_errors, axis=0)
        mean_errors = np.concatenate(mean_errors, axis=0)
        std_errors = np.concatenate(std_errors, axis=0)

        return inputs, anomalies, outputs, rec_errors, mean_errors, std_errors

    def plot_results(self, data, plot_width: int = 800):
        inputs, anomalies, outputs, errors, mean_errors, std_errors = self.predict(data)
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
                                 name='Reconstruction Errors',
                                 line=dict(color='red')))
        fig.add_trace(go.Scatter(x=list(range(len(mean_errors))),
                                 y=errors,
                                 mode='lines',
                                 name='Mean Errors',
                                 line=dict(color='pink')))
        fig.add_trace(go.Scatter(x=list(range(len(std_errors))),
                                 y=errors,
                                 mode='lines',
                                 name='STD Errors',
                                 line=dict(color='green')))
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
        xs = np.arange(len(self.recon_losses)) + 1
        plt.figure(figsize=fig_size)
        plt.plot(xs, self.recon_losses, label='Reconstruction Loss')
        plt.plot(xs, self.mean_losses, label='Mean Loss')
        plt.plot(xs, self.std_losses, label='STD Loss')
        plt.grid()
        plt.xticks(xs)
        plt.legend()
        plt.show()

    def save(self, path: str = ''):
        """
        Save the model, optimizer state, and training history to a file.
        """
        if path == '':
            path = self.name + '_' + str(len(self.recon_losses)).zfill(3) + '.pth'
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'recon_losses': self.recon_losses,
            'mean_losses': self.mean_losses,
            'std_losses': self.std_losses,
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
        model = TransformerSAE(
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
        model.recon_losses = checkpoint['recon_losses']
        model.mean_losses = checkpoint['mean_losses']
        model.std_losses = checkpoint['std_losses']

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