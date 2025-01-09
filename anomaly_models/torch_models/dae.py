import torch
import torch.nn as nn
from tqdm.notebook import tqdm, trange
import numpy as np

class DAE(nn.Module):
    def __init__(self, n_features: int = 1, window_size: int = 256, latent_dim: int = 32, lstm_units: int = 64,
                 mean_coef: float = 1, std_coef: float = 1, device: str = 'cpu'):
        super(DAE, self).__init__()
        self.name = 'DAE'
        self.lr = 0.0001
        self.device = device
        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.stationary_loss = StationaryLoss(mean_coef, std_coef)

        self.encoder_lstm1 = nn.LSTM(n_features, lstm_units, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True)
        self.encoder_lstm3 = nn.LSTM(lstm_units, latent_dim, batch_first=True)

        self.decoder_lstm1 = nn.LSTM(latent_dim, lstm_units, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True)
        self.decoder_lstm3 = nn.LSTM(lstm_units, n_features, batch_first=True)

        self.to(device)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)

    def forward(self, x):
        # Encode
        x, _ = self.encoder_lstm1(x)
        x, _ = self.encoder_lstm2(x)
        x, _ = self.encoder_lstm3(x)
        latent = x[:, -1, :]

        # Apply custom loss to the latent space
        latent_with_loss, _ = self.stationary_loss(latent)

        # Decode
        latent_repeated = latent_with_loss.unsqueeze(1).repeat(1, self.window_size, 1)
        x, _ = self.decoder_lstm1(latent_repeated)
        x, _ = self.decoder_lstm2(x)
        output, _ = self.decoder_lstm3(x)

        return output

    def learn(self, train_loader, n_epochs: int):
        self.train()
        mse = nn.MSELoss(reduction='mean').to(self.device)
        losses = []
        for _ in (pbar := trange(n_epochs)):
            recons, means, stds = [], [], []
            for d, a in tqdm(train_loader, leave=False):
                d = d.to(self.device)
                x = self.forward(d)
                recon = mse(x, d)
                mean = self.stationary_loss.mse_loss
                std = self.stationary_loss.std_loss
                loss = recon + mean + std
                recons.append(recon.item()), means.append(mean.item()), stds.append(std.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            pbar.set_description(
                f'Rec Loss = {np.mean(recons):.4f}, Avg Loss = {np.mean(means):.4f}, STD Loss = {np.mean(stds):.4f} ')
            losses.append(loss.item())
        return losses

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