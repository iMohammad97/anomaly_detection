import torch
import torch.nn as nn
from tqdm.notebook import tqdm, trange
import numpy as np
import plotly.graph_objects as go

class VAE(nn.Module):
    def __init__(self, n_features: int = 1, window_size: int = 256, latent_dim: int = 32, lstm_units: int = 64, device: str = 'cpu'):
        super(VAE, self).__init__()
        self.name = 'VAE'
        self.lr = 0.0001
        self.device = device
        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units

        self.encoder_lstm1 = nn.LSTM(n_features, lstm_units, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True)
        self.encoder_lstm3 = nn.LSTM(lstm_units, lstm_units, batch_first=True)

        self.fc_mu = nn.Linear(lstm_units * window_size, latent_dim)
        self.fc_logvar = nn.Linear(lstm_units * window_size, latent_dim)

        self.decoder_lstm1 = nn.LSTM(latent_dim, lstm_units, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True)
        self.decoder_lstm3 = nn.LSTM(lstm_units, n_features, batch_first=True)

        self.to(device)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.losses = []

    def encode(self, x):
        x, _ = self.encoder_lstm1(x)
        x, _ = self.encoder_lstm2(x)
        x, _ = self.encoder_lstm3(x)
        x = x.reshape(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = z.unsqueeze(1).repeat(1, self.window_size, 1)
        x, _ = self.decoder_lstm1(z)
        x, _ = self.decoder_lstm2(x)
        output, _ = self.decoder_lstm3(x)
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = nn.MSELoss(reduction='mean')(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE, KLD

    def learn(self, train_loader, n_epochs: int):
        self.train()
        for _ in (pbar := trange(n_epochs)):
            mses, klds = [], []
            for d, _ in tqdm(train_loader, leave=False):
                d = d.to(self.device)
                recon, mu, logvar = self.forward(d)
                bce, kld = self.loss_function(recon, d, mu, logvar)
                loss = bce + kld
                mses.append(bce.item()), klds.append(kld.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            pbar.set_description(f'MSE = {np.mean(mses):.4f}, KLD = {np.mean(klds):.4f}')
            self.losses.append(np.mean(mses) + np.mean(klds))

    def predict(self, data, name: str = ''):
        inputs, anomalies, outputs, errors = [], [], [], []
        mse = nn.MSELoss(reduction='none').to(self.device)
        for window, anomaly in data:
            inputs.append(window.squeeze().T[-1])
            anomalies.append(anomaly.squeeze().T[-1])
            window = window.to(self.device)
            recon, _, _ = self.forward(window)
            outputs.append(recon.cpu().detach().numpy().squeeze().T[-1])
            errors.append(mse(window, recon).cpu().detach().numpy().squeeze().T[-1])
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