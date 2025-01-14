import torch
from torch import nn
from tqdm.notebook import tqdm, trange
import numpy as np
import plotly.graph_objects as go

# MAD_GAN (ICANN 19)
class MAD_GAN(nn.Module):
    def __init__(self, feats: int = 1, device: str = 'cpu'):
        super(MAD_GAN, self).__init__()
        self.name = 'MAD_GAN'
        self.lr = 0.0001
        self.device = device
        self.n_feats = feats
        self.n_hidden = 16
        self.n_window = 5 # MAD_GAN w_size = 5
        self.n = self.n_feats * self.n_window
        self.generator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
        )
        self.discriminator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
            nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
        )
        self.to(device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.losses = []

    def forward(self, g):
        ## Generate
        z = self.generator(g)
        ## Discriminator
        real_score = self.discriminator(g)
        fake_score = self.discriminator(z)
        return z, real_score, fake_score

    def learn(self, train_loader, n_epochs: int):
        self.train()
        bcel = nn.BCELoss(reduction='mean').to(self.device)
        msel = nn.MSELoss(reduction='mean').to(self.device)
        for _ in (pbar := trange(n_epochs)):
            mses, gls, dls = [], [], []
            for d, _ in tqdm(train_loader, leave=False):
                d = d.to(self.device)
                # training discriminator
                self.discriminator.zero_grad()
                _, real, fake = self.forward(d)
                real_label, fake_label = 0.9 * torch.ones_like(real), 0.1 * torch.ones_like(fake)
                dl = bcel(real, real_label) + bcel(fake, fake_label)
                dl.backward()
                self.generator.zero_grad()
                self.optimizer.step()
                # training generator
                z, _, fake = self.forward(d)
                mse = msel(z, d.squeeze())
                gl = bcel(fake, real_label)
                tl = gl + mse
                tl.backward()
                self.discriminator.zero_grad()
                self.optimizer.step()
                mses.append(mse.item()), gls.append(gl.item()), dls.append(dl.item())
            pbar.set_description(f'MSE = {np.mean(mses):.4f},\tG = {np.mean(gls):.4f},\tD = {np.mean(dls):.4f}')
            self.losses.append(np.mean(gls) + np.mean(dls))

    # def predict(self, data):
    #     self.eval()
    #     l = nn.MSELoss(reduction='none')
    #     outputs = []
    #     for d, a in data:
    #         d = d.to(self.device)
    #         z, _, _ = self.forward(d)
    #         outputs.append(z)
    #     outputs = torch.stack(outputs)
    #     y_pred = outputs[:, data.shape[1] - self.feats:data.shape[1]].view(-1, self.feats)
    #     loss = l(outputs, data)
    #     loss = loss[:, data.shape[1] - self.feats:data.shape[1]].view(-1, self.feats)
    #     return loss.detach().numpy(), y_pred.detach().numpy()

    def predict(self, data, name: str = ''):
        inputs, anomalies, outputs, errors = [], [], [], []
        mse = nn.MSELoss(reduction='none').to(self.device)
        for window, anomaly in data:
            inputs.append(window.squeeze().T[-1])
            anomalies.append(anomaly.squeeze().T[-1])
            window = window.to(self.device)
            recon, _, _ = self.forward(window)
            outputs.append(recon.cpu().detach().numpy().squeeze().T[-1])
            errors.append(mse(window.squeeze(), recon).cpu().detach().numpy().squeeze().T[-1])
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
