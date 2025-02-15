import torch
from torch import nn
from tqdm.notebook import tqdm, trange
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go

# MAD_GAN (ICANN 19)
class MAD_GAN(nn.Module):
    def __init__(self, feats: int = 1, device: str = 'cpu', seed: int = 0):
        super(MAD_GAN, self).__init__()
        torch.manual_seed(seed)
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
        self.mse_losses = []
        self.gls_losses = []
        self.dls_losses = []
        self.threshold = None

    def forward(self, g):
        ## Generate
        z = self.generator(g)
        ## Discriminator
        real_score = self.discriminator(g)
        fake_score = self.discriminator(z)
        return z, real_score, fake_score

    def learn(self, train_loader, n_epochs: int, seed: int = 42):
        torch.manual_seed(seed)
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
            ml, gl, dl = np.mean(mses), np.mean(gls), np.mean(dls)
            pbar.set_description(f'MSE = {ml:.4f},\tG = {gl:.4f},\tD = {dl:.4f}')
            self.mse_losses.append(ml)
            self.gls_losses.append(gl)
            self.dls_losses.append(dl)

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

    def predict(self, data, train: bool = False):
        self.eval()
        results = {}
        inputs, anomalies, outputs, errors = [], [], [], []
        mse = nn.MSELoss(reduction='none').to(self.device)
        with torch.no_grad():
            for window, anomaly in data:
                if window.shape[0] == 1:
                    break
                inputs.append(window.squeeze().T[-1])
                anomalies.append(anomaly.squeeze().T[-1])
                window = window.to(self.device)
                recon, _, _ = self.forward(window)
                outputs.append(recon.cpu().detach().numpy().squeeze().T[-1])
                errors.append(mse(window.squeeze(), recon).cpu().detach().numpy().squeeze().T[-1])
        results['inputs'] = np.concatenate(inputs)
        results['anomalies'] = np.concatenate(anomalies)
        results['outputs'] = np.concatenate(outputs)
        results['errors'] = np.concatenate(errors)
        if train:
            self.threshold = np.mean(results['errors']) + 3 * np.std(results['errors'])
        elif self.threshold:
            results['predictions'] = [1 if error > self.threshold else 0 for error in results['errors']]
        return results

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

        fig.show()

    def plot_losses(self, fig_size=(10, 6)):
        xs = np.arange(len(self.mse_losses)) + 1
        plt.figure(figsize=fig_size)
        plt.plot(xs, self.mse_losses, label='Reconstruction Loss')
        plt.plot(xs, self.gls_losses, label='Generation Loss')
        plt.plot(xs, self.dls_losses, label='Discrimination Loss')
        plt.grid()
        plt.xticks(xs)
        plt.legend()
        plt.show()

    def save(self, path: str = ''):
        """
        Save the model, optimizer state, and training history to a file.
        """
        if path == '':
            path = self.name + '_' + str(len(self.mse_losses)).zfill(3) + '.pth'
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'mse_losses': self.mse_losses,
            'gls_losses': self.gls_losses,
            'dls_losses': self.dls_losses,
            'config': {
                'feats': self.n_feats,
                'device': self.device,
            }
        }, path)
        print(f'Model saved to path = {path}')

    @staticmethod
    def load(path: str):
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint['config']
        model = MAD_GAN(
            feats=config['feats'],
            device=config['device']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.mse_losses = checkpoint['mse_losses']
        model.gls_losses = checkpoint['gls_losses']
        model.dls_losses = checkpoint['dls_losses']

        return model