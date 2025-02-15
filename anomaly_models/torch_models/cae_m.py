import torch
from torch import nn
import numpy as np
from tqdm.notebook import tqdm, trange
import plotly.graph_objects as go
from matplotlib import pyplot as plt

## CAE-M Model (TKDE 21)
class CAE_M(nn.Module):
    def __init__(self, window_size: int , feats: int = 1, device: str = 'cpu', seed: int = 0):
        super(CAE_M, self).__init__()
        torch.manual_seed(seed)
        self.name = 'CAE_M'
        self.lr = 0.001
        self.device = device
        self.n_feats = feats
        self.n_window = window_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), 1, 1), nn.Sigmoid(),
            nn.Conv2d(8, 16, (3, 3), 1, 1), nn.Sigmoid(),
            nn.Conv2d(16, 32, (3, 3), 1, 1), nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 4, (3, 3), 1, 1), nn.Sigmoid(),
            nn.ConvTranspose2d(4, 4, (3, 3), 1, 1), nn.Sigmoid(),
            nn.ConvTranspose2d(4, 1, (3, 3), 1, 1), nn.Sigmoid(),
        )
        self.to(device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.losses = []
        self.threshold = None

    def forward(self, g):
        ## Encode
        z = g.view(-1, 1, self.n_feats, self.n_window)
        z = self.encoder(z)
        ## Decode
        x = self.decoder(z)
        return x.view(-1, self.n_window, self.n_feats)

    def learn(self, train_loader, n_epochs: int, seed: int = 42):
        torch.manual_seed(seed)
        self.train()
        mse = nn.MSELoss(reduction='mean').to(self.device)
        for _ in (pbar := trange(n_epochs)):
            l1s = []
            for d, a in tqdm(train_loader, leave=False):
                d = d.to(self.device)
                x = self.forward(d)
                loss = mse(x, d)
                l1s.append(torch.mean(loss).item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            pbar.set_description(f'MSE = {np.mean(l1s):.4f}')
            self.losses.append(np.mean(l1s))

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
                recon = self.forward(window)
                outputs.append(recon.cpu().detach().numpy().squeeze().T[-1])
                errors.append(mse(window, recon).cpu().detach().numpy().squeeze().T[-1])
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
        model = CAE_M(
            window_size=config['window_size'],
            feats=config['feats'],
            device=config['device']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.losses = checkpoint['losses']

        return model