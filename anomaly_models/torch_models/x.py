import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm, trange
import numpy as np
import plotly.graph_objects as go
from xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig, sLSTMBlockConfig
from xlstm.blocks.slstm.layer import sLSTMLayerConfig

class xLSTM_AE(nn.Module):
    def __init__(self, n_features: int = 1, window_size: int = 256, latent_dim: int = 32,
                 num_blocks: int = 4, embedding_dim: int = 128, device: str = 'cpu', seed: int = 0):
        super(xLSTM_AE, self).__init__()
        torch.manual_seed(seed)
        self.name = 'xLSTM_AE'
        self.lr = 0.0001
        self.device = device
        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.embedding_dim = embedding_dim

        # **Fix: Project input features to match embedding_dim**
        self.input_projection = nn.Linear(n_features, embedding_dim)

        # Encoder using xLSTMBlockStack
        self.encoder = xLSTMBlockStack(
            xLSTMBlockStackConfig(
                slstm_block=sLSTMBlockConfig(slstm=sLSTMLayerConfig(backend="vanilla")),
                slstm_at="all",
                num_blocks=num_blocks,
                embedding_dim=embedding_dim
            )
        )

        # Bottleneck (latent space)
        self.bottleneck = nn.Linear(embedding_dim, latent_dim)
        self.unbottleneck = nn.Linear(latent_dim, embedding_dim)

        # Decoder using xLSTMBlockStack
        self.decoder = xLSTMBlockStack(
            xLSTMBlockStackConfig(
                slstm_block=sLSTMBlockConfig(slstm=sLSTMLayerConfig(backend="vanilla")),
                slstm_at="all",
                num_blocks=num_blocks,
                embedding_dim=embedding_dim
            )
        )

        # **Fix: Project output back to original feature dimension**
        self.output_projection = nn.Linear(embedding_dim, n_features)

        self.to(device)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        self.losses = []
        self.threshold = None

    def forward(self, x):
        # **Fix: Project input from `n_features` to `embedding_dim`**
        x = self.input_projection(x)

        # Encode
        encoded = self.encoder(x)
        latent = self.bottleneck(encoded)

        # Decode
        unbottlenecked = self.unbottleneck(latent)
        reconstructed = self.decoder(unbottlenecked)

        # **Fix: Project output back to `n_features`**
        reconstructed = self.output_projection(reconstructed)

        return reconstructed

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
            for d, a in (p := tqdm(train_loader, leave=False)):
                d = d.to(self.device)
                x = self.forward(d)
                loss = loss_fn(x, d)
                recons.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                p.set_description(f'Batch {loss_name} Loss = {loss.item():.4f}')
            pbar.set_description(f'{loss_name} Loss = {np.mean(recons):.4f}')
            self.losses.append(np.mean(recons))

    def predict(self, data, train: bool = False):
        self.eval()
        results = {}
        inputs, anomalies, outputs, errors = [], [], [], []
        loss = nn.MSELoss(reduction='none').to(self.device)
        with torch.no_grad():
            for window, anomaly in data:
                if window.shape[0] == 1:
                    break
                inputs.append(window.squeeze().T[-1])
                anomalies.append(anomaly.squeeze().T[-1])
                window = window.to(self.device)
                recons = self.forward(window)
                outputs.append(recons.cpu().detach().numpy().squeeze().T[-1])
                errors.append(loss(window, recons).cpu().detach().numpy().squeeze().T[-1])
        results['inputs'] = np.concatenate(inputs)
        results['anomalies'] = np.concatenate(anomalies)
        results['outputs'] = np.concatenate(outputs)
        results['errors'] = np.concatenate(errors)
        if train:
            self.threshold = np.mean(results['errors']) + 3 * np.std(results['errors'])
        elif not train and self.threshold is not None:
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
                'num_blocks': self.num_blocks,
                'embedding_dim': self.embedding_dim,
                'device': self.device,
                'lr': self.lr,
            }
        }, path)
        print(f'Model saved to path = {path}')

    @staticmethod
    def load(path: str):
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint['config']
        model = xLSTM_AE(
            n_features=config['n_features'],
            window_size=config['window_size'],
            latent_dim=config['latent_dim'],
            num_blocks=config['num_blocks'],
            embedding_dim=config['embedding_dim'],
            device=config['device']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.losses = checkpoint['losses']
        return model