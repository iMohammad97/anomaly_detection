import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Sampler
from tqdm.notebook import trange, tqdm
import plotly.graph_objects as go
from matplotlib import pyplot as plt


def softmax(x, temp=1.0):
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum()


class PrioritizedSampler(Sampler):
    def __init__(self, data_source, energies, temp=0.1):
        super().__init__(data_source)
        self.data_source = data_source
        self.energies = np.array(energies, dtype=np.float32)  # Ensure array format
        self.temp = temp
        self.weights = softmax(self.energies, temp)

    def __iter__(self):
        """Return sampled indices based on priority."""
        self.sampled_indices = np.random.choice(len(self.data_source), size=len(self.data_source), p=self.weights, replace=True)
        return iter(self.sampled_indices)

    def __len__(self):
        return len(self.data_source)

    def update_weights(self, new_energies):
        # Update the energies
        for idx, new_energy in zip(self.sampled_indices, new_energies):
            self.energies[idx] = new_energy

        # Update the sampling weights
        self.weights = softmax(self.energies, self.temp)


class PrioritizedEBM(nn.Module):
    def __init__(self, window_size=64, hidden_dim=64, device='cpu', seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.device = device
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        self.energy_net = nn.Sequential(
            nn.Linear(window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.to(device)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        self.losses = []

    def forward(self, x):
        x = x.squeeze()
        return self.energy_net(x).squeeze()

    def learn(self, dataset, n_epochs=10, temp=0.1):
        self.train()
        energies = np.ones(len(dataset))  # Initialize uniform energies
        sampler = PrioritizedSampler(dataset, energies, temp)
        train_loader = DataLoader(dataset, batch_size=32, sampler=sampler)

        for _ in (pbar := trange(n_epochs)):
            epoch_loss = []
            for data, _ in (p := tqdm(train_loader, leave=False)):
                data = data.to(self.device)
                energy = self.forward(data)
                loss = torch.mean(energy)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                sampler.update_weights(energy.detach().cpu().numpy())
                epoch_loss.append(loss.item())
                p.set_description(f'Batch Loss = {int(epoch_loss[-1])}')

            self.losses.append(np.mean(epoch_loss))
            pbar.set_description(f'Epoch Loss = {int(self.losses[-1])}')

    def predict(self, dataset):
        self.eval()
        test_loader = DataLoader(dataset, batch_size=256)
        inputs, anomalies, outputs = [], [], []
        with torch.no_grad():
            for window, anomaly in test_loader:
                inputs.append(window.squeeze().T[-1])
                anomalies.append(anomaly.squeeze().T[-1])
                window = window.to(self.device)
                energy = self.forward(window)
                outputs.append(energy.cpu().detach().numpy().squeeze().T)
        inputs = np.concatenate(inputs)
        anomalies = np.concatenate(anomalies)
        outputs = np.concatenate(outputs)
        outputs -= np.min(outputs)
        outputs /= np.max(outputs)
        return inputs, anomalies, outputs

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

    def plot_losses(self, fig_size=(10, 6)):
        xs = np.arange(len(self.losses)) + 1
        plt.figure(figsize=fig_size)
        plt.plot(xs, -np.log(-np.array(self.losses)), label='Energy Loss')
        plt.grid()
        plt.xticks(xs)
        plt.legend()
        plt.show()