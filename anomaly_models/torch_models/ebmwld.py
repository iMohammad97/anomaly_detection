import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go
from tqdm.notebook import tqdm, trange


class EBMwLD(nn.Module):
    def __init__(self, n_features: int = 1, window_size: int = 64, device="cpu"):
        super(EBMwLD, self).__init__()
        self.device = device
        self.n_features = n_features

        # Simplified energy network with only Linear layers
        self.energy_net = nn.Sequential(
            nn.Linear(n_features * window_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.to(self.device)

    def forward(self, x):
        """Compute the energy of the input."""
        return self.energy_net(x.squeeze()).squeeze()

    def langevin_dynamics(self, x, n_steps=20, step_size=0.01, noise_scale=0.005):
        """Perform Langevin dynamics to generate negative samples."""
        x = x.clone().detach().to(self.device).requires_grad_(True)

        for _ in range(n_steps):
            # Compute energy and gradients
            energy = self.forward(x)
            energy.sum().backward()

            # Update with Langevin dynamics
            x.data -= step_size * x.grad.data
            x.data += noise_scale * torch.randn_like(x)
            x.grad.detach_()
            x.grad.zero_()

        return x.detach()

    def learn(self, train_loader, n_epochs=10, seed=None):
        """Train the model using contrastive divergence."""
        if seed is not None:
            torch.manual_seed(seed)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        for _ in (pbar := trange(n_epochs)):
            energy_values = []
            for data, _ in tqdm(train_loader, leave=False):
                data = data.to(self.device)

                # Compute energy for positive samples
                positive_energy = self.forward(data)

                # Generate negative samples using Langevin dynamics
                negative_samples = torch.randn_like(data).to(self.device)
                negative_samples = self.langevin_dynamics(negative_samples)
                negative_energy = self.forward(negative_samples)

                # Compute contrastive divergence loss
                loss = torch.mean(positive_energy) - torch.mean(negative_energy)
                energy_values.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = sum(energy_values) / len(energy_values)
            pbar.set_description(f"Loss = {avg_loss:.4f}")

    def predict(self, test_loader):
        self.eval()
        inputs, anomalies, outputs, errors = [], [], [], []
        with torch.no_grad():
            for window, anomaly in test_loader:
                if window.shape[0] == 1:
                    break
                inputs.append(window.squeeze().T[-1])
                anomalies.append(anomaly.squeeze().T[-1])
                window = window.to(self.device)
                energy = self.forward(window)
                outputs.append(energy.cpu().detach().numpy().squeeze())
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