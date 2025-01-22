import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm, trange
import plotly.graph_objects as go

class StudentAE(nn.Module):
    def __init__(self, teacher, window_size: int = 256, latent_dim: int = 16, device: str = 'cpu', seed: int = 0):
        super(StudentAE, self).__init__()
        torch.manual_seed(seed)
        self.name = 'StudentAE'
        self.teacher = teacher
        self.teacher_dim = teacher.latent_dim
        self.window_size = window_size
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(teacher.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, teacher.latent_dim)
        )

        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.losses = []

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def learn(self, train_loader, n_epochs: int):
        mse = nn.MSELoss().to(self.device)
        self.train()
        for _ in (pbar := trange(n_epochs)):
            epoch_loss = 0
            for d, _ in tqdm(train_loader, leave=False):
                d = d.to(self.device)
                teacher_latent = self.teacher.encode(d)
                _, output = self.forward(teacher_latent)
                loss = mse(output, teacher_latent)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            self.losses.append(epoch_loss / len(train_loader))
            pbar.set_description(f'Loss: {epoch_loss / len(train_loader):.4f}')

    def predict(self, data):
        inputs, anomalies, outputs, errors = [], [], [], []
        for window, anomaly in data:
            inputs.append(window.squeeze().T[-1])
            anomalies.append(anomaly.squeeze().T[-1])
            window = window.to(self.device)
            teacher_latent = self.teacher.encode(window)
            _, student_output = self.forward(teacher_latent)
            recons = self.teacher.decode(student_output)
            outputs.append(recons.cpu().detach().numpy().squeeze().T[-1])
            anomaly_score = torch.mean((teacher_latent - student_output) ** 2, dim=1)
            errors.append(anomaly_score.cpu().detach().numpy().squeeze().T[-1])
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

    def plot_losses(self, fig_size=(10, 6)):
        xs = np.arange(len(self.losses)) + 1
        plt.figure(figsize=fig_size)
        plt.plot(xs, self.losses, label='Total Loss')
        plt.grid()
        plt.xticks(xs)
        plt.legend()
        plt.show()
