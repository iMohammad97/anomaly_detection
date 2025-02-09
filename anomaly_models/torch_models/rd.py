import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import trange, tqdm
import plotly.graph_objects as go
import matplotlib.pyplot as plt


class StudentDecoder(nn.Module):
    def __init__(self, teacher_latent_dim: int, hidden_dim: int = 8, lr: float = 0.0001,
                 device: str = 'cpu'):
        super(StudentDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.teacher_latent_dim = teacher_latent_dim
        self.device = device

        # Fully connected layers for reconstruction
        self.decoder_fc1 = nn.Linear(teacher_latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_fc3 = nn.Linear(hidden_dim, teacher_latent_dim)

        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.losses = []

    def forward(self, embedding):
        x = torch.relu(self.decoder_fc1(embedding))
        x = torch.relu(self.decoder_fc2(x))
        x = self.decoder_fc3(x)
        return x

    def learn(self, teacher_model, train_loader, n_epochs: int = 10):
        teacher_model.eval()  # Ensure the teacher is in evaluation mode
        self.train()
        criterion = nn.MSELoss()
        self.to(self.device)
        teacher_model.to(self.device)

        for _ in (pbar := trange(n_epochs, desc="Training")):
            epoch_loss = 0
            for data, _ in tqdm(train_loader, leave=False, desc="Batch Progress"):
                data = data.to(self.device)

                # Pass data through the teacher to get embeddings
                teacher_embedding = teacher_model.encode(data)

                # Pass teacher embeddings to the student model
                reconstructed = self.forward(teacher_embedding)

                # Compute reconstruction loss
                loss = criterion(reconstructed, teacher_embedding)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            self.losses.append(epoch_loss)
            pbar.set_description(f"Epoch Loss: {epoch_loss:.4f}")

    def predict(self, teacher_model, data_loader):
        teacher_model.eval()
        self.eval()

        inputs, anomalies, outputs, errors = [], [], [], []
        criterion = nn.MSELoss(reduction='none').to(self.device)

        for window, anomaly in data_loader:
            if window.shape[0] == 1:
                break
            inputs.append(window.squeeze().T[-1])
            anomalies.append(anomaly.squeeze().T[-1])
            window = window.to(self.device)

            with torch.no_grad():
                teacher_embedding = teacher_model.encode(window)
                reconstructed = self.forward(teacher_embedding)
                error = criterion(teacher_embedding, reconstructed)
                reconstructed = teacher_model.decode(reconstructed)

            outputs.append(reconstructed.cpu().detach().numpy().squeeze().T[-1])
            errors.append(error.cpu().detach().numpy().squeeze().T[-1])

        inputs = np.concatenate(inputs)
        anomalies = np.concatenate(anomalies)
        outputs = np.concatenate(outputs)
        errors = np.concatenate(errors)
        return inputs, anomalies, outputs, errors

    def plot_results(self, teacher, data, plot_width: int = 800):
        inputs, anomalies, outputs, errors = self.predict(teacher, data)
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