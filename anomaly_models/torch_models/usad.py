import torch
from torch import nn
import numpy as np
import plotly.graph_objects as go
from tqdm.notebook import tqdm, trange

## USAD Model (KDD 20)
class USAD(nn.Module):
	def __init__(self, feats: int = 1, device: str = 'cpu'):
		super(USAD, self).__init__()
		self.name = 'USAD'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_hidden = 16
		self.n_latent = 5
		self.n_window = 5 # USAD w_size = 5
		self.n = self.n_feats * self.n_window
		self.encoder = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(True),
		)
		self.decoder1 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.decoder2 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(True),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.device = device
		self.optimizer = torch.optim.AdamW(self.parameters() , lr=self.lr, weight_decay=1e-5)
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)
		self.losses1, self.losses2 = [], []

	def forward(self, g):
		## Encode
		z = self.encoder(g)
		## Decoders (Phase 1)
		ae1 = self.decoder1(z)
		ae2 = self.decoder2(z)
		## Encode-Decode (Phase 2)
		ae2ae1 = self.decoder2(self.encoder(ae1))
		return ae1.view(-1, self.n_window, self.n_feats), ae2.view(-1, self.n_window, self.n_feats), ae2ae1.view(-1, self.n_window, self.n_feats)

	def learn(self, train_loader, n_epochs: int):
		self.train()
		mse = nn.MSELoss(reduction='none').to(self.device)
		for n in (pbar := trange(1, n_epochs, + 1)):
			l1s, l2s = [], []
			for d, a in tqdm(train_loader, leave=False):
				d = d.to(self.device)
				ae1s, ae2s, ae2ae1s = self.forward(d)
				l1 = (1 / n) * mse(ae1s, d) + (1 - 1 / n) * mse(ae2ae1s, d)
				l2 = (1 / n) * mse(ae2s, d) - (1 - 1 / n) * mse(ae2ae1s, d)
				l1s.append(torch.mean(l1).item())
				l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1 + l2)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
			self.scheduler.step()
			l1, l2 = np.mean(l1s), np.mean(l2s)
			pbar.set_description(f'L1 = {l1:.3f},\tL2 = {l2:.3f}')
			self.losses1.append(l1), self.losses2.append(l2)

	def predict(self, data):
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