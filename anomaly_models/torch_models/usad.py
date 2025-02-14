import torch
from torch import nn
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm, trange

## USAD Model (KDD 20)
class USAD(nn.Module):
	def __init__(self, feats: int = 1, device: str = 'cpu', seed: int = 0):
		super(USAD, self).__init__()
		torch.manual_seed(seed)
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
		self.to(device)
		self.optimizer = torch.optim.AdamW(self.parameters() , lr=self.lr, weight_decay=1e-5)
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)
		self.losses1, self.losses2 = [], []
		self.threshold = None

	def forward(self, g):
		## Encode
		z = self.encoder(g)
		## Decoders (Phase 1)
		ae1 = self.decoder1(z)
		ae2 = self.decoder2(z)
		## Encode-Decode (Phase 2)
		ae2ae1 = self.decoder2(self.encoder(ae1))
		return ae1.view(-1, self.n_window, self.n_feats), ae2.view(-1, self.n_window, self.n_feats), ae2ae1.view(-1, self.n_window, self.n_feats)

	def learn(self, train_loader, n_epochs: int, seed: int = 42):
		torch.manual_seed(seed)
		self.train()
		mse = nn.MSELoss(reduction='none').to(self.device)
		for n in (pbar := trange(1, n_epochs + 1)):
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
				recon, _, ae2ae1s = self.forward(window)
				outputs.append(recon.cpu().detach().numpy().squeeze().T[-1])
				e = mse(window, recon).cpu().detach().numpy().squeeze().T[-1]
				e += mse(window, ae2ae1s).cpu().detach().numpy().squeeze().T[-1]
				errors.append(e)
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
		xs = np.arange(len(self.losses1)) + 1
		plt.figure(figsize=fig_size)
		plt.plot(xs, self.losses1, label='Loss 1')
		plt.plot(xs, self.losses2, label='Loss 2')
		plt.grid()
		plt.xticks(xs)
		plt.legend()
		plt.show()

	def save(self, path: str = ''):
		"""
        Save the model, optimizer state, and training history to a file.
        """
		if path == '':
			path = self.name + '_' + str(len(self.losses1)).zfill(3) + '.pth'
		torch.save({
			'model_state_dict': self.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'losses1': self.losses1,
			'losses2': self.losses2,
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
		model = USAD(
			feats=config['feats'],
			device=config['device']
		)
		model.load_state_dict(checkpoint['model_state_dict'])
		model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		model.losses1 = checkpoint['losses1']
		model.losses2 = checkpoint['losses2']

		return model