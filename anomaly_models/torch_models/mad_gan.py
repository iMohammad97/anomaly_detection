import torch
from torch import nn
from tqdm.notebook import tqdm, trange
import numpy as np


# MAD_GAN (ICANN 19)
class MAD_GAN(nn.Module):
	def __init__(self, feats: int = 1):
		super(MAD_GAN, self).__init__()
		self.name = 'MAD_GAN'
		self.lr = 0.0001
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
		self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)

	def forward(self, g):
		## Generate
		z = self.generator(g)
		## Discriminator
		real_score = self.discriminator(g)
		fake_score = self.discriminator(z)
		return z, real_score, fake_score

	def train_model(self, train_loader, n_epochs: int):
		bcel = nn.BCELoss(reduction='mean')
		msel = nn.MSELoss(reduction='mean')
		losses = []
		for _ in (pbar := trange(n_epochs)):
			mses, gls, dls = [], [], []
			for d, a in tqdm(train_loader, leave=False):
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
			losses.append(np.mean(gls) + np.mean(dls))
		return losses

	def predict(self, data):
		l = nn.MSELoss(reduction='none')
		outputs = []
		for d, a in data:
			z, _, _ = self.forward(d)
			outputs.append(z)
		outputs = torch.stack(outputs)
		y_pred = outputs[:, data.shape[1] - self.feats:data.shape[1]].view(-1, self.feats)
		loss = l(outputs, data)
		loss = loss[:, data.shape[1] - self.feats:data.shape[1]].view(-1, self.feats)
		return loss.detach().numpy(), y_pred.detach().numpy()
