from tqdm import tqdm
import torch.nn as nn
import torch
import random
import numpy as np
import pandas as pd
import os
from torch.utils.data import  DataLoader
from time import time

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'

class OmniAnomaly(nn.Module):
	def __init__(self, feats):
		super(OmniAnomaly, self).__init__()
		self.name = 'OmniAnomaly'
		self.lr = 0.002
		self.beta = 0.01
		self.n_feats = feats
		self.n_hidden = 32
		self.n_latent = 8
		self.lstm = nn.GRU(feats, self.n_hidden, 2)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Flatten(),
			nn.Linear(self.n_hidden, 2*self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
		)

	def forward(self, x, hidden = None):
		hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float32) if hidden is not None else hidden
		out, hidden = self.lstm(x.view(1, 1, -1), hidden)
		## Encode
		x = self.encoder(out)
		mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
		## Reparameterization trick
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		x = mu + eps*std
		## Decoder
		x = self.decoder(x)
		return x.view(-1), mu.view(-1), logvar.view(-1), hidden

	def train_model(self, model_args, trainD, trainO,num_epochs=5):
		optimizer, scheduler, epoch, accuracy_list = model_args
		e = epoch + 1
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = self.backprop(e, trainD, trainO, optimizer, scheduler)
			accuracy_list.append((lossT, lr))
		return (optimizer, scheduler)
	
	def test_model(self,train_args,testD, testO):
		optimizer, scheduler = train_args
		torch.zero_grad = True
		self.eval()
		loss, _ = self.backprop(0, testD, testO, optimizer, scheduler, training=False)
		return loss
	
	def backprop(self,epoch, data, dataO, optimizer, scheduler, training = True):
		l = nn.MSELoss(reduction = 'mean' if training else 'none')
		feats = dataO.shape[1]
		if training:
			mses, klds = [], []
			for i, d in enumerate(data):
				y_pred, mu, logvar, hidden = self(d, hidden if i else None)
				MSE = l(y_pred, d)
				KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
				loss = MSE + self.beta * KLD
				mses.append(torch.mean(MSE).item()); klds.append(self.beta * torch.mean(KLD).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			y_preds = []
			for i, d in enumerate(data):
				y_pred, _, _, hidden = self(d, hidden if i else None)
				y_preds.append(y_pred)
			y_pred = torch.stack(y_preds)
			MSE = l(y_pred, data)
			return MSE.detach().numpy(), y_pred.detach().numpy()

# Function to load model
def load_model(model, dataset, use_pretrain=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{model.name}_{dataset}/model.ckpt'
    if os.path.exists(fname) and use_pretrain:
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1
        accuracy_list = []
    return model, (optimizer, scheduler, epoch, accuracy_list)

# Function for rolling window training and testing
def rolling_window_train_test(model, train_data, window_size, dataset_name):
	torch.manual_seed(40)
	random.seed(40)
	np.random.seed(40)
	index = train_data.index
	train_data = train_data.to_numpy().reshape(-1, 1).astype(np.float32)

	train_loader = DataLoader(train_data, batch_size=train_data.shape[0])
	train_data = next(iter(train_loader))
	print('-----------------Train Data--------------------')
	print(train_data)

	model, model_args = load_model(model, dataset_name)

	total_data = train_data.shape[0]
	print('-----------------Total Data--------------------')
	print(total_data)
	test_losses = []

	for start in range(0, total_data, window_size):
		end = start + window_size
		if end > total_data:
			break

		trainD = train_data[:end]
		testD = train_data[end:end + window_size]

		if len(testD) == 0:
			break

		trainO, testO = trainD, testD

		start_time = time()
		train_args = model.train_model(model_args, trainD, trainO)
		print(f'Training time for data points {start} to {end}: {time() - start_time:.4f} s')

		loss = model.test_model(train_args, testD, testO)
		print(f'Test loss for data points {end} to {end + window_size}:')
		
		test_losses.extend(loss)

	flat_test_losses = [item[0] for item in test_losses]
	flat_test_losses = [0.0] * 300 + flat_test_losses

	result = pd.DataFrame({'test_losses': flat_test_losses}, index=index)
	return result


