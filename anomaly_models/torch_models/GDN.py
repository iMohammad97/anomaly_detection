import torch
from torch import nn
import numpy as np
import dgl
from dgl.nn.pytorch import GATConv

## GDN Model (AAAI 21)
class GDN(nn.Module):
	def __init__(self, feats):
		super(GDN, self).__init__()
		self.name = 'GDN'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_window = 5
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats
		src_ids = np.repeat(np.array(list(range(feats))), feats)
		dst_ids = np.array(list(range(feats))*feats)
		self.g = dgl.graph((torch.tensor(src_ids), torch.tensor(dst_ids)))
		self.g = dgl.add_self_loop(self.g)
		self.feature_gat = GATConv(1, 1, feats)
		self.attention = nn.Sequential(
			nn.Linear(self.n, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_window), nn.Softmax(dim=0),
		)
		self.fcn = nn.Sequential(
			nn.Linear(self.n_feats, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_window), nn.Sigmoid(),
		)

	def forward(self, data):
		# Bahdanau style attention
		att_score = self.attention(data).view(self.n_window, 1)
		data = data.view(self.n_window, self.n_feats)
		data_r = torch.matmul(data.permute(1, 0), att_score)
		# GAT convolution on complete graph
		feat_r = self.feature_gat(self.g, data_r)
		feat_r = feat_r.view(self.n_feats, self.n_feats)
		# Pass through a FCN
		x = self.fcn(feat_r)
		return x.view(-1)
