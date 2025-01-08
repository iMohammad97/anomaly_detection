import torch
from torch import nn
import dgl
from dgl.nn.pytorch import GATConv

## MTAD_GAT Model (ICDM 20)
class MTAD_GAT(nn.Module):
	def __init__(self, feats):
		super(MTAD_GAT, self).__init__()
		self.name = 'MTAD_GAT'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_window = feats
		self.n_hidden = feats * feats
		self.g = dgl.graph((torch.tensor(list(range(1, feats+1))), torch.tensor([0]*feats)))
		self.g = dgl.add_self_loop(self.g)
		self.feature_gat = GATConv(feats, 1, feats)
		self.time_gat = GATConv(feats, 1, feats)
		self.gru = nn.GRU((feats+1)*feats*3, feats*feats, 1)

	def forward(self, data, hidden):
		hidden = torch.rand(1, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
		data = data.view(self.n_window, self.n_feats)
		data_r = torch.cat((torch.zeros(1, self.n_feats), data))
		feat_r = self.feature_gat(self.g, data_r)
		data_t = torch.cat((torch.zeros(1, self.n_feats), data.t()))
		time_r = self.time_gat(self.g, data_t)
		data = torch.cat((torch.zeros(1, self.n_feats), data))
		data = data.view(self.n_window+1, self.n_feats, 1)
		x = torch.cat((data, feat_r, time_r), dim=2).view(1, 1, -1)
		x, h = self.gru(x, hidden)
		return x.view(-1), h