import torch
from torch import nn
import numpy as np
import torch_geometric
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

## GDN Model (AAAI 21)
class GDN(nn.Module):
    def __init__(self, feats: int = 1):
        super(GDN, self).__init__()
        self.name = 'GDN'
        self.lr = 0.0001
        self.n_feats = feats
        self.n_window = 5
        self.n_hidden = 16
        self.n = self.n_window * self.n_feats

        # Create edge indices for a fully connected graph
        src_ids = np.repeat(np.array(list(range(feats))), feats)
        dst_ids = np.array(list(range(feats)) * feats)
        edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)

        # Create PyTorch Geometric graph
        self.g = Data(edge_index=edge_index)

        # Initialize GATConv layers
        self.feature_gat = GATConv(1, 1, feats)

        # Initialize attention mechanism and fully connected network (FCN)
        self.attention = nn.Sequential(
            nn.Flatten(),
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
        att_score = self.attention(data).view(-1, self.n_window, 1)
        data = data.view(-1, self.n_window, self.n_feats)
        data_r = torch.matmul(data.mT, att_score)

        # GAT convolution on complete graph
        feat_r = self.feature_gat(data_r, self.g.edge_index)
        feat_r = feat_r.view(self.n_feats, self.n_feats)

        # Pass through a FCN
        x = self.fcn(feat_r)
        return x.view(-1)