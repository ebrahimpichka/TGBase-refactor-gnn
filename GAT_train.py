

import argparse
import sys
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

import torch
from torch_geometric.data import Data, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from tgbase_static import StaticTGBase
from tgbase_dyn import DynamicTGBase

from sklearn.preprocessing import MinMaxScaler


rnd_seed = 2021
random.seed(rnd_seed)

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dropout):
        super(DyGATLayer, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)

    def forward(self, x, edge_index):
        return self.gat(x, edge_index)

class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, num_layers, dropout):
        super(DyGATNet, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(DyGATLayer(in_channels, out_channels, num_heads, dropout))
            in_channels = out_channels * num_heads

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

# class DynamicGraphAttentionNetwork(nn.Module):
#     def __init__(self, in_channels, out_channels, num_heads, num_layers, dropout, num_classes, num_timesteps):
#         super(DynamicGraphAttentionNetwork, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.num_classes = num_classes
#         self.num_timesteps = num_timesteps

#         self.dygat = GATNet(in_channels, out_channels, num_heads, num_layers, dropout)
#         self.lstm = nn.LSTM(out_channels * num_heads, out_channels * num_heads, batch_first=True)
#         self.fc = nn.Linear(out_channels * num_heads, num_classes)
#         self.num_timesteps = num_timesteps

#     def forward(self, x_sequence, edge_index_sequence):
#         batch_size = x_sequence.size(0)
#         h = torch.zeros(1, batch_size, self.out_channels * self.num_heads).to(x.device)
#         c = torch.zeros(1, batch_size, self.out_channels * self.num_heads).to(x.device)
#         outputs = []

#         for t in range(self.num_timesteps):
#             x = x_sequence[t, :, :]
#             edge_index = edge_index_sequence[t, :, :]
#             x = self.dygat(x, edge_index)
#             x, (h, c) = self.lstm(x, (h, c))
#             output = self.fc(x.view(batch_size, -1))
#             outputs.append(output)

#         return torch.stack(outputs, dim=1)
        
def train(args):

    network = args.network
    data_path_partial = f'./data/{network}/'
    meta_col_names = ['node_id', 'timestamp', 'label', 'train_mask', 'val_mask', 'test_mask']

    tgbase_encoder = DynamicTGBase(network, args.val_ratio, args.test_ratio, args.use_validation)
    train_emb_df, val_emb_df , test_emb_df = tgbase_encoder.encode_features()
    
    training = [train_emb_df, val_emb_df]
    train_emb_df = pd.concat(training)
    y_train = train_emb_df['label'].tolist()
    X_train = train_emb_df.drop(meta_col_names, axis=1)

    y_test = test_emb_df['label'].tolist()
    X_test = test_emb_df.drop(meta_col_names, axis=1)

    model = GATNet(hidden_channels=8, heads=8)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # # scaling
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train.values)
    # X_test = scaler.transform(X_test.values)

    # TODO:change to torch tensor temporal graph data
    # TODO: add edge_index to dataclass for GNNs

