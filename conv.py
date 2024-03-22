import networkx as nx
import torch
from torch_geometric.utils import from_networkx, to_networkx
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphUNet, GATConv, SAGEConv, GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree
import scipy.sparse as sp
import numpy as np
import math
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_sparse import SparseTensor, set_diag
from torch_geometric.typing import (Adj, NoneType, OptPairTensor, OptTensor,
                                    Size)
from typing import Optional, Tuple, Union
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class Unet_Conv(MessagePassing):
    def __init__(self, num_edgefeats, emb_dim):
        super(Unet_Conv, self).__init__()
        self.conv = GraphUNet(in_channels=emb_dim, hidden_channels=emb_dim, out_channels = emb_dim, depth=4)
        self.edge_encoder = torch.nn.Linear(num_edgefeats, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.conv(x, edge_index)
        return out

class GAT_Conv(MessagePassing):
    def __init__(self, num_edgefeats, emb_dim):
        super(GAT_Conv, self).__init__()
        self.conv = GATConv(in_channels=emb_dim, out_channels=emb_dim)
        self.edge_encoder = torch.nn.Linear(num_edgefeats, 1)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.conv(x, edge_index, edge_embedding)
        return out
        
    
class SAGE_Conv(MessagePassing):
    def __init__(self, num_edgefeats, emb_dim):
        super(SAGE_Conv, self).__init__(aggr = "add")
        self.conv = SAGEConv(in_channels=emb_dim, out_channels=emb_dim)
        self.edge_encoder = torch.nn.Linear(num_edgefeats, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_encoder.weight)
        self.lin_l = torch.nn.Linear(emb_dim, emb_dim)
        self.lin_r = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)    
        out = self.propagate(edge_index, x=x, edge_attr=edge_embedding)
        out = self.lin_l(out)
       
        return out

    def message(self, x_j, edge_attr):
        return  F.relu(x_j + edge_attr)
    
    def update(self, aggr_out):
        return aggr_out


class GIN_Conv(MessagePassing):
    def __init__(self, num_edgefeats, emb_dim):
        super(GIN_Conv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        
        self.edge_encoder = torch.nn.Linear(num_edgefeats, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_encoder.weight)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)    
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out


class GCN_Conv(MessagePassing):
    def __init__(self, num_edgefeats, emb_dim):
        super(GCN_Conv, self).__init__(aggr='add')
        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = torch.nn.Linear(num_edgefeats, emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_encoder.weight)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        if edge_attr.shape[1] != self.emb_dim:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = edge_attr
        row, col = edge_index

        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    def __init__(self, num_nodefeats, num_edgefeats, num_layer, emb_dim, drop_ratio = 0.5, gnn_type = 'gcn'):

        super(GNN_node, self).__init__()
        self.num_nodefeats = num_edgefeats

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        self.node_encoder = torch.nn.Linear(num_nodefeats, emb_dim)
        torch.nn.init.xavier_uniform_(self.node_encoder.weight)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GIN_Conv(num_edgefeats, emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCN_Conv(num_edgefeats, emb_dim))
            elif gnn_type == 'gat':
                self.convs.append(GAT_Conv(num_edgefeats, emb_dim))
            elif gnn_type == 'sage':
                self.convs.append(SAGE_Conv(num_edgefeats, emb_dim))
            elif gnn_type == 'unet':
                self.convs.append(Unet_Conv(num_edgefeats, emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))
                
            self.emb_layer = torch.nn.Embedding(100, num_nodefeats)
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
            self.atom_encoder = AtomEncoder(emb_dim = emb_dim)
            self.bond_encoder = BondEncoder(emb_dim = emb_dim)
            

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
    
        if x == None:
            x = torch.ones_like(batch, dtype=torch.float).unsqueeze(dim=1).repeat(1,10)
        if edge_attr == None:
            edge_attr = torch.ones_like(edge_index.transpose(0, 1), dtype=torch.float)[:, 0:1]
        
        ### computing input node embedding
        if str(x.dtype) == 'torch.int64':   
            h_list = [self.atom_encoder(x)]
            edge_attr = self.bond_encoder(edge_attr)
        else:
            h_list = [self.node_encoder(x)]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            h_list.append(h)

        node_representation = h_list[-1]

        return node_representation

if __name__ == "__main__":
    pass