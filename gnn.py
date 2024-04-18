import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np
from conv import *
from gread import GlobalReadout
from sread import SemanticReadout
from torch_geometric.utils import to_dense_adj
from torch import Tensor
from kernel import Kernel_readout


class GNN(torch.nn.Module):

    def __init__(self, gnn_type, num_classes, num_nodefeats, num_edgefeats,
                    num_layer = 5, emb_dim = 300, drop_ratio = 0.5,  
                    read_op = 'sum', num_centers = 4, kernel = 'gaussian', agg = 'mlp'):
        super(GNN, self).__init__()

        self.num_classes = num_classes
        self.num_nodefeats = num_nodefeats
        self.num_edgefeats = num_edgefeats        

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.rep_dim = emb_dim * num_centers 

        self.read_op = read_op
        self.num_centers = num_centers
        self.node_encoder = torch.nn.Sequential(torch.nn.Linear(self.num_nodefeats, self.emb_dim),
                            torch.nn.BatchNorm1d(emb_dim))
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        ### GNN to generate node-level representations
        self.gnn_node = GNN_node(num_nodefeats, num_edgefeats, num_layer, emb_dim, drop_ratio = drop_ratio, gnn_type = gnn_type)
        
        ### Readout layer to generate graph-level representations
        if read_op == 'sread':
            self.read_op = SemanticReadout(self.emb_dim, read_op=self.read_op, num_position=self.num_centers)
        elif read_op == 'kr':
            self.read_op = Kernel_readout(self.emb_dim, n_centers=self.num_centers, kernel = kernel, agg = agg)
        else:
            self.read_op = GlobalReadout(self.num_nodefeats, self.emb_dim, read_op=read_op)
        
        if read_op == 'set2set': self.emb_dim *= 2
        if read_op == 'sread': self.emb_dim *= 4
        
        self.graph_mlp_pred = torch.nn.Sequential(torch.nn.Linear(self.emb_dim, self.emb_dim),
                                                  torch.nn.Sigmoid(), torch.nn.Linear(self.emb_dim, self.num_classes))
        self.graph_lin_pred = torch.nn.Linear(self.emb_dim, self.num_classes)
    

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        # ++++++ if backbone is SGC, using below code +++++
        # h_node = self.node_encoder(batched_data.filter_x)

        h_graph = self.read_op(h_node, batched_data)  
        return self.graph_mlp_pred(h_graph)

    def get_embedding(self, batched_data):
        h_node = self.gnn_node(batched_data)
        return F.normalize(self.read_op(h_node, batched_data))

    def get_alignment(self, batched_data):
        h_node = self.gnn_node(batched_data)
        return self.read_op.get_alignment(h_node)

    def get_aligncost(self, batched_data):
        h_node = self.node_encoder(batched_data.filter_x)
        return self.read_op.get_aligncost(h_node, batched_data.batch)

if __name__ == '__main__':
    GNN(num_classes = 10)
