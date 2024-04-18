import torch
import argparse

import networkx as nx
import scipy.sparse as sp
import numpy as np
from torch_geometric.utils import to_networkx
from sklearn.model_selection import StratifiedKFold 

def get_kfold_idx_split(dataset, num_fold=10, random_state=0):
    '''
    StratifiedKFold function
    '''
    skf = StratifiedKFold(num_fold, shuffle=True, random_state=random_state)

    train_indices, augvalid_indices, test_indices = [], [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).long())
    valid_indices = [test_indices[i - 1] for i in range(num_fold)]

    for i in range(num_fold):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[valid_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
        
        augvalid_mask = torch.ones(len(dataset), dtype=torch.uint8)
        augvalid_mask[test_indices[i]] = 0
        augvalid_indices.append(augvalid_mask.nonzero(as_tuple=False).view(-1))

    return {"train": train_indices, "valid": valid_indices, "augvalid": augvalid_indices, "test": test_indices}

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid boolean value')


def preprocess_graph(adj, norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj
    d = np.zeros((adj.shape[0],adj.shape[0]))
    for i in range(adj.shape[0]): d[i, i] = adj_.sum(1)[i]
    rowsum = np.array(adj_.sum(1))
    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized
    return adj_normalized.todense()

import copy
def get_filtered_feat(graph):
    G = to_networkx(graph)
    adjacency_matrix = nx.adjacency_matrix(G)
    adjacency_matrix = adjacency_matrix.todense()
    F = preprocess_graph(adjacency_matrix)
    # F = adjacency_matrix
    if graph.x == None:
        feat = np.ones((adjacency_matrix.shape[0],10))
    else:
        feat = graph.x.numpy()
    filtered_x = copy.deepcopy(feat)
    for _ in range(2):
        filtered_x = np.matmul(F, filtered_x)
    return torch.FloatTensor(np.array(filtered_x))
