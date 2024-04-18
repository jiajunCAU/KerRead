from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
import json
import json
import os.path as osp
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
import argparse
from sklearn.cluster import SpectralClustering, KMeans
from clustering_metric import clustering_metrics
from torch.nn import Sequential, Linear, ReLU
import math
from sklearn.preprocessing import normalize
from sklearn import metrics
from tqdm import *
from gread import GlobalReadout
from sread import SemanticReadout
from torch_geometric.utils import to_dense_adj
from torch import Tensor
from kernel import Kernel_readout
from aug import *

def clustering(Cluster, feature, true_labels, cluster = "kmeans"):
    feature = normalize(feature)
    if cluster == "kmeans":
        f_adj = feature
    predict_labels = Cluster.fit_predict(f_adj)
    cm = clustering_metrics(true_labels, predict_labels)
    db = -metrics.davies_bouldin_score(f_adj, predict_labels)
    acc, nmi, ari, q = cm.evaluationClusterModelFromLabel(tqdm)
    return acc, nmi, ari

def log_sum_exp(x, axis=None):
    """Log sum exp function

    Args:
        x: Input.
        axis: Axis over which to perform sum.

    Returns:
        torch.Tensor: log sum exp

    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y

def random_permute(X):
    """Randomly permutes a tensor.

    Args:
        X: Input tensor.

    Returns:
        torch.Tensor

    """
    X = X.transpose(1, 2)
    b = torch.rand((X.size(0), X.size(1))).to('cuda:3')
    idx = b.sort(0)[1]
    adx = torch.range(0, X.size(1) - 1).long()
    X = X[idx, adx[None, :]].transpose(1, 2)
    return X

def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    raise NotImplementedError(
        'Measure `{}` not supported. Supported: {}'.format(measure,
                                                           supported_measures))


def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.

    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.

    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.

    Returns:
        torch.Tensor

    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq

def local_global_loss_(l_enc, g_enc, edge_index, batch, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = torch.zeros((num_nodes, num_graphs)).to('cuda:3')
    neg_mask = torch.ones((num_nodes, num_graphs)).to('cuda:3')
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos

def adj_loss_(l_enc, g_enc, edge_index, batch):
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    adj = torch.zeros((num_nodes, num_nodes)).to('cuda:3')
    mask = torch.eye(num_nodes).to('cuda:3')
    for node1, node2 in zip(edge_index[0], edge_index[1]):
        adj[node1.item()][node2.item()] = 1.
        adj[node2.item()][node1.item()] = 1.

    res = torch.sigmoid((torch.mm(l_enc, l_enc.t())))
    res = (1-mask) * res
    loss = nn.BCELoss()(res, adj)
    return loss

class GlobalDiscriminator(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        
        self.l0 = nn.Linear(32, 32)
        self.l1 = nn.Linear(32, 32)

        self.l2 = nn.Linear(512, 1)
    def forward(self, y, M, data):

        adj = Variable(data['adj'].float(), requires_grad=False).to('cuda:3')
        # h0 = Variable(data['feats'].float()).to('cuda:3')
        batch_num_nodes = data['num_nodes'].int().numpy()
        M, _ = self.encoder(M, adj, batch_num_nodes)
        # h = F.relu(self.c0(M))
        # h = self.c1(h)
        # h = h.view(y.shape[0], -1)
        h = torch.cat((y, M), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))


class FF(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super().__init__()
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1)
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, hid_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, hid_dim)
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1, stride=1, padding=0)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)




class Encoder(torch.nn.Module):
    def __init__(self, read_op, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()
        if read_op == 'sread':
            self.read_op = SemanticReadout(dim, read_op=read_op, num_position=4, gamma=1)
        elif read_op == 'kr':
            self.read_op = Kernel_readout(dim, n_centers=4)
        else:
            self.read_op = GlobalReadout(num_features, dim, read_op=read_op)
        
        # if read_op == 'set2set': dim *= 2
        # if read_op == 'sread': dim *= 4
        
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch, data):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to('cuda:3')

        xs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)
        xpool = [self.read_op(x, data) for x in xs]
        
        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):

        device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to('cuda:3')
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to('cuda:3')
                x, _ = self.forward(x, edge_index, batch, data)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        try:
            num_features = dataset.num_features
        except:
            num_features = 1
        dim = 32

        self.encoder = Encoder(num_features, dim)

        self.fc1 = Linear(dim*5, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones(batch.shape[0]).to('cuda:3')

        x, _ = self.encoder(x, edge_index, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

class GcnInfomax(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(GcnInfomax, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)
    # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
    # self.global_d = MIFCNet(self.embedding_dim, mi_units)

    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, num_graphs):

    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0], 1).to('cuda:3')

    y, M = self.encoder(x, edge_index, batch)
    
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)

    mode='fd'
    measure='JSD'
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
 
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return local_global_loss + PRIOR


class simclr(nn.Module):
  def __init__(self, read_op, dataset_num_features, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(simclr, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(read_op, dataset_num_features, hidden_dim, num_gc_layers)
    if read_op == 'set2set': self.embedding_dim*=2
    if read_op == 'sread': self.embedding_dim*=4
    
    self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, num_graphs, data):

    # batch_size = data.num_graphs
    if x is None:
        x = torch.ones(batch.shape[0], 1).to('cuda:3')

    y, M = self.encoder(x, edge_index, batch, data)
    
    y = self.proj_head(y)
    
    return y

  def loss_cal(self, x, x_aug):

    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()

    return loss


import random
def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main(args):
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.dataset in ['MUTAG','DD','PROTEINS','NCI1','Mutagenicity','IMDB-BINARY','IMDB-MULTI', 'COLLAB']:
        dataset = TUDataset('./dataset', name=args.dataset)
        node_features = []
        dataset_num_features = max(dataset.num_features, 1)
        num_edgefeats = max(1, dataset.num_edge_labels)
    elif args.dataset in ['ogbg-molhiv']:
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(name = 'ogbg-molhiv') 
        dataset_num_features = max(1,dataset.x.shape[1])
        num_edgefeats = max(1, dataset.num_edge_features)
    num_classes = int(dataset.data.y.max()) + 1
    Cluster = KMeans(n_clusters = num_classes)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    print('================')
    print('lr: {}'.format(args.learning_rate))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.emb_dim))
    print('num_gc_layers: {}'.format(args.num_layer))
    print('================')
    print(device)
    # model = InfoGraph(args.read_op, dataset_num_features, args.emb_dim, args.num_layer).to('cuda:3')
    model = simclr(args.read_op, dataset_num_features, args.emb_dim, args.num_layer).to('cuda:3')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader)
    best_acc = 0
    for epoch in range(1, args.max_epochs+1):
        loss_all = 0
        model.train()
        for data in dataloader:

            # print('start')
            # data, data_aug = data
            # data_aug = data.clone()
            data_aug = permute_edges(data)

            optimizer.zero_grad()
            
            node_num = data.y.shape[0]
            data = data.to('cuda:3')
            x = model(data.x, data.edge_index, data.batch, data.num_graphs, data)

            data_aug = data_aug.to('cuda:3')

            '''
            print(data.edge_index)
            print(data.edge_index.size())
            print(data_aug.edge_index)
            print(data_aug.edge_index.size())
            print(data.x.size())
            print(data_aug.x.size())
            print(data.batch.size())
            print(data_aug.batch.size())
            pdb.set_trace()
            '''

            x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs, data_aug)

            # print(x)
            # print(x_aug)
            loss = model.loss_cal(x, x_aug)
            # print(loss)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
            # print('batch')
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

        if epoch % 1 == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader)
            acc, nmi, ari = clustering(Cluster, emb, y)
            print('acc:{}, nmi:{}, ari:{}'.format(acc, nmi, ari))
            if acc > best_acc:
                best_acc, best_nmi, best_ari = acc, nmi, ari

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=False)
    parser.add_argument('--gnn', type=str, default='gcn',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gcn)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    
    # SSRead parameters
    parser.add_argument('--read_op', type=str, default='sum',
                        help='graph readout operation (default: sum)')
    parser.add_argument('--num_position', type=int, default=2,
                        help='number of centers (default: 4)')
    parser.add_argument('--tao', type=float, default=1,
                        help='smoothing parameter for soft semantic alignment (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=1,
                        help='smoothing parameter for soft semantic alignment (default: 0.01)')
    parser.add_argument('--init', type=str, default='trans',
                        help='graph readout operation (default: sum)')
    parser.add_argument('--update', type=str, default='tt',
                        help='graph readout operation (default: sum)')
    parser.add_argument('--agg', type=str, default='lin',
                        help='graph readout operation (default: sum)')
    
    # Optimization parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='maximum number of epochs to train (default: 500)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--split', type=float, default=0.1,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='initial learning rate of the optimizer (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--early_stop', type=int, default=30,
                        help='patience for early stopping criterion (default: 50)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for reproducibility (default: 42)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')

    # Dataset parameters
    parser.add_argument('--datapath', type=str, default="./dataset",
                        help='path to the directory of datasets (default: ./dataset)')
    parser.add_argument('--dataset', type=str, default="NCI1",
                        help='dataset name (default: NCI1)')
    parser.add_argument('--num_fold', type=int, default=10,
                        help='number of fold for cross-validation (default: 10)')
    # kernel readout parameters
    parser.add_argument('--kernel', type=str, default="rbf",
                        help='kernel choice')

    parser.add_argument('--aug', type=str, default='dnodes')
    
    args = parser.parse_args()
    args.init = 'trans'
    main(args) 
