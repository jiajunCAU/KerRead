import networkx as nx
import torch
from torch_geometric.utils import from_networkx, to_networkx
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GraphConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree
import scipy.sparse as sp
import numpy as np
import math
from gread import GlobalReadout
from torch_scatter import scatter
from conv import *


class Kernel_readout(torch.nn.Module):
    def __init__(self, hidden_size, n_centers = 4, kernel = 'gaussian', agg = 'mlp', init = 'trans', update = 'tt'):
        super(Kernel_readout, self).__init__()
        self.hidden_size = hidden_size
        self.n_centers = n_centers
        self.kernel = kernel
        self.agg = agg
        update_map = {
            'tt':[True,True], 'tf':[True, False], 'ft':[False, True], 'ff':[False,False]
        }
        update_list = update_map[update]
        self.centers = torch.nn.Parameter(torch.ones(1, self.n_centers), requires_grad = update_list[0]) 
        self.agg = agg

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.n_centers*self.hidden_size, self.n_centers*self.hidden_size),
            torch.nn.ReLU(),torch.nn.Linear(self.n_centers*self.hidden_size, self.hidden_size))
        self.linear = torch.nn.Linear(self.n_centers*self.hidden_size, self.hidden_size)
        self.w = torch.nn.Linear(self.n_centers, 1)

        self.init = init
        # torch.nn.init.kaiming_uniform_(self.centers)
        self.trans_1_n = torch.nn.Linear(self.hidden_size, 1)
        self.x_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size), 
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.center_list = [torch.nn.Linear(self.hidden_size, 1) for _ in range(self.n_centers)]

        self.alpha = torch.nn.Parameter(torch.ones(1, 1), requires_grad = True)
        self.beta = torch.nn.Parameter(torch.ones(1, 1), requires_grad = True)

        self.d = 2
    
    def forward(self, x, batch):
        # x    # n x d
        # c    # 1 x k
        # nodes = batch.nodes
        c = self.centers
        x = self.x_encoder(x)
        weight = self.trans_1_n(x) 
        n, d, k = x.shape[0], x.shape[1], self.centers.shape[1]
        if self.init == 'trans':
            c = (c.T) @ (weight.T)
            c = c.reshape(1,k,n)
        elif self.init == 'expand':
            c = c.expand(n, k).reshape(1,k,n)
        elif self.init == 'linear':
            c = torch.stack([self.center_list[i](x) for i in range(self.n_centers)]).reshape(1,k,n)
        
        x = x.T.reshape(d,1,n)
        g_nums = batch.batch.max() + 1

        if self.kernel == 'gaussian':
            res = torch.pow((x - c), 2)
            # res = F.normalize(res, 0)
            res = scatter(res, batch.batch, dim=2, reduce='sum').permute(2,1,0)
            res = torch.exp(-1*torch.pow(res,0.5)/self.beta )
            # res = F.normalize(res, dim=2)

        elif self.kernel == 'lap':
            res = torch.abs(x - c)
            res = scatter(res, batch.batch, dim=2, reduce='sum').permute(2,1,0)
            res = torch.exp(-1*res/self.beta)

        elif self.kernel == 'linear':
            c = c.reshape(k,n)
            x = x.squeeze().T
            res = []
            for i in range(k):
                temp = scatter((c[i].squeeze().unsqueeze(1))*x, batch.batch, dim=0, reduce='sum')
                res.append(temp)
            res = torch.stack(res,dim=0).permute(1,0,2)

        elif self.kernel == 'sigmoid':
            c = c.reshape(k,n)
            x = x.squeeze().T
            res = []
            for i in range(k):
                temp = scatter((c[i].squeeze().unsqueeze(1))*x, batch.batch, dim=0, reduce='sum')
                temp = F.tanh(self.alpha*temp + self.beta)
                res.append(temp)
            res = torch.stack(res).permute(1,0,2)
            
        elif self.kernel == 'poly':
            c = c.reshape(k,n)
            x = x.squeeze().T
            res = []
            for i in range(k):
                temp = scatter((c[i].squeeze().unsqueeze(1))*x, batch.batch, dim=0, reduce='sum')
                temp = 1 + (temp)**self.d
                res.append(temp)
            res = torch.stack(res).permute(1,0,2)
            
        if self.agg == 'mlp':
            res = res.reshape(g_nums,-1)
            g = self.mlp(F.normalize(res))
        elif self.agg == 'weight':
            g = self.w(res.permute(2,1,0)).squeeze().T
        elif self.agg == 'mean':
            g = res.mean(1).squeeze()
        return g

    def scale(self, z):
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
        return z_scaled
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
