import random
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN
from utils import get_kfold_idx_split, str2bool
import argparse
import time
import numpy as np
import copy
import os
from tqdm import *
from utils import *

mcls_criterion = torch.nn.CrossEntropyLoss()

def train(model, device, loader, optimizer_gnn, optimizer_seg, args):
    model.train()
    all_loss = 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch)
        if args.dataset == 'ogbg-molhiv':
            y = batch.y.squeeze(1)
        else: y = batch.y
        is_labeled = y == y
        optimizer_gnn.zero_grad()
        
        
        pred_loss = mcls_criterion(pred.to(torch.float32)[is_labeled], y[is_labeled])
        pred_loss.backward(retain_graph = True)
        optimizer_gnn.step()

        if args.read_op == 'sread':
            optimizer_seg.zero_grad()
            align_loss = model.get_aligncost(batch)
            align_loss.backward(retain_graph=True)
            optimizer_seg.step()

        all_loss += pred_loss.item()
    return all_loss


def eval(model, device, loader):
    model.eval()
    
    y_true, y_pred = [], []
    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred= model(batch)
            pred = torch.max(pred, dim=1)[1]
                
        y_true.append(batch.y.view(pred.shape))
        y_pred.append(pred)
    
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    correct = y_true == y_pred
    return {'acc': correct.sum().item()/correct.shape[0]}


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
    # 计算不同中心度指标
    # degree_centrality = nx.degree_centrality(G)
    # print(sorted(nx.closeness_centrality(G).values()))
    closeness_centrality = torch.FloatTensor(list(nx.closeness_centrality(G).values()))
    betweenness_centrality = torch.FloatTensor(list(nx.betweenness_centrality(G).values()))
    # eigenvector_centrality = torch.FloatTensor(list(nx.eigenvector_centrality(G).values()))
    pagerank_centrality = torch.FloatTensor(list(nx.pagerank(G).values()))
    adjacency_matrix = nx.adjacency_matrix(G)
    adjacency_matrix = adjacency_matrix.todense()
    F = preprocess_graph(adjacency_matrix)
    if graph.x == None:
        feat = np.ones((adjacency_matrix.shape[0],10))
    else:
        feat = graph.x.numpy()
    filtered_x = copy.deepcopy(feat)
    for _ in range(2):
        filtered_x = np.matmul(F, filtered_x)
    return torch.FloatTensor(np.array(filtered_x)), closeness_centrality, betweenness_centrality, pagerank_centrality



class AddGraphIdTransform:  
    def __init__(self):
        self.graph_id = 0
        
    def __call__(self, data):
        nodes = data.num_nodes
        data.filter_x, data.c1, data.c2, data.c3 = get_filtered_feat(data)
        data.nodes = nodes
        data.graph_id = self.graph_id
        self.graph_id += 1
        return data

def tsne(embedding_tsne, label_array, name):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE(n_components=2, random_state=42)
    embedding_tsne = tsne.fit_transform(embedding_tsne)
    colors = ['steelblue','darkorange']
    plt.figure(figsize=(10, 8))
    for i, c in zip(np.unique(label_array), colors):
        plt.scatter(embedding_tsne[label_array == i, 0], embedding_tsne[label_array == i, 1], label=str(i), color=c)
    # plt.legend()
    plt.savefig(f'./tsne/{name}.pdf',bbox_inches='tight')
    plt.show()
    

def get_graph_embedding(args, model, test_loader):
    embedding_list = []
    y_list = []
    for step, batch in enumerate(test_loader):
        batch = batch.to(args.device)
        with torch.no_grad():
            embedding = model.get_embedding(batch)
            y = batch.y.cpu().detach()
            embedding_list.append(embedding.cpu().detach())
            y_list.append(y)
    embedding_list = torch.cat(embedding_list,dim=0)
    y_list = torch.cat(y_list,dim=0)
    print(embedding_list.shape)
    tsne(embedding_list, y_list, f'{args.read_op}_{args.dataset}')
        
            

def main(args):
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    if args.dataset in ['MUTAG','DD','PROTEINS','NCI1','Mutagenicity','IMDB-BINARY','IMDB-MULTI', 'COLLAB']:
        transform = AddGraphIdTransform()
        dataset = TUDataset(root = args.datapath, name = args.dataset, pre_transform = transform)
        node_features = []
        num_nodefeats = max(1, dataset.num_node_labels)
        num_edgefeats = max(1, dataset.num_edge_labels)
    elif args.dataset in ['ogbg-molhiv']:
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(name = 'ogbg-molhiv') 
        num_nodefeats = max(1,dataset.x.shape[1])
        num_edgefeats = max(1, dataset.num_edge_features)
    
    num_classes = int(dataset.data.y.max()) + 1
    split_idx = get_kfold_idx_split(dataset, num_fold=args.num_fold, random_state=args.seed)
    valid_list, test_list = [], []
    times = []

    if args.dataset == 'PROTEINS':
        num_nodefeats = dataset.filter_x.shape[1] - 1
    elif args.dataset in ['IMDB-BINARY','IMDB-MULTI']:
        num_nodefeats = 10
    
    for fold_idx in [6]: 
        train_loader = DataLoader(dataset[split_idx["train"][fold_idx]], batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
        valid_loader = DataLoader(dataset[split_idx["augvalid"][fold_idx]], batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
        test_loader = DataLoader(dataset[split_idx["test"][fold_idx]], batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
        model = GNN(gnn_type = args.gnn, num_classes = num_classes, num_nodefeats = num_nodefeats, num_edgefeats = num_edgefeats,
                        num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, 
                        read_op = args.read_op, num_centers = args.num_centers, kernel = args.kernel).to(device)
        
        optimizer_gnn = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        if args.read_op == 'sread':
            optimizer_seg = optim.Adam(model.read_op.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) 
        else:
            optimizer_seg = None
        
        valid_curve, test_curve = [], []
        best_valid_perf, no_improve_cnt = -np.inf, 0
        t = time.time()
        loss = 0
        for epoch in range(1, args.max_epochs + 1):

            # if os.path.exists(f'./checkpoints/{args.dataset}_{args.gnn}_{args.read_op}_{fold_idx}.pth'):
            #     model = torch.load(f'./checkpoints/{args.dataset}_{args.gnn}_{args.read_op}_{fold_idx}.pth')
            #     valid_perf = eval(model, device, valid_loader)
            #     test_perf = eval(model, device, test_loader)
            #     print('fold:%3d\t epoch:%3d\tvalid acc:%.6f\ttest acc:%.6f'%(fold_idx, epoch, valid_perf['acc'], test_perf['acc']))
            #     break
            # else:
            loss = train(model, device, train_loader, optimizer_gnn, optimizer_seg, args)
            valid_perf = eval(model, device, valid_loader)
            test_perf = eval(model, device, test_loader)

            
            print('%3d\t%.6f\t%.6f'%(epoch, valid_perf['acc'], test_perf['acc']))

            valid_curve.append(valid_perf['acc'])
            test_curve.append(test_perf['acc'])

            if no_improve_cnt > args.early_stop:
                break
            elif valid_perf['acc'] > best_valid_perf:
                best_valid_perf = valid_perf['acc']
                no_improve_cnt = 0
                best_model = model
                torch.save(best_model, f'./checkpoints/{args.dataset}_{args.gnn}_{args.read_op}_{fold_idx}.pth')
            else:
                no_improve_cnt += 1
        if loss != 0:
            torch.save(best_model, f'./checkpoints/{args.dataset}_{args.gnn}_{args.read_op}_{fold_idx}.pth')
            times.append((time.time() - t)/epoch)
            best_val_epoch = np.argmax(np.array(valid_curve))
            print('%2d-fold Valid\t%.6f'%(fold_idx+1, valid_curve[best_val_epoch]))
            print('%2d-fold Test\t%.6f'%(fold_idx+1, test_curve[best_val_epoch]))

            valid_list.append(valid_curve[best_val_epoch])
            test_list.append(test_curve[best_val_epoch])
          
    # valid_list = np.array(valid_list)*100
    # test_list = np.array(test_list)*100
    # times = np.array(times)*1000
    # print('Valid Acc:{:.4f}, Std:{:.4f}, Test Acc:{:.4f}, Std:{:.4f}'.format(np.mean(valid_list), np.std(valid_list), np.mean(test_list), np.std(test_list)))
    # print('Mean time :{:.4f}, Std:{:.4f} (ms)'.format(np.mean(times), np.std(times)))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gnn', type=str, default='gcn',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gcn)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--read_op', type=str, default='sum',
                        help='graph readout operation (default: sum)')
    
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--max_epochs', type=int, default=300,
                        help='maximum number of epochs to train (default: 500)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--split', type=float, default=0.1,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='initial learning rate of the optimizer (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--early_stop', type=int, default=50,
                        help='patience for early stopping criterion (default: 50)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for reproducibility (default: 42)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')

    parser.add_argument('--datapath', type=str, default="./dataset",
                        help='path to the directory of datasets (default: ./dataset)')
    parser.add_argument('--dataset', type=str, default="NCI1",
                        help='dataset name (default: NCI1)')
    parser.add_argument('--num_fold', type=int, default=10,
                        help='number of fold for cross-validation (default: 10)')

    # kernel readout parameters
    parser.add_argument('--kernel', type=str, default="gaussian",
                        help='kernel choice')
    parser.add_argument('--num_centers', type=int, default=4,
                        help='number of centers (default: 4)')
    parser.add_argument('--init', type=str, default='linear',
                        help='initializing strategy of adaptive centers (default: trans)')
    parser.add_argument('--update', type=str, default='tt',
                        help='updating strategy of adaptive centers')
    parser.add_argument('--agg', type=str, default='mlp',
                        help='strategy of aggregating graph vector of multiple centers')
    args = parser.parse_args()

    main(args) 


