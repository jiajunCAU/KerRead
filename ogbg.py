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
        
        # print(pred.shape, batch.y.shape, y.shape)
        
        pred_loss = mcls_criterion(pred.to(torch.float32)[is_labeled], y[is_labeled])
        pred_loss.backward(retain_graph = True)
        optimizer_gnn.step()

        if args.read_op == 'sread':
            optimizer_seg.zero_grad()
            align_loss = model.get_aligncost(batch)
            align_loss.backward(retain_graph=True)
            optimizer_seg.step()

        all_loss += pred_loss.item()
    # print(model.rbf_read.centers[0], model.rbf_read.beta)
    return all_loss

from sklearn.metrics import roc_auc_score, average_precision_score
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
    auc = roc_auc_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    ap = average_precision_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())    
    return {'acc': correct.sum().item()/correct.shape[0], 'auc':auc, 'ap':ap}


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

class AddGraphIdTransform:  
    def __init__(self):
        self.graph_id = 0
        
    def __call__(self, data):
        nodes = data.num_nodes
        data.filter_x = get_filtered_feat(data)
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
    # plt.title('t-SNE Visualization with Morandi Color Scheme')
    plt.legend()
    plt.savefig(f'./{name}.pdf',bbox_inches='tight')
    plt.show()
    

def get_graph_embedding(args, model, test_loader):
    # embedding_loader = DataLoader(dataset, batch_size = len(dataset), num_workers = args.num_workers)
    # DataLoader(dataset[split_idx["train"][fold_idx]], batch_size = 128, shuffle = True, num_workers = args.num_workers)
    for step, batch in enumerate(test_loader):
        batch = batch.to(args.device)
        with torch.no_grad():
            rbf_embedding, sum_embedding = model.get_embedding(batch)
            y = batch.y.cpu().detach().numpy()
            rbf_embedding = rbf_embedding.cpu().detach().numpy()
            sum_embedding = sum_embedding.cpu().detach().numpy()
            tsne(rbf_embedding, y, f'kernel_{args.dataset}')
            tsne(sum_embedding, y, f'sum_{args.dataset}')
        break
            
            

def main(args):
    print(args)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.dataset in ['MUTAG','DD','PROTEINS','NCI1','Mutagenicity','IMDB-B','IMDB-M']:
        transform = AddGraphIdTransform()
        dataset = TUDataset(root = args.datapath, name = args.dataset, pre_transform = transform)
        node_features = []
        num_nodefeats = max(1, dataset.num_node_labels)
        num_edgefeats = max(1, dataset.num_edge_labels)
    elif args.dataset in ['ogbg-molhiv', 'ogbg-ppa', 'ogbg-molpcba']:
        transform = AddGraphIdTransform()
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(name = args.dataset,pre_transform = transform) 
        num_nodefeats = max(1,dataset.x.shape[1])
        num_edgefeats = max(1, dataset.num_edge_features)


    num_classes = int(dataset.data.y.max()) + 1
    split_idx = get_kfold_idx_split(dataset, num_fold=args.num_fold, random_state=args.seed)
    node_metric_feat = None
    valid_list, test_list = {'acc':[], 'auc':[], 'ap':[]}, {'acc':[], 'auc':[], 'ap':[]}
    times = []
    folds_nums = 10
    for fold_idx in range(10): 
        train_loader = DataLoader(dataset[split_idx["train"][fold_idx]], batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
        valid_loader = DataLoader(dataset[split_idx["augvalid"][fold_idx]], batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
        test_loader = DataLoader(dataset[split_idx["test"][fold_idx]], batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
        model = GNN(gnn_type = args.gnn, num_classes = num_classes, num_nodefeats = num_nodefeats, num_edgefeats = num_edgefeats,
                        num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, 
                        read_op = args.read_op,).to(device)
        
        optimizer_gnn = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        if args.read_op == 'sread':
            optimizer_seg = optim.Adam(model.read_op.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) 
        else:
            optimizer_seg = None
        
        valid_curve, test_curve = {'acc':[], 'auc':[], 'ap':[]}, {'acc':[], 'auc':[], 'ap':[]}
        best_valid_perf, no_improve_cnt = -np.inf, 0
        t = time.time()
        for epoch in range(1, args.max_epochs + 1):
            loss = train(model, device, train_loader, optimizer_gnn, optimizer_seg, args)
            valid_perf = eval(model, device, valid_loader)
            test_perf = eval(model, device, test_loader)
            print('******************************************\n')
            print('epoch:{}\t valid acc{:.4f}\t valid auc{:.4f}\t valid ap{:.4f}\n'.format(epoch, valid_perf['acc'],  valid_perf['auc'],  valid_perf['ap']))
            print('loss:{:.4f}\t test acc{:.4f}\t test auc{:.4f}\t test ap{:.4f}\n'.format(loss, test_perf['acc'],  test_perf['auc'],  test_perf['ap'], ))
            print('******************************************\n')
            valid_curve['acc'].append(valid_perf['acc'])
            test_curve['acc'].append(test_perf['acc'])
            valid_curve['auc'].append(valid_perf['auc'])
            test_curve['auc'].append(test_perf['auc'])
            valid_curve['ap'].append(valid_perf['ap'])
            test_curve['ap'].append(test_perf['ap'])

            if args.dataset == 'ogbg-molhiv':
                if no_improve_cnt > args.early_stop:
                    break
                elif valid_perf['auc'] > best_valid_perf:
                    best_valid_perf = valid_perf['auc']
                    no_improve_cnt = 0
                else:
                    no_improve_cnt += 1
            else:
                if no_improve_cnt > args.early_stop:
                    break
                elif valid_perf['acc'] > best_valid_perf:
                    best_valid_perf = valid_perf['acc']
                    no_improve_cnt = 0
                else:
                    no_improve_cnt += 1
        # graph_embedding = get_graph_embedding(args, model, valid_loader)
        times.append((time.time() - t)/epoch)
        if args.dataset == 'ogbg-molhiv':
            best_val_epoch = np.argmax(np.array(valid_curve['auc']))
        else:
            best_val_epoch = np.argmax(np.array(valid_curve['acc']))
        print('%2d-fold Valid\t%.6f'%(fold_idx+1, valid_curve['acc'][best_val_epoch]))
        print('%2d-fold Test\t%.6f'%(fold_idx+1, test_curve['acc'][best_val_epoch]))

        valid_list['acc'].append(valid_curve['acc'][best_val_epoch])
        test_list['acc'].append(test_curve['acc'][best_val_epoch])
        valid_list['auc'].append(valid_curve['auc'][best_val_epoch])
        test_list['auc'].append(test_curve['auc'][best_val_epoch])
        valid_list['ap'].append(valid_curve['ap'][best_val_epoch])
        test_list['ap'].append(test_curve['ap'][best_val_epoch])
         
    acc_valid_list = np.array(valid_list['acc'])*100
    acc_test_list = np.array(test_list['acc'])*100
    auc_valid_list = np.array(valid_list['auc'])*100
    auc_test_list = np.array(test_list['auc'])*100
    ap_valid_list = np.array(valid_list['ap'])*100
    ap_test_list = np.array(test_list['ap'])*100
    
    times = np.array(times)*1000
    print('Valid Acc:{:.4f}, Std:{:.4f}, Test Acc:{:.4f}, Std:{:.4f}'.format(np.mean(acc_valid_list), np.std(acc_valid_list), np.mean(acc_test_list), np.std(acc_test_list)))
    print('Valid AUC:{:.4f}, Std:{:.4f}, Test AUC:{:.4f}, Std:{:.4f}'.format(np.mean(auc_valid_list), np.std(auc_valid_list), np.mean(auc_test_list), np.std(auc_test_list)))
    print('Valid AP:{:.4f}, Std:{:.4f}, Test AP:{:.4f}, Std:{:.4f}'.format(np.mean(ap_valid_list), np.std(ap_valid_list), np.mean(ap_test_list), np.std(ap_test_list)))
    print('Mean time :{:.4f}, Std:{:.4f} (ms)'.format(np.mean(times), np.std(times)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model parameters
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
    parser.add_argument('--num_position', type=int, default=4,
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
    parser.add_argument('--batch_size', type=int, default=128*4,
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


    args = parser.parse_args()
    main(args) 


