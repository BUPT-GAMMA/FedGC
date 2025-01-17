import os.path as osp
import torch
import random
import sys
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch_geometric
import torch_sparse
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import Planetoid
from deeprobust.graph.utils import get_train_val_test
import torch_geometric.transforms as T
from deeprobust.graph.data import Dataset
from torch_geometric.utils import to_undirected


def parse_index_file(filename: str) -> list:
    """
    This function reads and parses an index file

    Args:
    filename: (str) - name or path of the file to parse

    Return:
    index: (list) - list of integers, each integer in the list represents int of the lines lines of the input file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

    

def normalize(mx: sp.csc_matrix) -> sp.csr_matrix:
    """
    This function is to row-normalize sparse matrix for efficient computation of the graph
    
    Argument:
    mx: (sparse matrix) - Input sparse matrix to row-normalize. 

    Return:
    mx: (sparse matrix) - Returns the row-normalized sparse matrix.

    Note:
    Row-normalizing is usually done in graph algorithms to enable equal node contributions regardless of the node's degree 
    and to stabilize, ease numerical computations
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def mask_to_index(index, size):
    all_idx = np.arange(size)
    return all_idx[index]

class Pyg2Dpr(Dataset):
    def __init__(self, pyg_data, **kwargs):
        try:
            splits = pyg_data.get_idx_split()
        except:
            pass

        dataset_name = pyg_data.name
        pyg_data = pyg_data[0]
        n = pyg_data.num_nodes

        if dataset_name == 'ogbn-arxiv': # symmetrization
            pyg_data.edge_index = to_undirected(pyg_data.edge_index, pyg_data.num_nodes)

        self.adj = sp.csr_matrix((np.ones(pyg_data.edge_index.shape[1]),
                                  (pyg_data.edge_index[0], pyg_data.edge_index[1])), shape=(n, n))

        self.features = pyg_data.x.numpy()
        self.labels = pyg_data.y.numpy()

        if len(self.labels.shape) == 2 and self.labels.shape[1] == 1:
            self.labels = self.labels.reshape(-1) # ogb-arxiv needs to reshape

        if hasattr(pyg_data, 'train_mask'):
            # for fixed split
            self.idx_train = mask_to_index(pyg_data.train_mask, n)
            self.idx_val = mask_to_index(pyg_data.val_mask, n)
            self.idx_test = mask_to_index(pyg_data.test_mask, n)
            self.name = 'Pyg2Dpr'
        else:
            try:
                # for ogb
                self.idx_train = splits['train']
                self.idx_val = splits['valid']
                self.idx_test = splits['test']
                self.name = 'Pyg2Dpr'
            except:
                # for other datasets
                self.idx_train, self.idx_val, self.idx_test = get_train_val_test(
                    nnodes=n, val_size=0.1, test_size=0.8, stratify=self.labels)



def load_data(dataset_str: str) -> tuple:
    '''
    This function loads input data from gcn/data directory

    Argument:
    dataset_str: Dataset name

    Return: 
    All data input files loaded (as well as the training/test data).

    Note:
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.
    '''
    if dataset_str in ['cora', 'citeseer']:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset_str)
        dataset = Planetoid(path, dataset_str)
        dataset.transform = T.NormalizeFeatures()
        dpr_data = Pyg2Dpr(dataset)
        adj, features, labels = dpr_data.adj, dpr_data.features, dpr_data.labels
        idx_train, idx_val, idx_test = torch.LongTensor(np.array(dpr_data.idx_train)), torch.LongTensor(np.array(dpr_data.idx_val)), \
                                       torch.LongTensor(np.array(dpr_data.idx_test))
        adj = torch_sparse.tensor.SparseTensor.from_dense(torch.Tensor(adj.todense()))


    elif dataset_str in ['pubmed']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("../../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("../../data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        number_of_nodes=adj.shape[0]


        idx_test = test_idx_range.tolist()
        idx_train = range(len(y)) #140
        idx_val = range(len(y), len(y)+500)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)


        features=torch.tensor(features.toarray()).float()
        adj = torch.tensor(adj.toarray()).float()
        adj = torch_sparse.tensor.SparseTensor.from_dense(adj)
        labels=torch.tensor(labels)
        labels=torch.argmax(labels,dim=1)

    elif dataset_str in ['ogbn-arxiv']:
        from ogb.nodeproppred import PygNodePropPredDataset

        # Download and process data at './dataset/.'

        dataset = PygNodePropPredDataset(name=dataset_str,
                                     transform=torch_geometric.transforms.ToSparseTensor())

        split_idx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        data = dataset[0]
        
        features = data.x
        labels = np.array(data.y.reshape(-1))
        if dataset_str == 'ogbn-arxiv':
            adj = data.adj_t.to_symmetric()
        else:
            adj = data.adj_t

        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)
            
    elif dataset_str == 'reddit':
        from dgl.data import RedditDataset
        data = RedditDataset()
        g = data[0]
        num_classes = data.num_classes
        
        adj = torch_sparse.tensor.SparseTensor.from_edge_index(g.edges())

        features = g.ndata['feat']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        
        idx_train = (train_mask == True).nonzero().view(-1)
        idx_val = (val_mask == True).nonzero().view(-1)
        idx_test = (test_mask == True).nonzero().view(-1)
        
        labels = np.array(g.ndata['label'])

        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)

    elif dataset_str == 'flickr':
        from dgl.data import FlickrDataset
        data = FlickrDataset()
        g = data[0]
        num_classes = data.num_classes

        adj = torch_sparse.tensor.SparseTensor.from_edge_index(g.edges())

        features = g.ndata['feat']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']

        idx_train = (train_mask == True).nonzero().view(-1)
        idx_val = (val_mask == True).nonzero().view(-1)
        idx_test = (test_mask == True).nonzero().view(-1)

        labels = np.array(g.ndata['label'])

        scaler = StandardScaler()
        scaler.fit(features)
        features = scaler.transform(features)


    return features, adj, labels, idx_train, idx_val, idx_test






