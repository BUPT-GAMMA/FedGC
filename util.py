import torch
import numpy as np
from torch.nn import functional as F
from torch.distributions.normal import Normal
import scipy.sparse as sp
def intersect1d(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    '''
    This function concatenates the two input tensors, finding common elements between these two

    Argument:
    t1: (PyTorch tensor) - The first input tensor for the operation
    t2: (PyTorch tensor) - The second input tensor for the operation

    Return:
    intersection: (PyTorch tensor) - Intersection of the two input tensors
    '''
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]
    return intersection

def label_dirichlet_partition(labels: np.array, N: int, K: int, n_parties: int, beta: float) -> list:
    '''
    This function partitions data based on labels by using the Dirichlet distribution, to ensure even distribution of samples

    Arguments:
    labels: (NumPy array) - An array with labels or categories for each data point
    N: (int) - Total number of data points in the dataset
    K: (int) - Total number of unique labels
    n_parties: (int) - The number of groups into which the data should be partitioned
    beta: (float) - Dirichlet distribution parameter value

    Return:
    split_data_indexes (list) - list indices of data points assigned into groups

    '''
    min_size = 0
    min_require_size = 10

    split_data_indexes = []

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))

            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])

            proportions = proportions / proportions.sum()

            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        split_data_indexes.append(idx_batch[j])
    return split_data_indexes

def reparameterize_diagonal(model, input, mode):
    if model is not None:
        mean_logit = model(input)
    else:
        mean_logit = input
    if mode.startswith("diagg"):
        if isinstance(mean_logit, tuple):
            mean = mean_logit[0]
        else:
            mean = mean_logit
        std = torch.ones(mean.shape).to(mean.device)
        dist = Normal(mean, std)
        return dist, (mean, std)
    elif mode.startswith("diag"):#this
        if isinstance(mean_logit, tuple):
            mean_logit = mean_logit[0]
        size = int(mean_logit.size(-1) / 2)
        mean = mean_logit[:, :size]
        std = F.softplus(mean_logit[:, size:], beta=1) + 1e-10
        dist = Normal(mean, std)
        return dist, (mean, std)
    else:
        raise Exception("mode {} is not valid!".format(mode))


def get_loops(args):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if args.one_step:
        if args.dataset =='ogbn-arxiv':
            return 5, 0
        return 1, 0
    if args.dataset in ['ogbn-arxiv']:
        return args.outer, args.inner
    if args.dataset in ['cora']:
        return 20, 15 # sgc#20，15
    if args.dataset in ['citeseer']:
        return 20, 15
    if args.dataset in ['physics']:
        return 20, 10
    else:
        return 20, 10

def feature_smoothing(adj, X):
    adj = (adj.t() + adj)/2
    rowsum = adj.sum(1)
    r_inv = rowsum.flatten()
    D = torch.diag(r_inv)
    L = D - adj

    r_inv = r_inv  + 1e-8
    r_inv = r_inv.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    # L = r_mat_inv @ L
    L = r_mat_inv @ L @ r_mat_inv

    XLXT = torch.matmul(torch.matmul(X.t(), L), X)
    loss_smooth_feat = torch.trace(XLXT)
    # loss_smooth_feat = loss_smooth_feat / (adj.shape[0]**2)
    return loss_smooth_feat

def regularization(adj, x, eig_real=None):
    # fLf
    loss = 0
    # loss += torch.norm(adj, p=1)
    loss += feature_smoothing(adj, x)
    return loss

def distance_wb(gwr, gws):
    shape = gwr.shape

    # TODO: output node!!!!
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T

    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis

def match_loss(gw_syn, gw_real, args, device):
    dis = torch.tensor(0.0).to(device)

    if args.dis_metric == 'ours':

        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    sparserow=torch.LongTensor(sparse_mx.row).unsqueeze(1)
    sparsecol=torch.LongTensor(sparse_mx.col).unsqueeze(1)
    sparseconcat=torch.cat((sparserow, sparsecol),1)
    sparsedata=torch.FloatTensor(sparse_mx.data)
    return torch.sparse.FloatTensor(sparseconcat.t(),sparsedata,torch.Size(sparse_mx.shape))

def sparse_tensor_to_csr(graph):
    graph = graph.to_torch_sparse_coo_tensor()
    row = graph._indices()[0]
    col = graph._indices()[1]
    data = graph._values()
    shape = graph.size()
    adj = sp.csr_matrix((data, (row, col)), shape = shape)
    return adj

def sample(dist, n=None):
    """Sample n instances from distribution dist"""
    if n is None:
        return dist.rsample()#
    else:
        return dist.rsample((n,))

def row_normalize_tensor(mx):
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1).flatten()
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    return mx

def print_log(file):
    import logging
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(file, mode='w'),
        ]
    )
    # 输出日志信息
    logging.info('information will display in terminal and file')
