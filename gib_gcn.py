''''*** copy from gcn.py****'''
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from sklearn.metrics import f1_score
from torch.nn import init
import torch_sparse
import numpy as np
from torch.distributions.normal import Normal
import sys
sys.path.append('../util')
from util import *


class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        if isinstance(adj, torch_sparse.SparseTensor):
            output = torch_sparse.matmul(adj, support)
        else:
            output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GIB_GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, lr=0.01, weight_decay=5e-4,
                 with_relu=True, with_bias=True, with_bn=False, device=None, beta=0.003):

        super(GIB_GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nclass = nclass

        self.layers = nn.ModuleList([])
        #
        if nlayers == 1:
            self.layers.append(GraphConvolution(nfeat, nclass, with_bias=with_bias))
        else:
            if with_bn:
                self.bns = torch.nn.ModuleList()
                self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nfeat, nhid, with_bias=with_bias))
            for i in range(nlayers-2):
                self.layers.append(GraphConvolution(nhid, nhid, with_bias=with_bias))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.layers.append(GraphConvolution(nhid, nclass, with_bias=with_bias))

        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bn = with_bn
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.multi_label = None
        self.z_w =nn.Sequential(
            nn.Linear(nfeat, nfeat),
            nn.ReLU(),
            nn.Linear(nfeat, nfeat*2),
        )
        self.beta = beta


    def forward(self, x, adj):
        l2 = z_x_temp = None
        features = self.z_w(x)

        if isinstance(adj, torch_sparse.SparseTensor):
            z_x_hat = torch_sparse.matmul(adj, features)#(n,d*2)
        else:
            z_x_hat = torch.spmm(adj, features)

        self.dist, _ = reparameterize_diagonal(model=None, input=z_x_hat, mode='diag')
        z_x = sample(self.dist, 1)


        self.q_z_x = Normal(loc = torch.zeros(x.size(0), x.size(1)).to(self.device),
                            scale = torch.ones(x.size(0), x.size(1)).to(self.device)
                            )

        Z_logit = self.dist.log_prob(z_x).sum(-1)  # [1, n]
        prior_logit = self.q_z_x.log_prob(z_x).sum(-1)  # [1, n]
        # upper bound of L2 = I(z_x;x,N):
        ixz = (Z_logit - prior_logit).mean(0)  # [n]
        l2 = ixz.mean()# （1） loss2


        #z_x = F.dropout(z_x, self.dropout, training=self.training)
        z_x_out = z_x.mean(0)#(n,d)
        z_x_temp = z_x_out.clone().detach().cpu()#(n,d)
        x = z_x_out
        #end gib

        for ix, layer in enumerate(self.layers):
            x = layer(x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)


        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1), l2, z_x_temp

    def forward_sampler(self, x, adjs):
        # for ix, layer in enumerate(self.layers):
        for ix, (adj, _, size) in enumerate(adjs):
            x = self.layers[ix](x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)

    def forward_sampler_syn(self, x, adjs):
        for ix, (adj) in enumerate(adjs):
            x = self.layers[ix](x, adj)
            if ix != len(self.layers) - 1:
                x = self.bns[ix](x) if self.with_bn else x
                if self.with_relu:
                    x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        if self.multi_label:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=1)


    def initialize(self, init_feature=None):
        """Initialize parameters with given feature.
        """
        if(init_feature==None):
            for layer in self.layers:
                layer.reset_parameters()
            if self.with_bn:
                for bn in self.bns:
                    bn.reset_parameters()
            #nn.init.normal_(self.z_w, std=0.01)
            for param in self.z_w:
                if(type(param) == nn.Linear):
                    #print("yes")
                    param.reset_parameters()
        else:
            for p, mp in zip(init_feature, self.parameters()):
                mp.data = p.clone().detach()
            # print(list(self.parameters())[0])

    def initialize_layer(self, init_feature=None):
        """Initialize parameters with given feature.
        """
        if(init_feature==None):
            for layer in self.layers:
                layer.reset_parameters()
            if self.with_bn:
                for bn in self.bns:
                    bn.reset_parameters()
        else:
            for p, mp in zip(init_feature, self.parameters()):
                mp.data = p.clone().detach()


    def fit_with_val(self, features, adj, labels, data, train_iters=200, initialize=True, verbose=False, normalize=True, patience=None, noval=False, **kwargs):
        '''data: full data class'''
        if type(initialize)==bool:
            self.initialize()
        else:
            self.initialize(init_feature=initialize)

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_sparse_tensor(adj)
                #adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_sparse_tensor(adj)
                #adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        if 'feat_norm' in kwargs and kwargs['feat_norm']:
            from ..util import row_normalize_tensor
            features = row_normalize_tensor(features-features.min())

        self.adj_norm = adj_norm
        self.features = features

        if len(labels.shape) > 1:
            self.multi_label = True
            self.loss = torch.nn.BCELoss()
        else:
            self.multi_label = False
            self.loss = F.nll_loss

        labels = labels.float() if self.multi_label else labels
        self.labels = labels

        if noval:#false
            self._train_with_val(labels, data, train_iters, verbose, adj_val=True)
        else:
            self._train_with_val(labels, data, train_iters, verbose)

    def _train_with_val(self, labels, data, train_iters, verbose, adj_val=False):
        if adj_val:
            feat_full, adj_full = data['feat_val'], data['adj_val']
        else:
            feat_full, adj_full = data['feat_full'], data['adj_full']
        feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=self.device)
        adj_full_norm = utils.normalize_sparse_tensor(adj_full)
        # adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
        labels_val = torch.LongTensor(data['labels_val']).to(self.device)
        labels_train = torch.LongTensor(data['labels_train']).to(self.device)

        if verbose: #false
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0

        for i in range(train_iters):
            # print(i)
            if i == train_iters // 2:
                lr = self.lr*0.1
                optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=self.weight_decay)

            self.train()
            optimizer.zero_grad()
            output, l2, z_x_out = self.forward(self.features, self.adj_norm)#all
            if(adj_val==False):
                loss_train = self.loss(output[data['idx_train']], labels_train)
            else:
                loss_train = self.loss(output, labels_train)
            #print("loss_train:{}, l2:{}".format(loss_train.item(),l2.item()))
            loss_train = loss_train +self.beta*l2
            loss_train.backward()
            optimizer.step()

            if verbose and i % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            with torch.no_grad():
                self.eval()

                output,_,_ = self.forward(feat_full, adj_full_norm)

                if adj_val:
                    loss_val = F.nll_loss(output, labels_val)
                    acc_val = utils.accuracy(output, labels_val)
                else:
                    loss_val = F.nll_loss(output[data['idx_val']], labels_val)
                    acc_val = utils.accuracy(output[data['idx_val']], labels_val)

                if acc_val > best_acc_val:
                    best_acc_val = acc_val
                    self.output = output
                    self.z_x_out = z_x_out
                    weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    @torch.no_grad()
    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency
        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)[0]
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_sparse_tensor(adj)
                #self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_sparse_tensor(adj)
                #self.adj_norm = utils.normalize_adj_tensor(adj)

            return self.forward(self.features, self.adj_norm)[0]

    @torch.no_grad()
    def predict_unnorm(self, features=None, adj=None):
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            self.adj_norm = adj
            return self.forward(self.features, self.adj_norm)