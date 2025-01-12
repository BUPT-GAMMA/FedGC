import copy

import torch
import torch.nn as nn
from util import *
from random import sample
import sys
# sys.path.append('./models')
from models.gcn import GCN
import deeprobust.graph.utils as utils
from torch_sparse import SparseTensor
from warnings import simplefilter
from torch_geometric.loader import NeighborSampler
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
simplefilter(action='ignore', category=FutureWarning)



class Client_general(nn.Module):
    def __init__(self, client_id, origin_node_idx, idx_train, idx_val, idx_test, adj, labels, features, class_num, args):
        super().__init__()
        self.args = args
        self.client_id = client_id
        self.origin_node_idx = np.array(origin_node_idx)
        self.idx_train = np.array(idx_train)
        self.idx_test = np.array(idx_test)
        self.idx_val = np.array(idx_val)
        self.features = np.array(features)
        self.features_train = self.features[self.idx_train]
        self.features_val = self.features[self.idx_val]
        self.features_test = self.features[self.idx_test]
        self.labels = np.array(labels)
        self.labels_train = self.labels[self.idx_train]
        self.labels_val = self.labels[self.idx_val]
        self.labels_test = self.labels[self.idx_test]
        self.adj = sparse_tensor_to_csr(adj)#sparseTensor
        self.adj_train = self.adj[np.ix_(idx_train, idx_train)]
        self.adj_val = self.adj[np.ix_(idx_val, idx_val)]
        self.adj_test = self.adj[np.ix_(idx_test, idx_test)]
        self.class_num = class_num
        #self.class_dict = None
        self.class_dict2 = None
        self.samplers = None
        self.best_test_acc = 0
        self.optimizer = None
        self.gib_model = None



    def update(self, gnn_model):
        self.gnn_model = copy.deepcopy(gnn_model)

    def update_gib_model(self, gib_model):
        if(self.gib_model==None):#first init all param
            self.gib_model = copy.deepcopy(gib_model)
        else:
            param = list((_.detach().clone() for _ in list(gib_model.parameters())[0:4]))
            ls_model_param_user = list(self.gib_model.parameters())[0:4]
            for i in range(len(ls_model_param_user)):
                ls_model_param_user[i].data = param[i].data

    def update_self_train(self, self_train_model):
        self.self_train_model = copy.deepcopy(self_train_model)


    def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None, no_self_train=True):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.class_num):
                if transductive:
                    if(no_self_train==False):
                        idx = self.relabel_idx_train[self.relabel_labels_train == i]
                    else:
                        idx = self.idx_train[self.labels_train == i]
                else:
                    if(no_self_train==False):
                        idx = np.arange(len(self.relabel_labels_train))[self.relabel_labels_train==i]
                    else:
                        idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx

        if args.nlayers == 1:
            sizes = [15]
        if args.nlayers == 2:
            sizes = [10, 5]
            # sizes = [-1, -1]
        if args.nlayers == 3:
            sizes = [15, 10, 5]
        if args.nlayers == 4:
            sizes = [15, 10, 5, 5]
        if args.nlayers == 5:
            sizes = [15, 10, 5, 5, 5]


        if self.samplers is None:
            self.samplers = {}
            for i in range(self.class_num):
                node_idx = torch.LongTensor(self.class_dict2[i])
                if(len(node_idx)!=0):
                    #print("yes1")
                    self.samplers[i] = NeighborSampler(adj,
                                                       node_idx=node_idx,
                                                       sizes=sizes, batch_size=num,
                                                       num_workers=12, return_e_id=False,
                                                       num_nodes=adj.size(0),
                                                       shuffle=True)
        if c not in self.samplers.keys():
            #print("yes")
            return 0, 0, 0
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[c].sample(batch)
        return out

    def knn_adj(self, z_x):
        from sklearn.metrics.pairwise import cosine_similarity
        # features[features!=0] = 1
        k = 2
        sims = cosine_similarity(z_x.cpu().numpy())
        sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
        for i in range(len(sims)):
            indices_argsort = np.argsort(sims[i])
            sims[i, indices_argsort[: -k]] = 0
        adj_knn = torch.FloatTensor(sims).to(self.device)
        return adj_knn

    def self_train(self, ratio=1):

        sample_node_set = np.concatenate((self.idx_val, self.idx_test), axis=0)#idx_val

        ratio = ratio
        label_size = int(len(sample_node_set)*ratio)
        sampled_idx = np.array(sample(list(sample_node_set), label_size))
        self.relabel_idx_train = np.concatenate((self.idx_train, sampled_idx), axis=0)


        self.self_train_model.eval()
        output = self.self_train_model.predict(self.features, self.adj)
        preds = np.array(output.max(1)[1].cpu())#.type_as(self.labels)
        labeled_sample = preds[sampled_idx]
        # labels_train = labels_train.to('cpu')
        self.relabel_labels_train = np.concatenate((self.labels_train, labeled_sample), axis=0)
        self.relabel_labels = self.labels.copy()
        self.relabel_labels[self.relabel_idx_train] = self.relabel_labels_train


    def train_self_train(self):

        features, adj, labels = utils.to_tensor(self.features, self.adj, self.labels, device=self.args.device)#utils.to_tensor(self.features, self.adj, self.labels, device=self.args.device)

        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_sparse_tensor(adj)
            #adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_sparse_tensor(adj)
        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                           value=adj._values(), sparse_sizes=adj.size()).t()

        loss =0

        BN_flag = False
        for module in self.self_train_model.modules():
            if 'BatchNorm' in module._get_name(): #BatchNorm
                BN_flag = True
        if BN_flag:
            self.self_train_model.train() # for updating the mu, sigma of BatchNorm
            output_real = self.self_train_model.forward(features, adj)
            for module in self.self_train_model.modules():
                if 'BatchNorm' in module._get_name():  #BatchNorm
                    module.eval() # fix mu and sigma of every BatchNorm layer

        model_parameters = list(self.self_train_model.parameters())
        output = self.self_train_model.forward(features, adj)
        loss_real = F.nll_loss(output[self.idx_train], labels[self.idx_train])

        gw_real = torch.autograd.grad(loss_real, model_parameters)
        gw_real = list((_.detach().clone() for _ in gw_real))

        loss+=loss_real.item()
        return gw_real, loss, self.client_id


    def train_gib_model_param(self):

        features, adj, labels = utils.to_tensor(self.features, self.adj, self.labels, device=self.args.device)

        if utils.is_sparse_tensor(adj):
            #print("yes")
            adj_norm = utils.normalize_sparse_tensor(adj)
            #adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_sparse_tensor(adj)
        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                           value=adj._values(), sparse_sizes=adj.size()).t()

        loss =0
        if(self.optimizer==None):
            self.optimizer = optim.Adam(list(self.gib_model.parameters())[4:8], lr=self.args.gib_lr, weight_decay=0)
        self.gib_model.train()
        self.optimizer.zero_grad()

        output, l2, z_x_temp = self.gib_model.forward(features, adj)
        if(self.args.no_self_train==True):
            loss_train = F.nll_loss(output[self.idx_train], labels[self.idx_train])
        else:
            relabel_labels_train = torch.LongTensor(self.relabel_labels_train).to(self.args.device)
            loss_train = F.nll_loss(output[self.relabel_idx_train], relabel_labels_train)
        loss_real = loss_train +self.args.gib_beta*l2

        loss+=loss_real.item()

        loss_real.backward()
        self.optimizer.step()
        param = list((_.detach().clone() for _ in list(self.gib_model.parameters())))


        return param, loss


    def train_(self, no_self_train=True):#

        if(no_self_train==False):
            features, adj, labels = utils.to_tensor(self.features, self.adj, self.relabel_labels, device=self.args.device)
        else:
            features, adj, labels = utils.to_tensor(self.features, self.adj, self.labels, device=self.args.device)

        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_sparse_tensor(adj)
            #adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_sparse_tensor(adj)
            #adj_norm = utils.normalize_adj_tensor(adj)
        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                           value=adj._values(), sparse_sizes=adj.size()).t()

        real_grad_per_class = {}
        loss =0

        BN_flag = False
        for module in self.gnn_model.modules():
            if 'BatchNorm' in module._get_name(): #BatchNorm
                BN_flag = True
        if BN_flag:
            self.gnn_model.train() # for updating the mu, sigma of BatchNorm
            output_real = self.gnn_model.forward(features, adj_norm)
            for module in self.gnn_model.modules():
                if 'BatchNorm' in module._get_name():  #BatchNorm
                    module.eval() # fix mu and sigma of every BatchNorm layer

        model_parameters = list(self.gnn_model.parameters())

        for c in range(self.class_num):
            batch_size, n_id, adjs = self.retrieve_class_sampler(
                c, adj, transductive=True, args=self.args, no_self_train=no_self_train)
            if(batch_size==0):
                continue
            if self.args.nlayers == 1:
                adjs = [adjs]

            adjs = [adj.to(self.args.device) for adj in adjs]
            output = self.gnn_model.forward_sampler(features[n_id], adjs)
            #loss_real = F.nll_loss(output, relabel_labels[n_id[:batch_size]])
            loss_real = F.nll_loss(output, labels[n_id[:batch_size]])

            gw_real = torch.autograd.grad(loss_real, model_parameters)
            gw_real = list((_.detach().clone() for _ in gw_real))

            real_grad_per_class[c] = gw_real
            loss+=loss_real

        return real_grad_per_class, self.client_id#loss


    def test(self, condensed_graph, verbose=True):
        (feat_syn, adj_syn, labels_syn) = condensed_graph
        #选择模型
        model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                    lr=0.005, weight_decay=5e-4, nlayers=2, #weight_decay:5e-4 lr:0.01
                    nclass=self.class_num, device=self.args.device).to(self.args.device)

        if self.args.dataset in ['ogbn-arxiv']:
            model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                        weight_decay=0e-4, nlayers=2, with_bn=False,
                        nclass=self.class_num, device=self.args.device).to(self.args.device)

        data = {'idx_train': self.idx_train, 'idx_val': self.idx_val, 'idx_test':self.idx_test,
                'feat_train':self.features_train, 'feat_val':self.features_val, 'feat_test':self.features_test,
                'labels_train':self.labels_train, 'labels_val':self.labels_val, 'labels_test':self.labels_test,
                'adj_train':self.adj_train, 'adj_val':self.adj_val, 'adj_test':self.adj_test,
                'feat_full':self.features, 'adj_full':self.adj}

        if(self.args.dataset=='reddit' or self.args.dataset=='ogbn-arxiv' or self.args.dataset=='flickr'):
            model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                               train_iters=600, normalize=True, noval = True, verbose=False)
        else:
            model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                               train_iters=600, normalize=True, verbose=False)

        if(self.args.no_finetune==False):
            model.fit_with_val(data['feat_full'], data['adj_full'], data['labels_train'], data, train_iters=30,
                               initialize=False, normalize=True, verbose=False, finetune=True)

        model.eval()
        labels_test = torch.LongTensor(self.labels_test).to(self.args.device)#cuda()
        labels_train = torch.LongTensor(self.labels_train).to(self.args.device)#cuda()
        res = []

        output = model.predict(self.features, self.adj)
        loss_train = F.nll_loss(output[self.idx_train], labels_train)
        acc_train = utils.accuracy(output[self.idx_train], labels_train)
        res.append(acc_train.item())

        loss_test = F.nll_loss(output[self.idx_test], labels_test)
        acc_test = utils.accuracy(output[self.idx_test], labels_test)
        res.append(acc_test.item())

        return res, self.client_id

    def test_gib_model(self, condensed_graph=None, verbose=True):

        self.gib_model.eval()
        labels_test = torch.LongTensor(self.labels_test).to(self.args.device)#cuda()
        labels_train = torch.LongTensor(self.labels_train).to(self.args.device)#cuda()
        res = []

        if type(self.adj) is not torch.Tensor:
            features, adj = utils.to_tensor(self.features, self.adj, device=self.args.device)

        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_sparse_tensor(adj)
            #adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_sparse_tensor(adj)
        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                           value=adj._values(), sparse_sizes=adj.size()).t()

        output, _, z_x_temp = self.gib_model.forward(features, adj)
        loss_train = F.nll_loss(output[self.idx_train], labels_train)
        acc_train = utils.accuracy(output[self.idx_train], labels_train)
        res.append(acc_train.item())

        loss_test = F.nll_loss(output[self.idx_test], labels_test)
        acc_test = utils.accuracy(output[self.idx_test], labels_test)
        res.append(acc_test.item())

        if(acc_test.item()>self.best_test_acc):
            self.best_test_acc = acc_test.item()
            self.current_z = z_x_temp.clone().detach()

        return res, self.client_id

    def test_self_train(self, verbose=True):

        self.self_train_model.eval()
        labels_test = torch.LongTensor(self.labels_test).to(self.args.device)
        labels_train = torch.LongTensor(self.labels_train).to(self.args.device)
        res = []

        output = self.self_train_model.predict(self.features, self.adj)
        loss_train = F.nll_loss(output[self.idx_train], labels_train)
        acc_train = utils.accuracy(output[self.idx_train], labels_train)
        res.append(acc_train.item())

        loss_test = F.nll_loss(output[self.idx_test], labels_test)
        acc_test = utils.accuracy(output[self.idx_test], labels_test)
        res.append(acc_test.item())

        return res, self.client_id


    def update_z(self):

        self.features_origin = self.features
        self.features_train_origin = self.features_train
        self.features_val_origin = self.features_val
        self.features_test_origin = self.features_test

        self.features = self.current_z

        scaler = StandardScaler()
        scaler.fit(self.features)
        self.features = np.array(scaler.transform(self.features))

        self.features_train = self.features[self.idx_train]
        self.features_val = self.features[self.idx_val]
        self.features_test = self.features[self.idx_test]
