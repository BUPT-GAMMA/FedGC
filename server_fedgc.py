import ray
import torch
import torch.nn as nn
from models.parametrized_adj import PGE
import deeprobust.graph.utils as utils
from util import *
from collections import Counter



class Server(nn.Module):
    def __init__(self, client_list, gnn_model, gib_model, self_train_model, total_train_labels, fea_dim, class_num, train_data_weights, test_data_weights, client_train_labels, args, **kwargs):
        super().__init__()
        self.args = args
        self.client_list = client_list
        self.gnn_model = gnn_model
        self.gib_model = gib_model
        self.self_train_model = self_train_model

        self.total_train_labels = total_train_labels
        self.class_num = class_num
        self.lr = args.lr

        self.outer_loop, self.inner_loop = get_loops(args)
        # self.features = features
        self.weight_decay = args.weight_decay

        self.labels_syn = torch.LongTensor(self.generate_labels_syn(total_train_labels)).to(args.device)

        self.fea_dim = fea_dim
        self.nnodes_syn = len(self.labels_syn)
        self.train_data_weights = train_data_weights
        self.test_data_weights = test_data_weights
        self.client_train_labels = client_train_labels


        counter = Counter(self.total_train_labels)
        #print(counter)
        self.client_train_labels_ratio = []
        for ct in self.client_train_labels:
            c = Counter(ct)
            ctlr = dict()
            for key in c.keys():
                ctlr[key] = c[key]*1.0 / counter[key]
            self.client_train_labels_ratio.append(ctlr)

        feat_ = np.random.normal(size = (self.nnodes_syn, fea_dim))

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(feat_)
        feat_ = scaler.transform(feat_)

        self.feat_syn = nn.Parameter(torch.FloatTensor(feat_).to(args.device))#14
        nn.init.normal_(self.feat_syn, std=0.01)

        self.pge = PGE(nfeat=self.fea_dim, nnodes=self.nnodes_syn, device=args.device,args=args).to(args.device)

        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        print('adj_syn:', (self.nnodes_syn,self.nnodes_syn), 'feat_syn:', self.feat_syn.shape)

        if 'all_data' in kwargs:
            if(kwargs['all_data']!=None):
                self.class_dict = None
                self.all_feature, self.all_adj, self.all_labels, self.all_idx_train, self.all_idx_val, self.all_idx_test = kwargs['all_data']
                if sp.issparse(self.all_feature):
                    self.all_feature = sparse_mx_to_torch_sparse_tensor(self.all_feature)
                features = self.all_feature.to(self.args.device)
                feat_sub, adj_sub = self.get_sub_adj_feat(features)
                self.feat_syn.data.copy_(feat_sub)

    def update_label_ratio(self, all_train_labels, client_train_labels):
        counter = Counter(all_train_labels)
        self.client_train_labels_ratio = []
        for ct in client_train_labels:
            c = Counter(ct)
            ctlr = dict()
            for key in c.keys():
                ctlr[key] = c[key]*1.0 / counter[key]
            self.client_train_labels_ratio.append(ctlr)

    def init_gnn_param(self):
        self.gnn_model.initialize()
        #print(list(self.gnn_model.parameters())[0])

    def init_gib_param(self):
        self.gib_model.initialize()

    def init_gib_param_layer(self):
        self.gib_model.initialize_layer()

    def init_self_train_model_param(self):
        self.self_train_model.initialize()

    def generate_labels_syn(self, labels_train):

        counter = Counter(labels_train)
        num_class_dict = {}
        n = len(labels_train)#29

        sorted_counter = sorted(counter.items(), key=lambda x:x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn

    def aggregate_self_train(self, param_list):
        flag=False
        number = len(param_list)
        for parameter in param_list:
            model_grad = parameter
            if not flag:
                flag = True
                gradient_model = []
                for i in range(len(model_grad)):
                    gradient_model.append(model_grad[i])
            else:
                for i in range(len(model_grad)):
                    gradient_model[i] += model_grad[i]
        for i in range(len(gradient_model)):
            gradient_model[i] = gradient_model[i] / number
        ls_model_param = list(self.self_train_model.parameters())
        for i in range(len(ls_model_param)):
            ls_model_param[i].data = ls_model_param[i].data - self.lr * gradient_model[i] - self.weight_decay * ls_model_param[i].data


    def aggregate(self, param_list):
        if(self.args.no_pclass_agg==True):
            model_real_gradient_per_class = {}
            for c in range(self.class_num):
                c_num = 0
                for parameter in param_list:
                    if(c in parameter.keys()):
                        c_num+=1
                        if(c not in model_real_gradient_per_class.keys()):
                            model_real_gradient_per_class[c] = parameter[c]
                        else:
                            for i,p in enumerate(parameter[c]):
                                model_real_gradient_per_class[c][i]+=p
                #model_real_gradient_per_class[c]/=c_num #fedavg
                model_real_gradient_per_class[c]  = [p/c_num for p in model_real_gradient_per_class[c]]
            self.model_real_gradient_per_class = model_real_gradient_per_class


        else:
            model_real_gradient_per_class = {}
            for c in range(self.class_num):#每一类
                c_num = 0
                for index, parameter in enumerate(param_list):#每一个client
                    if(c in parameter.keys()):
                        c_num+=1
                        if(c not in model_real_gradient_per_class.keys()):
                            model_real_gradient_per_class[c] = parameter[c]
                            for i,p in enumerate(parameter[c]):
                                model_real_gradient_per_class[c][i]=p*self.client_train_labels_ratio[index][c]
                            #model_real_gradient_per_class[c] = parameter[c]
                        else:
                            for i,p in enumerate(parameter[c]):
                                model_real_gradient_per_class[c][i]+=p*self.client_train_labels_ratio[index][c]
                #model_real_gradient_per_class[c]/=c_num #fedavg
                #model_real_gradient_per_class[c]  = [p/c_num for p in model_real_gradient_per_class[c]]
            self.model_real_gradient_per_class = model_real_gradient_per_class


    def distribute(self, client_list):
        for client in client_list:
            client.update.remote(self.gnn_model)

    def distribute_gib_model(self, client_list):
        #print(list(self.gib_model.parameters())[0])
        for client in client_list:
            client.update_gib_model.remote(self.gib_model)

    def distribute_self_train(self, client_list):
        for client in client_list:
            client.update_self_train.remote(self.self_train_model)

    def retrieve_class(self, c, num = 256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.class_num):
                self.class_dict['class_%s'%i] = (self.all_labels[self.all_idx_train] == i)
        idx = np.arange(len(self.all_labels[self.all_idx_train]))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]#num是合成的标签中类别为c的数量，这个函数目的是为合成的类随机找一些原始数据中的样本（train）

    def get_sub_adj_feat(self, features):
        #args = self.args
        idx_selected = []

        from collections import Counter
        counter = Counter(self.labels_syn.cpu().numpy())

        for c in range(self.class_num):
            tmp = self.retrieve_class(c, num=counter[c])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[self.all_idx_train][idx_selected]

        from sklearn.metrics.pairwise import cosine_similarity
        # features[features!=0] = 1
        k = 2
        sims = cosine_similarity(features.cpu().numpy())
        sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
        for i in range(len(sims)):
            indices_argsort = np.argsort(sims[i])
            sims[i, indices_argsort[: -k]] = 0
        adj_knn = torch.FloatTensor(sims).to(self.args.device)
        return features, adj_knn

    def train_(self, it, ol):
        args = self.args
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn
        syn_class_indices = self.syn_class_indices

        model_parameters = list(self.gnn_model.parameters())
        optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
        self.gnn_model.train()

        adj_syn = pge(self.feat_syn)
        adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
        feat_syn_norm = feat_syn

        BN_flag = False
        for module in self.gnn_model.modules():
            if 'BatchNorm' in module._get_name(): #BatchNorm
                BN_flag = True
        if BN_flag:
            self.gnn_model.train() # for updating the mu, sigma of BatchNorm
            #output_real = self.gnn_model.forward(features, adj_norm)
            for module in self.gnn_model.modules():
                if 'BatchNorm' in module._get_name():  #BatchNorm
                    module.eval() # fix mu and sigma of every BatchNorm layer
        #
        loss = torch.tensor(0.0).to(self.args.device)
        loss_avg = 0
        for c in range(self.class_num):

            gw_real = self.model_real_gradient_per_class[c]

            output_syn = self.gnn_model.forward(feat_syn, adj_syn_norm)

            ind = syn_class_indices[c]
            loss_syn = F.nll_loss(
                output_syn[ind[0]: ind[1]],
                labels_syn[ind[0]: ind[1]])
            gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
            coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
            loss += coeff  * match_loss(gw_syn, gw_real, args, device=args.device)

        loss_avg += loss.item()
        loss_avg /= self.class_num

        # TODO: regularize
        if args.alpha > 0:
            loss_reg = args.alpha * regularization(adj_syn, utils.tensor2onehot(labels_syn))
        else:
            loss_reg = torch.tensor(0)

        loss = loss + loss_reg

        # update sythetic graph
        self.optimizer_feat.zero_grad()
        self.optimizer_pge.zero_grad()
        loss.backward()
        if it % 50 < 10:
            self.optimizer_pge.step()
        else:
            self.optimizer_feat.step()

        if args.debug and ol % 5 ==0:
            print('Gradient matching loss:', loss.item())

        if ol == self.outer_loop - 1:
            # print('loss_reg:', loss_reg.item())
            # print('Gradient matching loss:', loss.item())
            return loss_avg

        feat_syn_inner = feat_syn.detach()
        adj_syn_inner = pge.inference(feat_syn_inner)
        adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
        feat_syn_inner_norm = feat_syn_inner
        for j in range(self.inner_loop):
            optimizer_model.zero_grad()
            output_syn_inner = self.gnn.model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
            loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
            loss_syn_inner.backward()
            # print(loss_syn_inner.item())
            optimizer_model.step() # update gnn param

        return loss_avg

    def get_syn_graph(self):
        feat_syn, pge, labels_syn = self.feat_syn.detach(), \
                                    self.pge, self.labels_syn

        adj_syn = pge.inference(feat_syn)
        args = self.args

        if self.args.save:
            torch.save(adj_syn, f'saved_ours/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
            torch.save(feat_syn, f'saved_ours/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')

        if self.args.lr_adj == 0:
            n = len(labels_syn)
            adj_syn = torch.zeros((n, n))

        condensed_graph = (feat_syn, adj_syn, labels_syn)
        return condensed_graph



    def test(self):
        condensed_graph = self.get_syn_graph()

        #save
        feat_syn, pge, labels_syn = condensed_graph
        save_c = [feat_syn, pge, labels_syn]
        import pickle
        with open('condensed_graph.pkl', 'wb') as f:
            pickle.dump(save_c, f)

        res_dict = {}
        for pd in range(self.args.n_trainer):
            res_dict[pd] = None

        reses = [client.test.remote(condensed_graph) for client in self.client_list]

        while True:
            ready, left = ray.wait(reses, num_returns=1, timeout=None)
            if ready:
                temp = ray.get(ready)
                res_dict[temp[0][1]] = temp[0][0]
            reses = left
            if not reses:
                break
        res_list = list(res_dict.values())

        res = np.array(res_list)
        avg_train_acc = np.average(res[:,0], weights = self.train_data_weights, axis = 0)#weighted mean
        avg_test_acc = np.average(res[:,1], weights = self.test_data_weights, axis = 0)
        #print(res[:,0], res[:,1])
        res = np.array([avg_train_acc, avg_test_acc])
        #res.mean(0)#
        return res

    def test_gib_model(self):

        res_dict = {}
        res_list = []
        for pd in range(self.args.n_trainer):
            res_dict[pd] = None
        reses = [client.test_gib_model.remote() for client in self.client_list]#本地测试

        while True:
            ready, left = ray.wait(reses, num_returns=1, timeout=None)
            if ready:
                temp = ray.get(ready)
                res_dict[temp[0][1]] = temp[0][0]
            reses = left
            if not reses:
                break
        res_list = list(res_dict.values())


        res = np.array(res_list)
        avg_train_acc = np.average(res[:,0], weights = self.train_data_weights, axis = 0)#weighted mean
        avg_test_acc = np.average(res[:,1], weights = self.test_data_weights, axis = 0)
        #print(res[:,0], res[:,1])
        res = np.array([avg_train_acc, avg_test_acc])
        return res

    def test_self_train(self):
        res_list = []
        res_dict = {}
        for pd in range(self.args.n_trainer):
            res_dict[pd] = None#
        reses = [client.test_self_train.remote() for client in self.client_list]
        while True:
            ready, left = ray.wait(reses, num_returns=1, timeout=None)
            if ready:
                temp = ray.get(ready)
                res_dict[temp[0][1]] = temp[0][0]
            reses = left
            if not reses:
                break
        res_list = list(res_dict.values())

        res = np.array(res_list)
        avg_train_acc = np.average(res[:,0], weights = self.train_data_weights, axis = 0)#weighted mean
        avg_test_acc = np.average(res[:,1], weights = self.test_data_weights, axis = 0)
        #print(res[:,0], res[:,1])
        res = np.array([avg_train_acc, avg_test_acc])
        return res











