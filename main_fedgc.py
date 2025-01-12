import logging
import numpy as np
import ray
import os
from time import time
from ourparse import *
from models.gcn import GCN
from models.sgc import SGC
from models.sgc_multi import SGC as SGC1
from models.gib_gcn import GIB_GCN
from data_process import *
from util import *
from client_fedgc import *
from server_fedgc import *
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print("use:", torch.device(args.device))


'''seed'''
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
'''seed end'''

'''log'''
file = os.path.basename(sys.argv[0])[0:-3]+"_"+str(time())
print_log(args.log_dir+file)
'log end'

logging.info(args)

features, adj, labels, idx_train, idx_val, idx_test = load_data(args.dataset)
print("train_len: ", len(idx_train))
print("val_len: ", len(idx_val))
print("test_len: ", len(idx_test))
print("all_node_len", len(features))
class_num = labels.max().item() + 1

in_feat = features.shape[1]
if args.dataset in ['simulate', 'cora', 'citeseer', 'pubmed', "reddit"]:
    args_hidden = 16
else:
    args_hidden = 256

row, col, edge_attr = adj.coo()
edge_index = torch.stack([row, col], dim=0)


total_train_labels = labels[idx_train]
#repeat experiments
all_res = []
all_attack_auroc_res = []
all_attack_f1_res = []
setup_seed(40)
print_test_list = []

for repeat in range(args.repeat_time):
    if args.device == 'cpu':
        @ray.remote(num_cpus=6, scheduling_strategy='SPREAD')
        class Client(Client_general):
            def __init__(self, client_id, idx_train, idx_val, idx_test, adj, labels, features, class_num, args):
                super().__init__(client_id, idx_train, idx_val, idx_test, adj, labels, features, class_num, args)

    elif args.dataset == "ogbn-arxiv":
        @ray.remote(num_gpus=0.3, num_cpus=3, scheduling_strategy='SPREAD')
        class Client(Client_general):
            def __init__(self, client_id, origin_node_index, idx_train, idx_val, idx_test, adj, labels, features, class_num, args):

                super().__init__(client_id, origin_node_index, idx_train, idx_val, idx_test, adj, labels, features, class_num, args)

    else:
        @ray.remote(num_gpus=0.1, num_cpus=1, scheduling_strategy='SPREAD')
        class Client(Client_general):
            def __init__(self, client_id, origin_node_index, idx_train, idx_val, idx_test, adj, labels, features, class_num, args):

                super().__init__(client_id, origin_node_index, idx_train, idx_val, idx_test, adj, labels, features, class_num, args)
    #beta = 0.0001 extremly Non-IID, beta = 10000, IID
    #print(len(labels))
    split_data_indexes = label_dirichlet_partition(labels, len(labels), class_num, args.n_trainer, beta = args.iid_beta)


    client_list = []
    train_data_weights = []
    test_data_weights = []
    client_train_labels = []

    for i in range(args.n_trainer):
        split_data_indexes[i] = np.array(split_data_indexes[i])
        split_data_indexes[i].sort()
        split_data_indexes[i] = torch.tensor(split_data_indexes[i])

        if(args.no_comm==True):
            communicate_index, current_edge_index, _, __ = torch_geometric.utils.k_hop_subgraph(split_data_indexes[i],0,edge_index, relabel_nodes=True)
        else:
            communicate_index = split_data_indexes[i]
            L_hop=args.l_hop
            for hop in range(L_hop):
                if hop != L_hop-1:
                    communicate_index = torch_geometric.utils.k_hop_subgraph(communicate_index,1,edge_index, relabel_nodes=True)[0]
                else:
                    communicate_index, current_edge_index, _, __ = torch_geometric.utils.k_hop_subgraph(communicate_index,1,edge_index, relabel_nodes=True)
                    del _
                    del __

        communicate_index = communicate_index.to('cpu')

        origin_node_index = torch.searchsorted(communicate_index, split_data_indexes[i]).clone()

        #edge_set
        current_edge_index = current_edge_index.to('cpu')#
        current_edge_index = torch_sparse.SparseTensor(row=current_edge_index[0], col=current_edge_index[1], sparse_sizes=(len(communicate_index), len(communicate_index)))

        #train_node
        inter = intersect1d(split_data_indexes[i], idx_train)
        current_train_node_index = torch.searchsorted(communicate_index, inter).clone()
        #print(current_train_node_index)

        #valid_node
        inter = intersect1d(split_data_indexes[i], idx_val)
        current_val_node_index = torch.searchsorted(communicate_index, inter).clone()

        #test_node
        inter = intersect1d(split_data_indexes[i], idx_test)
        current_test_node_index = torch.searchsorted(communicate_index, inter).clone()

        #feature
        current_features = features[communicate_index]

        #labels
        current_labels = labels[communicate_index]
        #print(current_labels)


        #test end
        client = Client.remote(i, origin_node_index, current_train_node_index, current_val_node_index, current_test_node_index, current_edge_index, current_labels, current_features, class_num, args)
        #print(len(current_train_node_index))

        client_list.append(client)
        train_data_weights.append(len(current_train_node_index))
        test_data_weights.append(len(current_test_node_index))

        if(len(current_train_node_index)==1):
            client_train_labels.append([current_labels[current_train_node_index]])
        else:
            client_train_labels.append(current_labels[current_train_node_index])
    #print(len(total_train_labels))


    #choose gnn model
    if args.dataset in ['ogbn-arxiv']:
        model = SGC1(nfeat=features.shape[1], nhid=args.hidden,
                     dropout=0.0, with_bn=False,
                     weight_decay=0e-4, nlayers=2,
                     nclass=class_num,
                     device=args.device).to(args.device)
    else:
        if args.sgc == 1:
            model = SGC(nfeat=features.shape[1], nhid=args.hidden,
                        nclass=class_num, dropout=args.dropout,
                        nlayers=args.nlayers, with_bn=False,
                        device=args.device).to(args.device)
        else:
            model = GCN(nfeat=features.shape[1], nhid=args.hidden,
                        nclass=class_num, dropout=args.dropout, nlayers=args.nlayers,
                        device=args.device).to(args.device)


    gib_model = GIB_GCN(nfeat=features.shape[1], nhid=args.hidden, dropout=0.5,
                        weight_decay=5e-4, nlayers=2,
                        nclass=class_num, device=args.device).to(args.device)

    '''self train'''
    self_train_model =  GCN(nfeat=features.shape[1], nhid=args.hidden,
                            nclass=class_num, dropout=args.dropout, nlayers=args.nlayers,
                            device=args.device).to(args.device)


    # init server
    all_data = (features, adj, labels, idx_train, idx_val, idx_test)
    server = Server(client_list, model, gib_model, self_train_model, total_train_labels, features.shape[1], class_num, train_data_weights, test_data_weights, client_train_labels, args, all_data = None).to(torch.device(args.device))

    '''begin training'''
    outer_loop, inner_loop = get_loops(args)

    eval_epochs = [100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2400, 2800, 3000, 3500, 4000, 4500, 5000, 5500, 6000]


    if(args.no_self_train==False):
        if(args.no_comm == True):#
            self_train_model_path = './pretrain_models/'+args.dataset+'_nocomm_'+str(args.iid_beta)+'.pth'
        else:
            self_train_model_path = './pretrain_models/'+args.dataset+'_'+str(args.iid_beta)+'.pth'

        # self_train_model_path = './pretrain_models/'+args.dataset+'_'+str(args.iid_beta)+'.pth'
        if(os.path.exists(self_train_model_path)):
            print("load self-train model...")
            self_train_model.load_state_dict(torch.load(self_train_model_path))
            server.self_train_model = copy.deepcopy(self_train_model)
            server.distribute_self_train(client_list)
        else:
            print("begin self-train")
            best=0
            server.init_self_train_model_param()
            server.distribute_self_train(client_list)
            for it in range(args.epochs+1):

                param_list = []
                loss_list = []
                param_dict = {}
                for pd in range(args.n_trainer):
                    param_dict[pd] = None
                params = [c.train_self_train.remote() for c in client_list]
                while True:
                    ready, left = ray.wait(params, num_returns=1, timeout=None)
                    if ready:
                        temp= ray.get(ready)
                        param_dict[temp[0][2]] = temp[0][0]
                        loss_list.append(temp[0][1])
                    params = left
                    if not params:
                        break
                param_list= param_dict.values()

                server.aggregate_self_train(param_list) #!聚合参数
                #print(server.model_real_gradient_per_class[0][0])
                server.distribute_self_train(client_list)
                loss = np.mean(np.array(loss_list)).item()
                print('Epoch {}, loss_avg: {}'.format(it, loss))

                '''begin test'''
                if it in eval_epochs:
                    # if verbose and (it+1) % 50 == 0:
                    res_list = []
                    runs = 1
                    for i in range(runs):
                        res = server.test_self_train()
                        res_list.append(res)

                    res = np.array(res_list)
                    logging.info('Train/Test mean accuracy:{}, std:{}'.format(res.mean(0), res.std(0)))
                    if(res.mean(0)[1]>best):
                        best = res.mean(0)[1]
                        best_self_train_model = copy.deepcopy(server.self_train_model)
            print(best)
            server.self_train_model = copy.deepcopy(best_self_train_model)
            server.distribute_self_train(client_list)
            model_state_dict = best_self_train_model.state_dict()
            torch.save(model_state_dict, self_train_model_path)

        res_list = []
        res = server.test_self_train()
        res_list.append(res)
        res = np.array(res_list)
        logging.info('self train best accuracy:{}, std:{}'.format(res.mean(0), res.std(0)))

        '''self train'''
        client_train_labels = []
        all_train_labels = []
        for c in client_list:
            c.self_train.remote(ratio = args.self_train_ratio)


    if(args.no_gib==False):
        print("begin local graph transformation with IB...")
        best = 0
        server.init_gib_param()
        server.distribute_gib_model(client_list)
        for it in range(args.gib_epochs):
            param_list=[]
            avg_loss = []

            params = [c.train_gib_model_param.remote() for c in client_list]
            while True:
                ready, left = ray.wait(params, num_returns=1, timeout=None)
                if ready:
                    temp= ray.get(ready)
                    avg_loss.append(temp[0][1])
                params = left
                if not params:
                    break

            loss = np.mean(np.array(avg_loss)).item()
            logging.info('Epoch {}, loss_avg: {}'.format(it, loss))

            '''begin test'''
            # if verbose and (it+1) % 50 == 0:
            res_list = []
            runs = 1 if args.dataset in ['ogbn-arxiv'] else 1
            for i in range(runs):
                res = server.test_gib_model()
                res_list.append(res)

            res = np.array(res_list)
            logging.info('Train/Test mean accuracy:{}, std:{}'.format(res.mean(0), res.std(0)))
            if(res.mean(0)[1]>best):
                best = res.mean(0)[1]

        for c in client_list:
            c.update_z.remote()


    best=0
    best_attack=0
    best_attack_auroc =0
    best_attack_f1=0
    print("begin federated training..")
    for it in range(args.epochs+1):
        server.init_gnn_param()
        server.distribute(client_list)

        match_loss_list = []

        for ol in range(outer_loop):
            param_dict = {}
            param_list = []
            for pd in range(args.n_trainer):
                param_dict[pd] = None#初始化
            params = [c.train_.remote() for c in client_list]
            while True:
                ready, left = ray.wait(params, num_returns=1, timeout=None)
                if ready:
                    temp= ray.get(ready)
                    param_dict[temp[0][1]] = temp[0][0]
                params = left
                if not params:
                    break
            param_list= param_dict.values()

            server.aggregate(param_list) #!聚合参数
            loss = server.train_(it, ol)
            match_loss_list.append(loss)

        match_loss_ = np.mean(np.array(match_loss_list)).item()
        logging.info('Epoch {}, match_loss_avg: {}'.format(it, match_loss_))

        '''begin test'''
        if it in eval_epochs:
            # if verbose and (it+1) % 50 == 0:
            res_list = []
            runs = 1 if args.dataset in ['ogbn-arxiv'] else 3 #测试几次
            for i in range(runs):
                res = server.test()
                res_list.append(res)

            res = np.array(res_list)
            logging.info('Train/Test mean accuracy:{}, std:{}'.format(res.mean(0), res.std(0)))
            if(res.mean(0)[1]>best):
                best = res.mean(0)[1]

            print_test_list.append([res.mean(0)[1]])


    all_res.append(best)
    ray.shutdown()
logging.info(np.mean(all_res))
logging.info(np.std(all_res))

#ray.shutdown()











