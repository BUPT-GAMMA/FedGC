import argparse
import torch.cuda as cuda


def parse_args():
    parser = argparse.ArgumentParser(description="Run Model.")
    parser.add_argument('--hidden', type=int, default=256, help='Dim of hidden vectors.')
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--path', nargs='?', default='Data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='cora', help='Choose a dataset.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device to run the model.')
    parser.add_argument('--no_pclass_agg', type=bool, default=True, help='whether pclass aggregation.')
    parser.add_argument('--no_finetune', type=bool, default=False, help='whether using local datasets to finetune.')
    parser.add_argument('--no_comm', type=bool, default=False, help='whether using communication in fedgcn')
    parser.add_argument('--no_gib', type=bool, default=False, help='whether using GIB')
    parser.add_argument('--no_self_train', type=bool, default=False, help='whether using self train')
    parser.add_argument('--no_conden_self_train', type=bool, default=True, help='whether using self train in condensation')
    parser.add_argument('--no_mia', type=bool, default=True, help='whether conducting mia')
    parser.add_argument('--dis_metric', type=str, default='mse')
    parser.add_argument('--repeat_time', type=int, default=5, help='Repeat_time.')
    parser.add_argument('--n_trainer', type=int, default=10, help='Number of clients.')
    parser.add_argument('-iid_b', '--iid_beta', default=1, type=float)
    parser.add_argument('--reduction_rate', '--r', type=float, default=0.5)
    parser.add_argument('--gib_lr', type=float, default=0.002)
    parser.add_argument('--gib_beta', type=float, default=0.0005)
    parser.add_argument('--self_train_ratio', type=float, default=1)
    parser.add_argument('--conden_self_train_ratio', type=float, default=1)
    parser.add_argument('--l_hop', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--lr_adj', type=float, default=1e-3)
    parser.add_argument('--lr_feat', type=float, default=1e-3)
    parser.add_argument('--lr_model', type=float, default=0.01)
    parser.add_argument('--num_sample', type=int, default=0, help='number of sampled neighbors.')
    parser.add_argument('--nonlinearity', type=str, default="relu", help='Which device to run the model.')
    parser.add_argument('--log_dir', type=str, default="./log/", help='Which device to run the model.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='lr weight_decay in optimizer.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs.')
    parser.add_argument('--gib_epochs', type=int, default=100, help='Number of gib epochs.')
    parser.add_argument('--sgc', type=int, default=0)
    parser.add_argument('--other_gnn', nargs='?', default='MLP', help='Choose a gnn.')
    parser.add_argument('--one_step', type=int, default=1)
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--alpha', type=float, default=0, help='regularization term.')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--clip', type=float, default=0.1)
    parser.add_argument('--laplace_lambda', type=float, default=0.2)
    return parser.parse_args()


args = parse_args()