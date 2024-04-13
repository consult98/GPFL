import os
from xmlrpc.client import boolean
# required for pytorch deterministic GPU behaviour
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import pickle
import torch
from data_utils import *
from models import *
from optimisers import *
import argparse
from sys import argv
from fl_algs import *
# from torchsummary import summary
import torchvision.models as models
import random

def get_fname(a):
    """
    Args:
        - a: (argparse.Namespace) command-line arguments
        
    Returns:
        Underscore-separated str ending with '.pkl', containing items in args.
    """
    fname = '_'.join([  k+'-'+str(v) for (k, v) in vars(a).items() 
                        if not v is None])
    return fname + '.pkl'



def save_data(data, fname):
    """
    Saves data in pickle format.
    
    Args:
        - data:  (object)   to save 
        - fname: (str)      file path to save to 
    """
    with open(fname, 'wb') as f:
        pickle.dump(data, f)



def any_in_list(x, y):
    """
    Args:
        - x: (iterable) 
        - y: (iterable) 
    
    Returns:
        True if any items in x are in y.
    """
    return any(x_i in y for x_i in x)


def parse_args():
    """
    Details for the experiment to run are passed via the command line. Some 
    experiment settings require specific arguments to be passed (e.g. the 
    different FL algorithms require different hyperparameters). 
    
    Returns:
        argparse.Namespace of parsed arguments. 
    """
    parser = argparse.ArgumentParser()
    # 数据集
    parser.add_argument('-dset', required=True, choices=['mnist', 'cifar10','metr-la','pems-bay'], 
                        help='Federated dataset')
    # 优化器
    parser.add_argument('-alg', required=True, help='Federated optimiser',
                        choices=[   'fedavg', 'MTFL','PFedSA','pfedme','SFL','GPFL','GPFL-C','GPFL-G'])
    # 每一轮选择用户比例
    parser.add_argument('-C', required=True, type=float,
                        help='Fraction of clients selected per round')
    # batch size
    parser.add_argument('-B', required=True, type=int, help='Client batch size')
    # 全局更新轮数
    parser.add_argument('-T', required=True, type=int, help='Total rounds')
    # 一轮遍历E次数据
    parser.add_argument('-E', required=True, type=int, help='Client num epochs')
    # 运行设备
    parser.add_argument('-device', required=True, choices=['cpu', 'gpu'], 
                        help='Training occurs on this device')
    # 用户个数
    parser.add_argument('-W', required=True, type=int,
                        help='Total workers to split data across')

    # parser.add_argument('-seed', required=True, type=int, help='Random seed')

    # 学习率
    parser.add_argument('-lr', required=True, type=float,
                        help='Client learning rate')

    # specific arguments for different FL algorithms
    if any_in_list(['fedavg'], argv):
        # bn私有参数选择
        parser.add_argument('-bn_private', choices=['usyb', 'us', 'yb', 'none'],
                            required=True,
                            help='Patch parameters to keep private')
        
    if any_in_list(['MTFL'], argv):
        # bn私有参数选择
        parser.add_argument('-bn_private', choices=['usyb', 'us', 'yb', 'none'],
                            required=True,
                            help='Patch parameters to keep private')

    if any_in_list(['PFedSA'], argv):
        # bn私有参数选择
        parser.add_argument('-bn_private', choices=['usyb', 'us', 'yb', 'none'],
                            required=True,
                            help='Patch parameters to keep private')
        # 预先聚类的轮数
        parser.add_argument('-preT', required=True, type=float,
                        help='预先聚类的轮数')
        # 距离阈值
        parser.add_argument('-distance', required=True, type=float,
                        help='距离阈值')
                 

    if any_in_list(['pfedme'], argv):
        parser.add_argument('-beta', required=True, type=float,
                            help='PerFedAvg/pFedMe beta parameter')
        parser.add_argument('-lamda', required=True, type=float,
                            help='pFedMe lambda parameter')
    
    if any_in_list(['SFL'], argv):
        parser.add_argument('-layers', type=int, default=3, help='number of layers')

        parser.add_argument('-serveralpha', type=float, default=1, help='图聚合结果占比') 


    if any_in_list(['GPFL','GPFL-C','GPFL-G'], argv):
        parser.add_argument('-layers', type=int, default=3, help='number of layers')

        parser.add_argument('-serveralpha', type=float, default=1, help='图聚合结果占比') 

        parser.add_argument('-adjbeta', type=float, default=0.8, help='学习结构占比')  

        # bn私有参数选择
        parser.add_argument('-bn_private', choices=['usyb', 'us', 'yb', 'none'],
                            required=True,
                            help='Patch parameters to keep private')
                                           
        parser.add_argument('-preT', required=True, type=float,help='预先聚类的轮数')
        # 距离阈值
        parser.add_argument('-distance',type=float, default=0.8,help='距离阈值')
        
    args = parser.parse_args()

    return args


    
def main():
    """
    Run experiment specified by command-line args.
    """
    print('---------------start--------------------')
    args = parse_args()
    
    torch.use_deterministic_algorithms(True)
    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)
    device = torch.device('cuda:3' if args.device=='gpu' else 'cpu')

    
    # load data 
    print('Loading data...')
    if args.dset == 'mnist':
        train, test,pre_A,assignment_list = load_mnist(   './MNIST_data', args.W, iid='non-iid2',user_test=True)
        model = MNISTModel(device)
        feeders   = [   PyTorchDataFeeder(x, torch.float32, y, 'long', device)
                        for (x, y) in zip(train[0], train[1])]
        test_data = (   [to_tensor(x, device, torch.float32) for x in test[0]],
                        [to_tensor(y, device, 'long') for y in test[1]])  
        
    elif args.dset == 'cifar10':
        train, test,pre_A,assignment_list = load_cifar(   './CIFAR10_data', args.W,iid='non-iid2', user_test=True)
        model       = CIFAR10Model(device)
        feeders   = [   PyTorchDataFeeder(x, torch.float32, y, 'long', device)
                        for (x, y) in zip(train[0], train[1])]
        test_data = (   [to_tensor(x, device, torch.float32) for x in test[0]],
                        [to_tensor(y, device, 'long') for y in test[1]])  
    
    elif args.dset == 'metr-la':
        train, test,pre_A, scaler=load_METR()
        model = METRModel(device, scaler)
        feeders   = [   PyTorchDataFeeder(x, torch.float32, y, torch.float32, device,args.dset)
                    for (x, y) in zip(train[0], train[1])]
        test_data = (   [to_tensor(x, device, torch.float32) for x in test[0]],
                    [to_tensor(y, device, torch.float32) for y in test[1]])

    elif args.dset == 'pems-bay':
        train, test,pre_A ,scaler=load_PEMS()
        model = PEMSModel(device, scaler)
        print(model.parameters)
        feeders   = [   PyTorchDataFeeder(x, torch.float32, y, torch.float32, device)
                    for (x, y) in zip(train[0], train[1])]
        test_data = (   [to_tensor(x, device, torch.float32) for x in test[0]],
                    [to_tensor(y, device, torch.float32) for y in test[1]])
  
    
    # miscellaneous settings
    fname             = get_fname(args) # 保存实验数据
    M                 = int(args.W * args.C) # 每轮参与用户个数
    str_to_bn_setting = {'usyb':0, 'yb':1, 'us':2, 'none':3}
    if args.alg in ['fedavg','MTFL','PFedSA','GPFL','GPFL-C']:
        bn_setting = str_to_bn_setting[args.bn_private]

    # run experiment
    print('Starting experiment...')
    if args.alg == 'fedavg':
        client_optim = ClientSGD(model.parameters(), lr=args.lr)
        model.set_optim(client_optim)  
        # data=run_fedavg_MTFL(feeders, test_data, model, client_optim, args.T, M, args.B,args.dset,bn_setting=bn_setting)
        data = run_fedavg(feeders, test_data, model, client_optim, args.T, M, args.B,args.dset)
    elif args.alg == 'MTFL':
        client_optim = ClientSGD(model.parameters(), lr=args.lr)
        model.set_optim(client_optim)          
        data=run_fedavg_MTFL(feeders, test_data, model, client_optim, args.T, M, args.B,args.dset,bn_setting=bn_setting)
    
    elif args.alg == 'pfedme':
        client_optim = pFedMeOptimizer( model.parameters(), device, lr=args.lr, lamda=args.lamda)
        model.set_optim(client_optim)
        data = run_pFedMe(  feeders, test_data, model, args.T, M, K=1, B=args.B,dset=args.dset,
                             lamda=args.lamda, eta=args.lr, 
                            beta=args.beta)
    elif args.alg == 'SFL':
        client_optim = ClientSGD(model.parameters(), lr=args.lr)
        model.set_optim(client_optim)       
        data = run_SFL(  feeders, test_data, model, M, args, pre_A)        

    elif args.alg == 'GPFL':
        client_optim = ClientSGD(model.parameters(), lr=args.lr)
        model.set_optim(client_optim)     
        data = run_GPFL(  feeders, test_data, model, M, args, pre_A,bn_setting=bn_setting)

    elif args.alg == 'GPFL-C':
        client_optim = ClientSGD(model.parameters(), lr=args.lr)
        model.set_optim(client_optim)     
        data = run_GPFLC(  feeders, test_data, model, M, args, bn_setting=bn_setting)     
    
    # save result
    fname = get_fname(args)
    # save_data(data, fname)
    print('Data saved to: {}'.format(fname))

if __name__ == '__main__':
    main()
