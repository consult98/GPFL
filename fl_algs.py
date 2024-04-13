from operator import mul
import numpy as np
import pandas as pd
import pickle
import torch
from progressbar import progressbar
from main import save_data
from models import NumpyModel
from sklearn.cluster import KMeans,AgglomerativeClustering,SpectralClustering,DBSCAN
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# np.set_printoptions(threshold=np.inf)

def init_stats_arrays(T,n):
    """
    Returns:
        (tupe) of n numpy 0-filled float32 arrays of length T.
    """
    return tuple(np.zeros(T, dtype=np.float32) for i in range(n))


def flat(m):
    result=m[0].flatten()
    for item in m[1:]:
        result=np.hstack((result,item.flatten()))
    return result


# 聚类算法
def clustered_algorithm(dset,distance,round_model,user_params):
    user_params_dif=[] # 参数差异
    for param in user_params:
        if dset=='mnist':
            user_params_dif.append(flat(param[2:6])-flat(round_model[2:6]))
        elif dset=='cifar10':
            user_params_dif.append(flat(param[6:8])-flat(round_model[6:8]))
        elif dset in ['metr-la','pems-bay']:
            user_params_dif.append(flat(param)-flat(round_model))

    # 计算余弦相似度矩阵和距离矩阵
    cos_arr = cosine_similarity(user_params_dif) # 范围(-1,1)
    modify_cos_arr=cos_arr+1 # 范围(0,2)
    cos_arr = np.maximum(cos_arr, 0)
    dis_arr = 1 - cos_arr
    print(modify_cos_arr)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance, affinity='precomputed',linkage='average').fit(dis_arr)
    
    if dset in ['metr-la','pems-bay']:
        mu=normalize(cos_arr)
    else:
        mu=normalize(modify_cos_arr)
    print(mu)
    imp=compute_imp(cos_arr,clustering.labels_,clustering.n_clusters_)
    imp = np.round(imp, 2)
    
    return clustering.labels_,clustering.n_clusters_,imp,mu

def normalize(arr):
    arr_sum=np.sum(arr,axis=1).reshape(-1,1)
    mu=np.divide(arr,arr_sum)
    return mu

# 计算各客户端在簇中重要因子
def compute_imp(cos_arr,cluster_index,cluster_nums):
    imp=[0 for c in range(len(cos_arr))]
    print('imp:',len(cos_arr))
    for cluster_idx in range(cluster_nums):
        user_list=[index for (index, value) in enumerate(cluster_index) if value == cluster_idx]
        for user_idx in user_list:
            sum=1
            count=1
            for oth_idx in user_list:
                if oth_idx!=user_idx:
                    sum+=cos_arr[user_idx][oth_idx]
                    count+=1
            imp[user_idx]=sum/count
    return imp


def run_fedavg(data_feeders, test_data, model, client_opt,T, M, B,dset):
    W = len(data_feeders)
    
    train_errs, train_accs, test_errs, test_accs,test_mae,test_mape,test_rmse = init_stats_arrays(T,7)


    # global model/optimiser updated at the end of each round
    round_model = model.get_params()

    # stores accumulated client models/optimisers each round
    round_agg = model.get_params()
    
    for t in progressbar(range(T)):
        round_agg = round_agg.zeros_like()      

        # select round clients and compute their weights for later sum
        user_idxs = range(W)
        weights = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights = weights.astype(np.float32)
        weights /= np.sum(weights)

        round_n_test_users = 0
        
        for (w, user_idx) in zip(weights, user_idxs):
            # download global model/optim, update with private BN params
            model.set_params(round_model)

            # test local model               
            if dset in ['metr-la','pems-bay']:
                mae,mape,rmse = model.test(test_data[0][user_idx], test_data[1][user_idx], 128)
                test_mae[t] += mae
                test_mape[t] += mape
                test_rmse[t] += rmse  
            else:
                err, acc = model.test(test_data[0][user_idx], test_data[1][user_idx], 128)
                test_errs[t] += err
                test_accs[t] += acc
            round_n_test_users += 1
            
            K = int(data_feeders[user_idx].n_samples / B)
            user_err, user_acc = 0, 0
            # perform local SGD
            if dset in ['metr-la','pems-bay']:
                for k in range(K):    
                    x, y = data_feeders[user_idx].next_batch(B)
                    output,_ = model(x)       
                    output[:,:,0] = model.scaler.inverse_transform(output[:,:,0])
                    loss = model.loss_fn(output[:,:,0], y[:,:,0], 0.0)
                    model.optim.zero_grad()
                    loss.backward()
                    model.optim.step()
                    user_err += loss
                    user_acc =0
            else:
                for k in range(K):
                    x, y = data_feeders[user_idx].next_batch(B)
                    err, acc = model.train_step(x, y)
                    user_err += err
                    user_acc += acc
            train_errs[t] += user_err / K
            train_accs[t] += user_acc / K           

            # upload local model/optim to server, store private BN params
            round_agg = round_agg + (model.get_params() * w)           

        # new global model is weighted sum of client models
        round_model = round_agg.copy()

        if dset in ['metr-la','pems-bay']:
            test_mae[t] /= round_n_test_users
            test_mape[t] /= round_n_test_users
            test_rmse[t] /= round_n_test_users
            print('第{0}轮的 mae {1},mape {2},rmse {3}'.format(t, test_mae[t],test_mape[t],test_rmse[t]))   
        else:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users
            print('第{0}轮的测试精度 {1},loss {2}'.format(t, test_accs[t],test_errs[t]))
        train_errs[t] /= M
        train_accs[t] /= M

    # 计算所有用户的平均UA
    total_err, total_acc,total_mae,total_mape,total_rmse = 0,0,0,0,0
    acc_list = []
    for idx in range(W):
        model.set_params(round_model)
        # test local model
        if dset in ['metr-la','pems-bay']:
            mae,mape,rmse = model.test(test_data[0][idx], test_data[1][idx], 128)
            total_mae += mae
            total_mape += mape
            total_rmse += rmse    
        else:
            err, acc = model.test(test_data[0][idx], test_data[1][idx], 128)
            total_acc += acc
            total_err += err
            acc_list.append(acc.cpu())

    if dset in ['metr-la','pems-bay']:
        total_mae /= W
        total_mape /= W
        total_rmse /= W
        print('所有用户的mae为 {},mape为{},rmse为{}'.format(total_mae,total_mape,total_rmse))
        return test_mae,test_mape,test_rmse,total_mae,total_mape,total_rmse
    else:        
        total_acc /= W
        total_err /= W   
        print('所有用户的平均测试精度为 {}，平均loss为 {}'.format(total_acc, total_err))
        return train_errs, train_accs, test_errs, test_accs, total_err, total_acc.cpu(), acc_list


def run_fedavg_MTFL(data_feeders, test_data, model, client_opt,T, M, B,dset,test_freq=1, bn_setting=0, noisy_idxs=[]):
    W = len(data_feeders)
    
    train_errs, train_accs, test_errs, test_accs,test_mae,test_mape,test_rmse = init_stats_arrays(T,7)

    # contains private model and optimiser BN vals (if bn_setting != 3)
    user_bn_model_vals = [model.get_bn_vals(setting=bn_setting) for w in range(W)]
    user_bn_optim_vals = [client_opt.get_bn_params(model) for w in range(W)]

    # global model/optimiser updated at the end of each round
    round_model = model.get_params()
    round_optim = client_opt.get_params()

    # stores accumulated client models/optimisers each round
    round_agg = model.get_params()
    round_opt_agg = client_opt.get_params()

    for t in progressbar(range(T)):
        round_agg = round_agg.zeros_like()
        round_opt_agg = round_opt_agg.zeros_like()

        # select round clients and compute their weights for later sum
        # user_idxs = np.random.choice(W, M, replace=False)
        user_idxs = range(W)
        weights = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights = weights.astype(np.float32)
        weights /= np.sum(weights)

        round_n_test_users = 0

        for (w, user_idx) in zip(weights, user_idxs):
            # download global model/optim, update with private BN params
            model.set_params(round_model)
            client_opt.set_params(round_optim)
            model.set_bn_vals(user_bn_model_vals[user_idx], setting=bn_setting)
            client_opt.set_bn_params(user_bn_optim_vals[user_idx],
                                     model, setting=bn_setting)

            # test local model               
            if dset in ['metr-la','pems-bay']:
                mae,mape,rmse = model.test(test_data[0][user_idx], test_data[1][user_idx], 128)
                test_mae[t] += mae
                test_mape[t] += mape
                test_rmse[t] += rmse
            else:
                err, acc = model.test(test_data[0][user_idx], test_data[1][user_idx], 128)
                test_errs[t] += err
                test_accs[t] += acc
            round_n_test_users += 1
            
            K = int(data_feeders[user_idx].n_samples / B)
            user_err, user_acc = 0, 0
            # perform local SGD
            if dset in ['metr-la','pems-bay']:
                for k in range(K):
                    h=None
                    x, y = data_feeders[user_idx].next_batch(B)
                    output,_ = model(x)       
                    output[:,:,0] = model.scaler.inverse_transform(output[:,:,0])
                    loss = model.loss_fn(output[:,:,0], y[:,:,0], 0.0)
                    model.optim.zero_grad()
                    loss.backward()
                    model.optim.step()
                    user_err += loss
                    user_acc =0
            else:
                for k in range(K):
                    x, y = data_feeders[user_idx].next_batch(B)
                    err, acc = model.train_step(x, y)
                    user_err += err
                    user_acc += acc
            train_errs[t] += user_err / K
            train_accs[t] += user_acc / K           

            # upload local model/optim to server, store private BN params
            round_agg = round_agg + (model.get_params() * w)
            round_opt_agg = round_opt_agg + (client_opt.get_params() * w)
            user_bn_model_vals[user_idx] = model.get_bn_vals(setting=bn_setting)
            user_bn_optim_vals[user_idx] = client_opt.get_bn_params(model,setting=bn_setting)

        # new global model is weighted sum of client models
        round_model = round_agg.copy()
        round_optim = round_opt_agg.copy()

        train_errs[t] /= M
        train_accs[t] /= M

        if dset in ['metr-la','pems-bay']:
            test_mae[t] /= round_n_test_users
            test_mape[t] /= round_n_test_users
            test_rmse[t] /= round_n_test_users
            print('第{0}轮的 mae {1},mape {2},rmse {3} train_mae{4}'.format(t, test_mae[t],test_mape[t],test_rmse[t],train_errs[t]))   
        else:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users
            print('第{0}轮的测试精度 {1},loss {2}'.format(t, test_accs[t],test_errs[t]))
        


    # 计算所有用户的平均UA
    total_err, total_acc,total_mae,total_mape,total_rmse = 0, 0,0,0,0
    acc_list = []
    for idx in range(W):
        model.set_params(round_model)
        model.set_bn_vals(user_bn_model_vals[idx], setting=bn_setting)
        # test local model
        if dset in ['metr-la','pems-bay']:
            mae,mape,rmse = model.test(test_data[0][idx], test_data[1][idx], 128)
            total_mae += mae
            total_mape += mape
            total_rmse += rmse    
        else:
            err, acc = model.test(test_data[0][idx], test_data[1][idx], 128)
            total_acc += acc
            total_err += err
            acc_list.append(acc.cpu())

    if dset in ['metr-la','pems-bay']:
        total_mae /= W
        total_mape /= W
        total_rmse /= W
        print('所有用户的mae为 {},mape为{},rmse为{}'.format(total_mae,total_mape,total_rmse))
        return test_mae,test_mape,test_rmse,total_mae,total_mape,total_rmse,train_errs
    else:        
        total_acc /= W
        total_err /= W   
        print('所有用户的平均测试精度为 {}，平均loss为 {}'.format(total_acc, total_err))
        return train_errs, train_accs, test_errs, test_accs, total_err, total_acc.cpu(), acc_list


def run_pFedMe( data_feeders, test_data, model, T, M, K, B, dset,lamda, eta, 
                beta, test_freq=1, noisy_idxs=[]):
    
    W = len(data_feeders)
        
    train_errs, train_accs, test_errs, test_accs,test_mae,test_mape,test_rmse = init_stats_arrays(T,7)
    
    # global model updated at the end of each round, and round model accumulator 
    round_model = model.get_params()
    round_agg   = model.get_params()
    
    # client personalised models
    user_models = [round_model.copy() for w in range(W)]
    
    for t in progressbar(range(T), redirect_stdout=True):
        round_agg = round_agg.zeros_like()
        
        # select round clients and compute their weights for later sum
        user_idxs = range(W)
        weights   = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights   = weights.astype(np.float32)
        weights   /= np.sum(weights)
        
        round_n_test_users  = 0
                
        for (w, user_idx) in zip(weights, user_idxs):
            
            model.set_params(user_models[user_idx])
            # test local model               
            if dset in ['metr-la','pems-bay']:
                mae,mape,rmse = model.test(test_data[0][user_idx], test_data[1][user_idx], 128)
                test_mae[t] += mae
                test_mape[t] += mape
                test_rmse[t] += rmse   
            else:
                err, acc = model.test(test_data[0][user_idx], test_data[1][user_idx], 128)
                test_errs[t] += err
                test_accs[t] += acc
            round_n_test_users += 1
            
            # download global model
            model.set_params(round_model)
            
            user_err = 0
            R = int(data_feeders[user_idx].n_samples / B)
            # perform k steps of local training
            
            if dset in ['metr-la','pems-bay']:                    
                for r in range(R):
                    x, y = data_feeders[user_idx].next_batch(B)
                    omega = user_models[user_idx]
                    for k in range(K):
                        h=None
                        output = model(x)
                        output[:,:,0] = model.scaler.inverse_transform(output[:,:,0])        
                        loss = model.loss_fn(output[:,:,0], y[:,:,0], 0.0)
                        model.optim.zero_grad()
                        loss.backward()
                        model.optim.step(omega)
                        user_err += loss
                        user_acc =0

                    theta = model.get_params()                
                    user_models[user_idx] = omega - (lamda * eta * (omega - theta))
            else:
                for r in range(R):
                    x, y = data_feeders[user_idx].next_batch(B)
                    omega = user_models[user_idx]
                    for k in range(K):
                        model.optim.zero_grad()
                        logits = model.forward(x)
                        loss = model.loss_fn(logits, y)
                        loss.backward()        
                        model.optim.step(omega)
                    
                    theta = model.get_params()                
                    user_models[user_idx] = omega - (lamda * eta * (omega - theta))
            
            train_errs[t] += user_err / R    
            round_agg = round_agg + (user_models[user_idx] * w)
            
        # new global model is weighted sum of client models
        round_model = (1 - beta) * round_model + beta * round_agg
        
        train_errs[t] /= M
        if dset in ['metr-la','pems-bay']:
            test_mae[t] /= round_n_test_users
            test_mape[t] /= round_n_test_users
            test_rmse[t] /= round_n_test_users
            print('第{0}轮的 mae {1},mape {2},rmse {3}train_mae{4}'.format(t, test_mae[t],test_mape[t],test_rmse[t],train_errs[t]))   
        else:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users
            print('第{0}轮的测试精度 {1},loss {2}'.format(t, test_accs[t],test_errs[t]))

    # 计算所有用户的平均UA
    total_err, total_acc,total_mae,total_mape,total_rmse = 0, 0,0,0,0
    acc_list = []
    for idx in range(W):
        model.set_params(user_models[idx])
        # test local model
        if dset in ['metr-la','pems-bay']:
            mae,mape,rmse = model.test(test_data[0][idx], test_data[1][idx], 128)
            total_mae += mae
            total_mape += mape
            total_rmse += rmse    
        else:
            err, acc = model.test(test_data[0][idx], test_data[1][idx], 128)
            total_acc += acc
            total_err += err
            acc_list.append(acc.cpu())

    if dset in ['metr-la','pems-bay']:
        total_mae /= W
        total_mape /= W
        total_rmse /= W
        print('所有用户的mae为 {},mape为{},rmse为{}'.format(total_mae,total_mape,total_rmse))
        return test_mae,test_mape,test_rmse,total_mae,total_mape,total_rmse,train_errs
    else:        
        total_acc /= W
        total_err /= W   
        print('所有用户的平均测试精度为 {}，平均loss为 {}'.format(total_acc, total_err))
        return test_errs, test_accs,total_acc.cpu(), total_err,acc_list    


def run_SFL(data_feeders, test_data, model, M, args,pre_A, test_freq=1,noisy_idxs=[]):
    W = len(data_feeders)
    
    train_errs, train_accs, test_errs, test_accs,test_mae,test_mape,test_rmse = init_stats_arrays(args.T,7)   

    # global model/optimiser updated at the end of each round
    round_model = model.get_params()

    # stores accumulated client models/optimisers each round
    round_agg = model.get_params()

    for t in progressbar(range(args.T)):
        round_agg = round_agg.zeros_like()

        # select round clients and compute their weights for later sum
        user_idxs = range(W)
        weights = np.array([data_feeders[u].n_samples for u in user_idxs])
        weights = weights.astype(np.float32)
        weights /= np.sum(weights)

        round_n_test_users = 0
        
        round_user_model=[]
        for (w, user_idx) in zip(weights, user_idxs):           
            # download global model/optim, update with private BN params
            if t==0:    
                model.set_params(round_model)
            else:
                model.set_params(NumpyModel(after_round_user_model[user_idx]))          

            # test local model            
            if args.dset in ['metr-la','pems-bay']:
                mae,mape,rmse = model.test(test_data[0][user_idx], test_data[1][user_idx], 128)
                test_mae[t] += mae
                test_mape[t] += mape
                test_rmse[t] += rmse
            else:
                err, acc = model.test(test_data[0][user_idx], test_data[1][user_idx], 128)           
                test_errs[t] += err
                test_accs[t] += acc
            round_n_test_users += 1
           
            K = int(data_feeders[user_idx].n_samples / args.B)
            user_err, user_acc = 0, 0
            # perform local SGD
            if args.dset in ['metr-la','pems-bay']:
                for k in range(K):
                    x, y = data_feeders[user_idx].next_batch(args.B)
                    output,_ = model(x)
                    output[:,:,0] = model.scaler.inverse_transform(output[:,:,0])      
                    loss = model.loss_fn(output[:,:,0], y[:,:,0], 0.0)
                    model.optim.zero_grad()
                    loss.backward()
                    model.optim.step()
                    user_err += loss
                    user_acc =0
            else:
                for k in range(K):
                    x, y = data_feeders[user_idx].next_batch(args.B)
                    err, acc = model.train_step(x, y)
                    user_err += err
                    user_acc += acc
            train_errs[t] += user_err / K
            train_accs[t] += user_acc / K

            # upload local model/optim to server, store private BN params
            round_agg = round_agg + (model.get_params() * w)            
            round_user_model.append(model.get_params())
        # new global model is weighted sum of client models
        round_model = round_agg.copy()
        # 图聚合
        after_round_user_model=graph_dic(round_user_model,round_model,args,pre_A)
        
        train_errs[t] /= M
        train_accs[t] /= M

        if args.dset in ['metr-la','pems-bay']:
            test_mae[t] /= round_n_test_users
            test_mape[t] /= round_n_test_users
            test_rmse[t] /= round_n_test_users
            print('第{0}轮的 mae {1},mape {2},rmse {3} train_mae{4}'.format(t, test_mae[t],test_mape[t],test_rmse[t],train_errs[t]))   
        else:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users
            print('第{0}轮的测试精度 {1},loss {2}'.format(t, test_accs[t], test_errs[t]))
       

    # 计算所有用户的平均UA
    total_err, total_acc,total_mae,total_mape,total_rmse = 0, 0,0,0,0
    acc_list = []
    for idx in range(W):
        model.set_params(NumpyModel(after_round_user_model[idx]))
        if args.dset in ['metr-la','pems-bay']:
            mae,mape,rmse = model.test(test_data[0][idx], test_data[1][idx], 128)
            total_mae += mae
            total_mape += mape
            total_rmse += rmse   
        else:
            err, acc = model.test(test_data[0][idx], test_data[1][idx], 128)
            total_acc += torch.tensor(acc, device='cpu')
            total_err += err
            acc_list.append(torch.tensor(acc, device='cpu'))
    
    if args.dset in ['metr-la','pems-bay']:
        total_mae /= W
        total_mape /= W
        total_rmse /= W
        print('所有用户的mae为 {},mape为{},rmse为{}'.format(total_mae,total_mape,total_rmse))
        return test_mae,test_mape,test_rmse,total_mae,total_mape,total_rmse,train_errs 
    else:        
        total_acc /= W
        total_err /= W   
        print('所有用户的平均测试精度为 {}，平均loss为 {}'.format(total_acc, total_err))
        return train_errs, train_accs, test_errs, test_accs, total_err, total_acc, acc_list
                                                              

def run_GPFL(data_feeders, test_data, model, M, args,pre_A,assignment_list=None,bn_setting=3):
    
    W = len(data_feeders)
   
    train_errs, train_accs, test_errs, test_accs,test_mae,test_mape,test_rmse = init_stats_arrays(args.T,7)

    # global model/optimiser updated at the end of each round
    round_model = model.get_params()

    # stores accumulated client models/optimisers each round
    round_agg = model.get_params()

    cluster_bn_model_vals = [model.get_bn_vals(setting=bn_setting) for c in range(W)]
    user_bn_model_vals = [model.get_bn_vals(setting=bn_setting) for c in range(W)]

    user_idxs = range(W)
        
    # 计算各客户端全局聚合的加权值
    weights = np.array([data_feeders[u].n_samples for u in user_idxs])
    weights = weights.astype(np.float32)
    weights /= np.sum(weights)

    # 初始化
    mu = normalize(pre_A)
    cluster_num=1
    cluster_index=[0 for u in user_idxs]
    user_imp=compute_imp(np.ones((W,W)),cluster_index,cluster_num)


     # 开始训练
    for t in range(args.T): 
        round_agg = round_agg.zeros_like()
        cluster_bn_agg = [model.get_bn_vals(setting=bn_setting).zeros_like() for c in range(cluster_num)]
        cluster_imp = [0 for c in range(cluster_num)] # 各簇的重要因子总和
        
        for idx in user_idxs:
            cluster_imp[cluster_index[idx]]  += user_imp[idx]
       
        round_n_test_users = 0  # 第t轮的测试用户数
        round_user_model=[]
        print('-----------------开始第{0}轮，此轮参与用户个数{1}----------------------'.format(t, W))
        for (w, user_idx) in zip(weights, user_idxs):                
            # download global model/optim, update with private BN params         
            if t==0:
                model.set_params(round_model)
            else:
                model.set_params(NumpyModel(after_round_user_model[user_idx]))
                model.set_bn_vals(cluster_bn_model_vals[cluster_index[user_idx]], setting=bn_setting)

            # test local model               
            if args.dset in ['metr-la','pems-bay']:
                mae,mape,rmse = model.test(test_data[0][user_idx], test_data[1][user_idx], 128)
                test_mae[t] += mae
                test_mape[t] += mape
                test_rmse[t] += rmse
            
            else:
                err, acc = model.test(test_data[0][user_idx], test_data[1][user_idx], 128)
                test_errs[t] += err
                test_accs[t] += acc
            round_n_test_users += 1

            K = int(data_feeders[user_idx].n_samples / args.B)
            user_err, user_acc = 0, 0
            # perform local SGD
            if args.dset in ['metr-la','pems-bay']:
                for k in range(K):               
                    model.train()
                    model.optim.zero_grad()
                    x, y = data_feeders[user_idx].next_batch(args.B)
                    output,_ = model(x)
                    output[:,:,0] = model.scaler.inverse_transform(output[:,:,0])        
                    loss = model.loss_fn(output[:,:,0], y[:,:,0], 0.0)
                    loss.backward()
                    model.optim.step()
                    user_err += loss
                    user_acc =0
            else:
                for k in range(K):
                    x, y = data_feeders[user_idx].next_batch(args.B)
                    err, acc = model.train_step(x, y)
                    user_err += err
                    user_acc += acc
            train_errs[t] += user_err / K
            train_accs[t] += user_acc / K

            # upload local model/optim to server, store private BN params
            round_agg = round_agg + (model.get_params() * w)
            round_user_model.append(model.get_params())
            cluster_bn_agg[cluster_index[user_idx]] += model.get_bn_vals(setting=bn_setting) * user_imp[user_idx] / cluster_imp[cluster_index[user_idx]]
        round_model = round_agg.copy()
        
        # new global model is weighted sum of client models    
        for c_idx in range(cluster_num):
            if cluster_bn_agg[c_idx].notzero():
                cluster_bn_model_vals[c_idx]=cluster_bn_agg[c_idx].copy()
        # 图聚合
        after_round_user_model=graph_dic(round_user_model,round_model,args,pre_A,mu)

        # 更新图结构
        if t < args.preT:
            cluster_index,cluster_num,user_imp,mu= clustered_algorithm(args.dset,args.distance,round_model, round_user_model)    
            
        
        train_errs[t] /= M
        train_accs[t] /= M

        if args.dset in ['metr-la','pems-bay']:
            test_mae[t] /= round_n_test_users
            test_mape[t] /= round_n_test_users
            test_rmse[t] /= round_n_test_users
            print('第{0}轮的 mae {1},mape {2},rmse {3} train_mae{4}'.format(t, test_mae[t],test_mape[t],test_rmse[t],train_errs[t]))   
        else:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users
            print('第{0}轮的测试精度 {1},loss {2}'.format(t, test_accs[t],test_errs[t]))
        
        
    # 计算所有用户的平均UA
    total_err, total_acc,total_mae,total_mape,total_rmse = 0,0, 0,0,0
    acc_list = []
    for idx in range(W):
        model.set_params(NumpyModel(after_round_user_model[idx]))
        model.set_bn_vals(cluster_bn_model_vals[cluster_index[idx]], setting=bn_setting)
        # test local model
        if args.dset in ['metr-la','pems-bay']:
            mae,mape,rmse = model.test(test_data[0][idx], test_data[1][idx], 128)
            total_mae += mae
            total_mape += mape
            total_rmse += rmse    
        else:
            err, acc = model.test(test_data[0][idx], test_data[1][idx], 128)
            total_acc += torch.tensor(acc, device='cpu')
            total_err += err
            # acc_list.append(acc.cpu())
            acc_list.append(torch.tensor(acc, device='cpu'))
    
    if args.dset in ['metr-la','pems-bay']:
        total_mae /= W
        total_mape /= W
        total_rmse /= W
        print('所有用户的mae为 {},mape为{},rmse为{}'.format(total_mae,total_mape,total_rmse))
        return test_mae,test_mape,test_rmse,total_mae,total_mape,total_rmse,train_errs
    else:        
        total_acc /= W
        total_err /= W   
        print('所有用户的平均测试精度为 {}，平均loss为 {}'.format(total_acc, total_err))
        return train_errs, train_accs, test_errs, test_accs, total_acc, total_err, acc_list


# 图聚合
def graph_dic(models_list, round_model, args, pre_A,mu=None):
    pre_A=normalize(pre_A)
    if args.alg in ['GPFL','GPFL-G']:
        A=mu #只用学习的结构
        A = (1.0 - args.adjbeta) * pre_A + args.adjbeta * A #学习结构和已有结构占比
        print(A)
    else:
        A = pre_A #只用已有的结构
    
    # Aggregating
    aggregated_param = A.dot(np.array(models_list))
    for i in range(args.layers - 1):
        aggregated_param = A.dot(np.array(aggregated_param))
    new_models_list = np.multiply(aggregated_param, args.serveralpha) + np.multiply(models_list, (1 - args.serveralpha))
    

    return new_models_list


def run_GPFLC(data_feeders, test_data, model, M, args,bn_setting=3):
    
    W = len(data_feeders)
   
    train_errs, train_accs, test_errs, test_accs,test_mae,test_mape,test_rmse = init_stats_arrays(args.T,7)

    # global model/optimiser updated at the end of each round
    round_model = model.get_params()

    # stores accumulated client models/optimisers each round
    round_agg = model.get_params()

    cluster_bn_model_vals = [model.get_bn_vals(setting=bn_setting) for c in range(W)]

    user_idxs = range(W)
        
    # 计算各客户端全局聚合的加权值
    weights = np.array([data_feeders[u].n_samples for u in user_idxs])
    weights = weights.astype(np.float32)
    weights /= np.sum(weights)

    # 初始化
    cluster_num=1
    cluster_index=[0 for u in user_idxs]
    user_imp=compute_imp(np.ones((W,W)),cluster_index,cluster_num)

     # 开始训练
    for t in range(args.T): 
        round_agg = round_agg.zeros_like()
        cluster_bn_agg = [model.get_bn_vals(setting=bn_setting).zeros_like() for c in range(cluster_num)]
        cluster_imp = [0 for c in range(cluster_num)] # 各簇的重要因子总和
        
        for idx in user_idxs:
            cluster_imp[cluster_index[idx]]  += user_imp[idx]
       
        round_n_test_users = 0  # 第t轮的测试用户数
        round_user_model=[]
        print('-----------------开始第{0}轮，此轮参与用户个数{1}----------------------'.format(t, W))
        for (w, user_idx) in zip(weights, user_idxs):
            # download global model/optim, update with private BN params         
            model.set_params(round_model)
            model.set_bn_vals(cluster_bn_model_vals[cluster_index[user_idx]], setting=bn_setting)

            # test local model               
            if args.dset in ['metr-la','pems-bay']:
                mae,mape,rmse = model.test(test_data[0][user_idx], test_data[1][user_idx], 128)
                test_mae[t] += mae
                test_mape[t] += mape
                test_rmse[t] += rmse   
            else:
                err, acc = model.test(test_data[0][user_idx], test_data[1][user_idx], 128)
                test_errs[t] += err
                test_accs[t] += acc
            round_n_test_users += 1

            K = int(data_feeders[user_idx].n_samples / args.B)
            user_err, user_acc = 0, 0
            # perform local SGD
            if args.dset in ['metr-la','pems-bay']:
                for k in range(K):
                    h=None
                    x, y = data_feeders[user_idx].next_batch(args.B)
                    output = model(x)
                    output[:,:,0] = model.scaler.inverse_transform(output[:,:,0])        
                    loss = model.loss_fn(output[:,:,0], y[:,:,0], 0.0)
                    model.optim.zero_grad()
                    loss.backward()
                    model.optim.step()
                    user_err += loss
                    user_acc =0
            else:
                for k in range(K):
                    x, y = data_feeders[user_idx].next_batch(args.B)
                    err, acc = model.train_step(x, y)
                    user_err += err
                    user_acc += acc
            train_errs[t] += user_err / K
            train_accs[t] += user_acc / K

            # upload local model/optim to server, store private BN params
            round_agg = round_agg + (model.get_params() * w)
            round_user_model.append(model.get_params())
            cluster_bn_agg[cluster_index[user_idx]] += model.get_bn_vals(setting=bn_setting) * user_imp[user_idx] / cluster_imp[cluster_index[user_idx]]
        round_model = round_agg.copy()
        
        # new global model is weighted sum of client models    
        for c_idx in range(cluster_num):
            if cluster_bn_agg[c_idx].notzero():
                cluster_bn_model_vals[c_idx]=cluster_bn_agg[c_idx].copy()

        # 更新图结构
        if t < args.preT:
            cluster_index,cluster_num,user_imp,mu= clustered_algorithm(args.dset,args.distance,round_model, round_user_model)        
        
        train_errs[t] /= M
        train_accs[t] /= M

        if args.dset in ['metr-la','pems-bay']:
            test_mae[t] /= round_n_test_users
            test_mape[t] /= round_n_test_users
            test_rmse[t] /= round_n_test_users
            print('第{0}轮的 mae {1},mape {2},rmse {3} train_mae{4}'.format(t, test_mae[t],test_mape[t],test_rmse[t],train_errs[t]))   
        else:
            test_errs[t] /= round_n_test_users
            test_accs[t] /= round_n_test_users
            print('第{0}轮的测试精度 {1},loss {2}'.format(t, test_accs[t],test_errs[t]))
        
        
    # 计算所有用户的平均UA
    total_err, total_acc,total_mae,total_mape,total_rmse = 0,0, 0,0,0
    acc_list = []
    for idx in range(W):
        model.set_params(round_model)
        model.set_bn_vals(cluster_bn_model_vals[cluster_index[idx]], setting=bn_setting)
        # test local model
        if args.dset in ['metr-la','pems-bay']:
            mae,mape,rmse = model.test(test_data[0][idx], test_data[1][idx], 128)
            total_mae += mae
            total_mape += mape
            total_rmse += rmse    
        else:
            err, acc = model.test(test_data[0][idx], test_data[1][idx], 128)
            total_acc += torch.tensor(acc, device='cpu')
            total_err += err
            # acc_list.append(acc.cpu())
            acc_list.append(torch.tensor(acc, device='cpu'))
    
    if args.dset in ['metr-la','pems-bay']:
        total_mae /= W
        total_mape /= W
        total_rmse /= W
        print('所有用户的mae为 {},mape为{},rmse为{}'.format(total_mae,total_mape,total_rmse))
        return test_mae,test_mape,test_rmse,total_mae,total_mape,total_rmse,train_errs
    else:        
        total_acc /= W
        total_err /= W   
        print('所有用户的平均测试精度为 {}，平均loss为 {}'.format(total_acc, total_err))
        return train_errs, train_accs, test_errs, test_accs, total_acc, total_err, acc_list

