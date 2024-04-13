import gzip 
import numpy as np 
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
import os
import pickle
from collections import Counter

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def save_pickle(data,pickle_file):
    try:
        with open(pickle_file, 'wb') as f:
            pickle.dump(data,f)
    except Exception as e:
        print('Unable to save data ', pickle_file, ':', e)
        raise

class PyTorchDataFeeder():
    """
    Contains data as torch.tensors. Allows easy retrieval of data batches and
    in-built data shuffling.
    """

    def __init__(   self, x, x_dtype, y, y_dtype, device, 
                    dset=None,cast_device=None, transform=None):
        """
        Return a new data feeder with copies of the input x and y data. Data is 
        stored on device. If the intended model to use with the input data is 
        on another device, then cast_device can be passed. Batch data will be 
        sent to this device before being returned by next_batch. If transform is
        passed, the function will be applied to x data returned by next_batch.
        
        Args:
        - x:            x data to store
        - x_dtype:      torch.dtype or 'long' 
        - y:            y data to store 
        - y_dtype:      torch.dtype or 'long'
        - device:       torch.device to store data on 
        - cast_device:  data from next_batch is sent to this torch.device
        - transform:    function to apply to x data from next_batch
        """
        if x_dtype == 'long':
            self.x = torch.tensor(  x, device=device, 
                                    requires_grad=False, 
                                    dtype=torch.int32).long()
        else:
            self.x = torch.tensor(  x, device=device, 
                                    requires_grad=False, 
                                    dtype=x_dtype)
        
        if y_dtype == 'long':
            self.y = torch.tensor(  y, device=device, 
                                    requires_grad=False, 
                                    dtype=torch.int32).long()
        else:
            self.y = torch.tensor(  y, device=device, 
                                    requires_grad=False, 
                                    dtype=y_dtype)
        
        self.idx = 0
        self.n_samples = x.shape[0]
        self.cast_device = cast_device
        self.transform = transform
        self.dset = dset
        self.shuffle_data()
        
    def shuffle_data(self):
        """
        Co-shuffle x and y data.
        """
        ord = torch.randperm(self.n_samples)
        self.x = self.x[ord]
        self.y = self.y[ord]
        
    def next_batch(self, B):
        """
        Return batch of data If B = -1, the all data is returned. Otherwise, a 
        batch of size B is returned. If the end of the local data is reached, 
        the contained data is shuffled and the internal counter starts from 0.
        
        Args:
        - B:    size of batch to return
        
        Returns (x, y) tuple of torch.tensors. Tensors are placed on cast_device
                if this is not None, else device.
        """
        if B == -1:
            x = self.x
            y = self.y
            self.shuffle_data()
            
        elif self.idx + B > self.n_samples:
            # if batch wraps around to start, add some samples from the start
            extra = (self.idx + B) - self.n_samples
            x = torch.cat((self.x[self.idx:], self.x[:extra]))
            y = torch.cat((self.y[self.idx:], self.y[:extra]))
            self.shuffle_data()
            self.idx = extra
            
        else:
            x = self.x[self.idx:self.idx+B]
            y = self.y[self.idx:self.idx+B]
            self.idx += B
            
        if not self.cast_device is None:
            x = x.to(self.cast_device)
            y = y.to(self.cast_device)

        if not self.transform is None:
            x = self.transform(x)

        return x, y



def load_mnist(data_dir, W, iid, user_test=False):
    """
    Load the MNIST_data dataset. The folder specified by data_dir should contain the
    standard MNIST_data files from (yann.lecun.com/exdb/mnist/).

    Args:
        - data_dir:  (str)  path to data folder
        - W:         (int)  number of workers to split dataset into
        - iid:       (bool) iid (True) or non-iid shard-based split (False)
        - user_test: (bool) split test data into users
        
    Returns:
        Tuple containing ((x_train, y_train), (x_test, y_test)). The training 
        variables are both lists of length W, each element being a 2D numpy 
        array. If user_test is True, the test variables will also be lists, 
        otherwise the returned test values are just 2D numpy arrays.
    """
    train_x_fname = data_dir + '/train-images-idx3-ubyte.gz'
    train_y_fname = data_dir + '/train-labels-idx1-ubyte.gz'
    test_x_fname = data_dir + '/t10k-images-idx3-ubyte.gz'
    test_y_fname = data_dir + '/t10k-labels-idx1-ubyte.gz'

    # load MNIST_data files
    with gzip.open(train_x_fname) as f:
        x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
        x_train = x_train.astype(np.float32) / 255.0

    with gzip.open(train_y_fname) as f:
        y_train = np.copy(np.frombuffer(f.read(), np.uint8, offset=8))

    with gzip.open(test_x_fname) as f:
        x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
        x_test = x_test.astype(np.float32) / 255.0        

    with gzip.open(test_y_fname) as f:
        y_test = np.copy(np.frombuffer(f.read(), np.uint8, offset=8))
    
    # split into iid/non-iid and users
    if iid=='iid':
        x_train, y_train = co_shuffle_split(x_train, y_train, W)
        if user_test:
            x_test, y_test = co_shuffle_split(x_test, y_test, W)
    
    elif iid=='non-iid1':
        x_train, y_train, assignment_list= shard_split(x_train, y_train, W, W * 2)
        if user_test:
            x_test, y_test, _ = shard_split(x_test, y_test, W, W * 2, assignment_list)
    else:
        x_train, y_train,pre_A,assignment_list,group_user_list,group_usernum_list= data_split_mnist(x_train, y_train, W)
        
        if user_test:
            x_test, y_test,_,_,_,_ = data_split_mnist(x_test, y_test, W, assignment_list,group_user_list,group_usernum_list)
    
    return (x_train, y_train), (x_test, y_test),pre_A,assignment_list



def load_CIFAR_batch(filename):
    with open(filename,'rb') as f:
        datadict=pickle.load(f, encoding='latin1')
        X=datadict['data']
        Y=datadict['labels']
        X=X.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
        Y=np.array(Y)
        return X,Y
               
def load_CIFAR10(ROOT):
    xs=[]
    ys=[]
    for b in range(1,6):
        f=os.path.join(ROOT,'data_batch_%d'%(b,))#os.path.join()将多个路径组合后返回
        X,Y=load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr=np.concatenate(xs)#这个函数用于将多个数组进行连接
    Ytr=np.concatenate(ys)
    del X,Y
    Xte,Yte=load_CIFAR_batch(os.path.join(ROOT,'test_batch'))
    return Xtr,Ytr,Xte,Yte

def load_cifar(data_dir, W, iid, user_test=False):
    """
    Load the CIFAR dataset. The folder specified by data_dir should contain the
    python version pickle files from (cs.toronto.edu/~kriz/cifar.html). 

    Args:
        - data_dir:  (str)  path to data folder
        - W:         (int)  number of workers to split dataset into
        - iid:       (bool) iid (True) or non-iid shard-based split (False)
        - user_test: (bool) split test data into users
        
    Returns:
        Tuple containing ((x_train, y_train), (x_test, y_test)). The training 
        variables are both lists of length W, each element being a 4D numpy 
        array. If user_test is True, the test variables will also be lists, 
        otherwise the returned test values are just 4D numpy arrays.
    """
    fnames = [  '/data_batch_1', 
                '/data_batch_2', 
                '/data_batch_3',
                '/data_batch_4', 
                '/data_batch_5']

    x_train,y_train,x_test,y_test=load_CIFAR10('CIFAR10_data')
    
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    
    # split into iid/non-iid and users
    if iid=='iid':
        x_train, y_train = co_shuffle_split(x_train, y_train, W)
        if user_test:
            x_test, y_test = co_shuffle_split(x_test, y_test, W)

    elif iid=='non-iid1':
        x_train, y_train, assignment_list= shard_split(x_train, y_train, W, W * 2)
        if user_test:
            x_test, y_test, _ = shard_split(x_test, y_test, W, W * 2, assignment_list)
    
    else:
        x_train, y_train, pre_A,assignment_list,group_user_list,group_usernum_list= data_split_cifar(x_train, y_train, W)
        if user_test:
            x_test, y_test,_,_,_,_ = data_split_cifar(x_test, y_test, W, assignment_list,group_user_list,group_usernum_list)
    

    return (x_train, y_train), (x_test, y_test), pre_A,assignment_list


def load_METR():
    # x_train, y_train=load_pickle('./data/METR-LA/train_data.pkl')   
    # x_test, y_test=load_pickle('./data/METR-LA/test_data.pkl')
   
    train_data = np.load('./data/METR-LA/train.npz')
    x_train=train_data['x']
    y_train=train_data['y']
    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())
    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_train = x_train.transpose((2,0,1,3))
    y_train = y_train.transpose((2,0,1,3))
    # selected_index = list(np.random.choice(x_train.shape[1],int(x_train.shape[1] * 0.1)))
    # x_train = x_train[:,selected_index,:,:]
    # y_train = y_train[:,selected_index,:,:]

    test_data = np.load('./data/METR-LA/test.npz')
    x_test=test_data['x']
    y_test=test_data['y']
    
    x_test[..., 0] = scaler.transform(x_test[..., 0])
    x_test = x_test.transpose((2,0,1,3))
    y_test = y_test.transpose((2,0,1,3))
    # selected_index = list(np.random.choice(x_test.shape[1],int(x_test.shape[1] * 0.1)))
    # x_test = x_test[:,selected_index,:,:]
    # y_test = y_test[:,selected_index,:,:]

    val_data = np.load('./data/METR-LA/val.npz')
    x_val=val_data['x']
    y_val=val_data['y']
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_val = x_val.transpose((2,0,1,3))
    y_val = y_val.transpose((2,0,1,3))
    # selected_index = list(np.random.choice(x_val.shape[1],int(x_val.shape[1] * 0.1)))
    # x_val = x_val[:,selected_index,:,:]
    # y_val = y_val[:,selected_index,:,:]

    sensor_ids, sensor_id_to_ind, pre_A = load_pickle('./data/sensor_graph/adj_mx.pkl')
    return (x_train, y_train), (x_test, y_test), pre_A, scaler

def load_PEMS():
    # x_train, y_train=load_pickle('./data/PEMS-BAY/train_data.pkl')
    # x_test, y_test=load_pickle('./data/PEMS-BAY/test_data.pkl')

    train_data = np.load('./data/PEMS-BAY/train.npz')
    x_train=train_data['x']
    y_train=train_data['y']
    scaler = StandardScaler(mean=x_train[..., 0].mean(), std=x_train[..., 0].std())
    x_train[..., 0] = scaler.transform(x_train[..., 0])
    x_train = x_train.transpose((2,0,1,3))
    y_train = y_train.transpose((2,0,1,3))
    # selected_index = list(np.random.choice(x_train.shape[1],int(x_train.shape[1] * 0.1)))
    # x_train = x_train[:,selected_index,:,:]
    # y_train = y_train[:,selected_index,:,:]

    test_data = np.load('./data/PEMS-BAY/test.npz')
    x_test=test_data['x']
    y_test=test_data['y']
    x_test[..., 0] = scaler.transform(x_test[..., 0])
    x_test = x_test.transpose((2,0,1,3))
    y_test = y_test.transpose((2,0,1,3))
    # selected_index = list(np.random.choice(x_test.shape[1],int(x_test.shape[1] * 0.1)))
    # x_test = x_test[:,selected_index,:,:]
    # y_test = y_test[:,selected_index,:,:]

    val_data = np.load('./data/PEMS-BAY/val.npz')
    x_val=val_data['x']
    y_val=val_data['y']
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_val = x_val.transpose((2,0,1,3))
    y_val = y_val.transpose((2,0,1,3))
    # selected_index = list(np.random.choice(x_val.shape[1],int(x_val.shape[1] * 0.1)))
    # x_val = x_val[:,selected_index,:,:]
    # y_val = y_val[:,selected_index,:,:]
    sensor_ids, sensor_id_to_ind, pre_A = load_pickle('./data/sensor_graph/adj_mx_bay.pkl')
    return (x_train, y_train), (x_test, y_test), pre_A ,scaler

def save_trans(data_dir):
    train_fname = data_dir + '/train.npz'
    test_fname = data_dir + '/test.npz'
    val_fname = data_dir + '/val.npz'

    train_data = np.load(train_fname)
    x_train=train_data['x']
    y_train=train_data['y']
    print(x_train.shape,x_train)
    x_train = x_train.transpose((2,0,1,3))
    print(x_train.shape,x_train[0][0],x_train[1][0])
    test_data = np.load(test_fname)
    x_test=test_data['x']
    y_test=test_data['y']
    val_data = np.load(val_fname)
    x_val=val_data['x']
    y_val=val_data['y']

    x_val, y_val= data_split_trans(x_val, y_val)

    save_pickle((x_val, y_val),data_dir +'/val_data.pkl')


def data_split_trans(x, y):

    x_data = [np.zeros(1) for i in range(x.shape[2])]
    y_data = [np.zeros(1) for i in range(x.shape[2])]

    init_flag=True
    item=1
    for x_item in x: #一条训练数据        
        for x_item_time in x_item: #一个时间点
            for user_idx,x_item_time_user in enumerate(x_item_time): #一个用户
                if init_flag:
                    x_data[user_idx]=x_item_time_user
                else:
                    x_data[user_idx]=np.append(x_data[user_idx],x_item_time_user,axis=0)
            init_flag=False
        print(item)
        item=item+1
        
    for i in range(x.shape[2]):
        x_data[i]=x_data[i].reshape(-1,12,2)


    init_flag=True
    item=1
    for y_item in y: #一条训练数据        
        for y_item_time in y_item: #一个时间点
            for user_idx,y_item_time_user in enumerate(y_item_time): #一个用户
                if init_flag:
                    y_data[user_idx]=y_item_time_user
                else:
                    y_data[user_idx]=np.append(y_data[user_idx],y_item_time_user,axis=0)
            init_flag=False
        print(item)
        item=item+1
        
    for i in range(x.shape[2]):
        y_data[i]=y_data[i].reshape(-1,12,2)

    return x_data,y_data


def co_shuffle_split(x, y, W):
    """
    Shuffle x and y using the same random order, split into W parts.
    
    Args:
        - x: (np.ndarray) samples
        - y: (np.ndarray) labels
        - W: (int)        num parts to split into
            
    Returns:
        Tuple containing (list of x arrays, list of y arrays)
    """
    order = np.random.permutation(x.shape[0])
    x_split = np.array_split(x[order], W)
    y_split = np.array_split(y[order], W)
    
    return x_split, y_split



def shard_split(x, y, W, n_shards, assignment=None):
    """
    Split x and y into W parts. Arrays are sorted according to classes in y,
    split into n_shards. If assignment is None, each W is assigned random
    shards, otherwise, the passed assignment is used. This function therefore
    creates a non-iid split based on classes.

    Args:
        - x:         (np.ndarray) samples
        - y:         (np.ndarray) labels
        - W:         (int)        num parts to split into
        - n_shards   (int)        num shards per W
        - assignment (np.array)   pre-determined shard assingment, or None

    Returns:
        Tuple containing (list of x arrays, list of y arrays, assingment), where
        assignment is created at random if passed assignment is None, otherwise
        just passed back out.
    """
    order = np.argsort(y)
    x_sorted = x[order]
    y_sorted = y[order]

    # split data into shards of (mostly) the same index
    x_shards = np.array_split(x_sorted, n_shards)
    y_shards = np.array_split(y_sorted, n_shards)

    print('共分为',n_shards,'类')

    #assignment:分配的数据分布
    if assignment is None:
        assignment = np.array_split(np.random.permutation(n_shards), W)
        print('每个客户端的数据分布为', assignment)

    x_sharded = []
    y_sharded = []

    # assign each worker two shards from the random assignment
    for w in range(W):
        x_sharded.append(np.concatenate([x_shards[i] for i in assignment[w]]))
        y_sharded.append(np.concatenate([y_shards[i] for i in assignment[w]]))

    return x_sharded, y_sharded, assignment


def data_split_cifar(x, y, W,assignment_list=None,group_user_list=None,group_usernum_list=None):
    order = np.argsort(y)
    x_sorted = x[order]
    y_sorted = y[order]

    # group_split
    groups_num = 3 # 群组数
    group_size= [0]*groups_num
    group_size[0]=np.sum((y_sorted==0)|(y_sorted==1)|(y_sorted == 2) | (y_sorted == 3))
    group_size[1]= group_size[0]+np.sum((y_sorted == 4) | (y_sorted == 5)|(y_sorted == 6) )
    group_size[2] = group_size[1]+ np.sum((y_sorted == 7) | (y_sorted == 8) |(y_sorted == 9) )
    x_groups = np.array_split(x_sorted , [group_size[0],group_size[1]])
    y_groups = np.array_split(y_sorted, [group_size[0],group_size[1]])
    print('数据共{0}条，分为{1}组'.format(len(y), groups_num),group_size)


    # 打乱每一个群组的数据分布
    for i in range(groups_num):
        state = np.random.get_state()
        np.random.shuffle(x_groups[i])
        np.random.set_state(state)
        np.random.shuffle(y_groups[i])

    # assignment_list:分配的数据分布
    if assignment_list is None:
        # 每个群组中的用户数
        group_usernum_list = [0] * groups_num
        # 每个群组中的用户
        group_user_list = [[] for i in range(groups_num)]
        assignment_list=[]

        for w in range(W):
            group_choices=np.random.choice(groups_num, 1)
            assignment_list.append(group_choices[0])
            for group_choice in group_choices:
                group_user_list[group_choice].append(w)
                group_usernum_list[group_choice] += 1
        print('每个客户端的数据分布为:', assignment_list)
        print('每个群组的客户分别为:', group_user_list)
        print('每个群组的客户数分别为:', group_usernum_list)

    # define pre_A
    link_list = []
    for user_arr in group_user_list:
            for user_a in user_arr:
                for user_b in user_arr:
                    link_list.append([user_a, user_b])

    pre_A = np.zeros((W, W))
    link_idx=list(range(len(link_list)))
    edge_frac = 0.5
    link_idx = np.random.choice(link_idx, int(edge_frac * len(link_idx)), replace=False)
    for idx in link_idx:
        pre_A[link_list[idx][0], link_list[idx][1]] = 1
    
    for i in range(W):
        pre_A[i, i] = 1
        
    # 给每个客户端划分数据
    x_data = [np.zeros(1) for i in range(W)]
    y_data = [np.zeros(1) for i in range(W)]

    for (group_idx,group_user) in enumerate(group_user_list):
        x_data_tem=np.array_split(x_groups[group_idx], group_usernum_list[group_idx])
        y_data_tem=np.array_split(y_groups[group_idx], group_usernum_list[group_idx])
        for idx,user in enumerate(group_user):
            # 分配主类
            x_data[user]=x_data_tem[idx]
            y_data[user]=y_data_tem[idx]
            # # 分配次类
            # data_length=len(x_data[user])
            # for oth_group_idx in range(groups_num):
            #     if oth_group_idx!=group_idx:
            #         length=len(x_groups[oth_group_idx])
            #         start=np.random.randint(length)     
            #         x_data[user]=np.append(x_data[user],np.array_split(x_groups[oth_group_idx], [start,start+int(data_length*0.3)])[1])
            #         y_data[user]=np.append(y_data[user],np.array_split(y_groups[oth_group_idx], [start,start+int(data_length*0.3)])[1])
            #         x_data[user]=x_data[user].reshape(-1,3,32,32)
            # print(user,'-------------',y_data[user])
    return x_data, y_data,pre_A,assignment_list,group_user_list,group_usernum_list

def data_split_mnist(x, y, W,assignment_list=None,group_user_list=None,group_usernum_list=None):
    order = np.argsort(y)
    x_sorted = x[order]
    y_sorted = y[order]

   # group_split
    groups_num = 3 # 群组数
    group_size= [0]*groups_num
    group_size[0]=np.sum((y_sorted==0)|(y_sorted==1)|(y_sorted == 2) | (y_sorted == 3))
    group_size[1]= group_size[0]+np.sum((y_sorted == 4) | (y_sorted == 5)|(y_sorted == 6) )
    group_size[2] = group_size[1]+ np.sum((y_sorted == 7) | (y_sorted == 8) |(y_sorted == 9) )
    x_groups = np.array_split(x_sorted , [group_size[0],group_size[1]])
    y_groups = np.array_split(y_sorted, [group_size[0],group_size[1]])
    print('数据共{0}条，分为{1}组'.format(len(y), groups_num),group_size)

    # 打乱每一个群组的数据分布
    for i in range(groups_num):
        state = np.random.get_state()
        np.random.shuffle(x_groups[i])
        np.random.set_state(state)
        np.random.shuffle(y_groups[i])

    # assignment_list:分配的数据分布
    if assignment_list is None:
        # 每个群组中的用户数
        group_usernum_list = [0] * groups_num
        # 每个群组中的用户
        group_user_list = [[] for i in range(groups_num)]
        assignment_list=[]

        for w in range(W):
            group_choices=np.random.choice(groups_num, 1)
            assignment_list.append(group_choices[0])
            for group_choice in group_choices:
                group_user_list[group_choice].append(w)
                group_usernum_list[group_choice] += 1
        print('每个客户端的数据分布为:', assignment_list)
        print('每个群组的客户分别为:', group_user_list)
        print('每个群组的客户数分别为:', group_usernum_list)

    # define pre_A
    link_list = []
    for user_arr in group_user_list:
            for user_a in user_arr:
                for user_b in user_arr:
                    link_list.append([user_a, user_b])

    pre_A = np.zeros((W, W))
    link_idx=list(range(len(link_list)))
    edge_frac=0.5
    link_idx = np.random.choice(link_idx, int(edge_frac * len(link_idx)), replace=False)
    for idx in link_idx:
        pre_A[link_list[idx][0], link_list[idx][1]] = 1
    for i in range(W):
        pre_A[i, i] = 1


    # 给每个客户端划分数据
    x_data = [np.zeros(1) for i in range(W)]
    y_data = [np.zeros(1) for i in range(W)]

    for (group_idx,group_user) in enumerate(group_user_list):
        x_data_tem=np.array_split(x_groups[group_idx], group_usernum_list[group_idx])
        y_data_tem=np.array_split(y_groups[group_idx], group_usernum_list[group_idx])
        for idx,user in enumerate(group_user):
            # 分配主类
            x_data[user]=x_data_tem[idx]
            y_data[user]=y_data_tem[idx]
            # # 分配次类
            # data_length=len(x_data[user])
            # for oth_group_idx in range(groups_num):
            #     if oth_group_idx!=group_idx:
            #         length=len(x_groups[oth_group_idx])
            #         start=np.random.randint(length)    
            #         x_data[user]=np.append(x_data[user],np.array_split(x_groups[oth_group_idx], [start,start+int(data_length*0.1)])[1])
            #         y_data[user]=np.append(y_data[user],np.array_split(y_groups[oth_group_idx], [start,start+int(data_length*0.1)])[1])
            #         x_data[user]=x_data[user].reshape(-1,784)
            # print(user,'-------------',y_data[user])
    return x_data, y_data,pre_A,assignment_list,group_user_list,group_usernum_list



def to_tensor(x, device, dtype):
    """
    Convert Numpy array to torch.tensor.
    
    Args:
    - x:        (np.ndarray)   array to convert
    - device:   (torch.device) to place tensor on
    - dtype:    (torch.dtype)  or 'long' to convert to pytorch long
    """
    if dtype == 'long':
        return torch.tensor(x, device=device, 
                            requires_grad=False, 
                            dtype=torch.int32).long()
    else:
        return torch.tensor(x, device=device, 
                            requires_grad=False, 
                            dtype=dtype)
