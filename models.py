
import torch
import torch.nn as nn
import Models.DeepSet as DeepSet
import Models.Pointnet as PointNet
import Models.RepSet as RepSet
import Models.SetTransformer as SetTransformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#returns net, criterion
def getmodel(nettype, dataset, pool):
    nettype, dataset, pool = nettype.lower(), dataset.lower(), pool.lower()
    if nettype == 'error':
        raise ValueError("No Net was specified" )
    if dataset == 'error':
        raise ValueError("No Dataset was specified" )
    if pool != 'max' and pool != 'mean' and pool != 'fspool' and pool != 'default' and pool!='sum':
        raise ValueError(f'{pool} is a invalid pooling function')
    if dataset == 'minmax' or dataset =='2min2max':
        dataset = 'min2max2'

    #Get Network
    if nettype == 'set_transformer' or nettype == 'settransformer':
        return  getSetTransformer(dataset, pool)
    elif nettype == 'deepset' or nettype == 'deep_set':
        return getDeepSet(dataset, pool)
    elif nettype == 'pointnet' or nettype == 'point_net':
        return getPointNet(dataset, pool)
    elif nettype == 'repset' or nettype == 'rep_set':
        return getRepSet(dataset, pool)
    else:
        raise ValueError('Invalid net {}'.format(nettype))


def getSetTransformer(dataset, pool):
    if pool != 'default':
        print(f"Warning: SetTransformer does not use a pooling function, so pool={pool} will be ignored")
    if dataset == 'pointcloud100' or dataset =='pointcloud1000'or dataset =='pointcloud5000':
        net = SetTransformer.SetTransformerPointCloud(dim_hidden = 256, num_heads = 4, num_inds=16)
        criterion = nn.CrossEntropyLoss()
    elif dataset == 'maximum' or dataset=='cardinality' or dataset == 'mode':
        net = SetTransformer.SmallSetTransformer()
        criterion = nn.L1Loss()
    elif dataset == 'max4' or dataset == 'min2max2':
        net = SetTransformer.Max4SetTransformer()
        criterion = nn.L1Loss()
    elif dataset == 'eq2':
        net = SetTransformer.BinarySetTransformer(dim_hidden = 32, num_heads = 32, num_inds=16)
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("This Dataset does not work: {}".format(dataset))
    return net, criterion


def getDeepSet(dataset, pool):
    if pool == 'default':
        pool = 'sum'
    if dataset == 'maximum' or dataset=='cardinality' or dataset == 'mode':
        net = DeepSet.SmallDeepSet(pool= pool)
        criterion = nn.L1Loss()
    elif dataset == 'max4' or dataset == 'min2max2':
        net = DeepSet.Max4DeepSet(pool=pool)
        criterion = nn.L1Loss()
    elif dataset == 'pointcloud100' or dataset == 'pointcloud1000':
        net = DeepSet.DeepSetPointCloud(d_dim=256, equipool='max', pool = pool) #will take max for equivariant layers, then mean to combine all inputs
        criterion = nn.CrossEntropyLoss()
    elif dataset == 'pointcloud5000':
        net =DeepSetPointCloud.D(d_dim=512, equipool='max', pool = pool) #will take max for equivariant layers, then mean to combine all inputs
        criterion = nn.CrossEntropyLoss()
    # elif dataset == 'mnist_sum' or dataset =='MNIST_sum' or dataset == 'mnistsum' or dataset == 'MNISTsum':
    #     #pool not implemented here yet
    #     print("Warning: pool is not implemented for this dataset")
    #     net = DeepSet.MNISTsumDeepSet(dim_input=784, num_outputs=1)
    #     criterion = nn.L1Loss()
    elif dataset == 'eq2':
        net = DeepSet.BinaryDeepSet(dim_input=1, dim_hidden = 64, pool=pool)
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("This Dataset does not work:{}".format(dataset))
    return net, criterion

def getPointNet(d, pool):
    if d =='pointcloud100' or d=='pointcloud1000' or d=='pointcloud5000':
        net = PointNet.PointNet(classes = 40, pool=pool)
        criterion = nn.NLLLoss()
        return net, criterion
    else: raise ValueError("pointnet only works on pointclouds, not with{}".format(d))


#so far only ApproxRepSet
def getRepSet(d, pool):
    if pool != 'default':
        print(f"Warning: RepSet does not use a pooling function, so pool={pool} will be ignored")
    if d == 'pointcloud100':
        net = RepSet.ApproxRepSetClassifier(n_hidden_sets = 10, n_elements=20, d = 3, n_classes = 40)
        criterion = nn.CrossEntropyLoss()
        return net, criterion
    if d == 'maximum' or d== 'cardinality' or d=='mode':
        net = RepSet.ApproxRepSet(n_hidden_sets = 20, n_elements=50, d = 1, n_out = 1)
        criterion = nn.L1Loss()
        return net, criterion
    if d == 'max4' or d=='min2max2':
            net = RepSet.ApproxRepSet(n_hidden_sets = 10, n_elements=20, d = 1, n_out = 4)
            criterion = nn.L1Loss()
            return net, criterion
    if d== 'eq2':
        net = RepSet.ApproxRepSetClassifier(n_hidden_sets = 10, n_elements=20, d = 1, n_classes = 2)
        criterion = nn.CrossEntropyLoss()
        return net, criterion
    else: raise ValueError("This Dataset does not work:{}".format(d))
