import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import os
import argparse
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import models
# from mixture_of_mvns import MixtureOfMVNs
# from mvn_diag import MultivariateNormalDiag

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--net', type=str, default='error')
parser.add_argument('--dataset',type=str, default='error')
parser.add_argument('--num_steps', type=int, default=50000)
parser.add_argument('--test_freq', type=int, default=200)
parser.add_argument('--save_freq', type=int, default=400)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if args.net == 'error':
    raise ValueError("No Net was specified" )
if args.dataset == 'error':
    raise ValueError("No Dataset was specified" )
if args.mode == 'test':
    raise NotImplementedError("Since Networks aren't saved yet, Testmode does not work yet")

#Get Network
if args.net == 'set_transformer':
    net, criterion = models.getSetTransformer(args.dataset)
elif args.net == 'deepset':
    net, criterion = models.getDeepSet(args.dataset)
else:
    raise ValueError('Invalid net {}'.format(args.net))

#Get Dataset -> Network, Criterion should automatically fit it
if args.dataset == 'maximum':
    raise NotImplementedError()
if args.dataset == 'pointcloud100':
    


def train():
    raise NotImplementedError
    #model.train(X,Y,max_steps) #Idea: each model has own train function, give input, labels and maxsteps

def test():
    raise NotImplementedError



if __name__ == '__main__':

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        bench = torch.load(benchfile)
        ckpt = torch.load(os.path.join(save_dir, 'model.tar'))
        net.load_state_dict(ckpt['state_dict'])
        test(bench)
    elif args.mode == 'plot':
        ckpt = torch.load(os.path.join(save_dir, 'model.tar'))
        net.load_state_dict(ckpt['state_dict'])
        plot()
