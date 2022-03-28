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
import datasets
from helperFunctions import wasCorrect, getProblemType, getError
# from mixture_of_mvns import MixtureOfMVNs
# from mvn_diag import MultivariateNormalDiag



parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--net', type=str, default='error')
parser.add_argument('--dataset',type=str, default='error')
parser.add_argument('--num_epochs', type=int, default=5000)
parser.add_argument('--test_freq', type=int, default=5)
parser.add_argument('--pool', type=str, default='default')
# parser.add_argument('--earlystop', type=str, default = 'true')
#parser.add_argument('--save_freq', type=int, default=400)


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
useCuda = torch.cuda.is_available()
if useCuda:
    print("Yay Cuda was available!")

#might implement this, definitely helpfull if i run a training on a cluster
# earlystop = args.earlystop.lower()
# if earlystop == 'true' or earlystop == 'yes' or earlystop == '+':
#     earlystop = True
# else: earlystop = False


if args.mode == 'test':
    raise NotImplementedError("Since Networks aren't saved yet, Testmode does not work yet")

net, criterion = models.getmodel(args.net, args.dataset, args.pool)
if(useCuda):
    net.cuda()

#Get Dataset -> Network, Criterion should automatically fit it
#idea: dataset is in data -> data.train_data, data.test_data
data = datasets.getdata(lower(args.dataset))
test_freq = args.test_freq
problemtype = getProblemType(args.dataset)
if args.dataset == 'eq2':
    data.avgY()


def train():
    #model.train(X,Y,max_steps) #Idea: each model has own train function, give input, labels and maxsteps
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    #test() #should be really low

    for epoch in range(1,args.num_epochs):
        net.train()
        losses, total, correct, error = [], 0, 0, 0
        for X,Y in data.train_data():
            X = torch.Tensor(X)
            Y = torch.Tensor(Y)
            if(problemtype == 0):
                Y = Y.long()
            if(useCuda):
                X = X.cuda(device)
                Y = Y.cuda(device)
            #print(X)
            pred = net(X)
            loss = criterion(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            total += Y.shape[0]
            correct += wasCorrect(pred, Y, problemtype)
            if problemtype !=0: #not classifier
                error += getError(pred, Y, problemtype)

        avg_loss, avg_acc, avg_err = np.mean(losses), correct / total, error/total
        print(f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f} ", end='')
        if problemtype !=0:
            print(f'train error {avg_err:.3f}', end='')
        print('')

        if epoch % test_freq == 0:
            test()



def test():
    net.eval()
    with torch.no_grad():
        losses, total, correct, error = [], 0, 0, 0
        for X,Y in data.test_data():
            X = torch.Tensor(X)
            Y = torch.Tensor(Y)
            if(problemtype == 0):
                Y = Y.long()
            if(useCuda):
                X = X.cuda(device)
                Y = Y.cuda(device)
            #print(X)
            pred = net(X)
            loss = criterion(pred, Y)
            losses.append(loss.item())
            total += Y.shape[0]
            correct += wasCorrect(pred, Y, problemtype)
            if problemtype !=0: #not classifier
                error += getError(pred, Y, problemtype)
        avg_loss, avg_acc, avg_err = np.mean(losses), correct / total, error/total
        print(f"Test loss {avg_loss:.3f}, test acc {avg_acc:.3f} ", end = '')
        if problemtype !=0:
            print(f'test error {avg_err:.3f}', end='')
        print('')


    #raise NotImplementedError()



if __name__ == '__main__':

    if args.mode == 'train':
        train()
    # elif args.mode == 'test':
    #     bench = torch.load(benchfile)
    #     ckpt = torch.load(os.path.join(save_dir, 'model.tar'))
    #     net.load_state_dict(ckpt['state_dict'])
    #     test(bench)
    # elif args.mode == 'plot':
    #     ckpt = torch.load(os.path.join(save_dir, 'model.tar'))
    #     net.load_state_dict(ckpt['state_dict'])
    #     plot()
