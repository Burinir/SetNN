import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import h5py
import pdb
from tqdm import tqdm, trange
import Models.FSPool
import Models.Janossy


# @incollection{deepsets,
# title = {Deep Sets},
# author = {Zaheer, Manzil and Kottur, Satwik and Ravanbakhsh, Siamak and Poczos, Barnabas and Salakhutdinov, Ruslan R and Smola, Alexander J},
# booktitle = {Advances in Neural Information Processing Systems 30},
# editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
# pages = {3391--3401},
# year = {2017},
# publisher = {Curran Associates, Inc.},
# url = {http://papers.nips.cc/paper/6931-deep-sets.pdf}
# }


class SmallDeepSet(nn.Module):
    def __init__(self, pool="max"):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
        )

        self.pool = None
        self.fspool = None
        if pool == 'max':
            self.pool = lambda a: torch.max(a, dim = 1)[0]
        if pool=='mean':
            self.pool = lambda a: torch.mean(a, dim=1)
        elif pool =='fspool':
            self.fspool =  Models.FSPool.FSPool(in_channels=64, n_pieces=100)
            self.pool =  lambda a: self.fspool(torch.transpose(a, 1,2))[0]
        elif pool == 'sum':
            self.pool = lambda a: torch.sum(a, dim=1)
        #self.pool = Models.Janossy.JanossyPool(in_dim=64, h_dim=64, out_dim=64, janossy_k=2)
        #self.pool = pool

    def forward(self, x):
        x = self.enc(x)
        x = self.pool(x)
        x = self.dec(x)
        return x

class BinaryDeepSet(nn.Module):
    def __init__(self, dim_input, dim_hidden, pool = 'sum'):
        super().__init__()
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, 2),
                nn.Softmax(dim=1))
        self.pool = None
        self.fspool = None
        if pool == 'max':
            self.pool = lambda a: torch.max(a, dim = 1)[0]
        if pool=='mean':
            self.pool = lambda a: torch.mean(a, dim=1)
        elif pool =='fspool':
            self.fspool =  Models.FSPool.FSPool(in_channels=64, n_pieces=100)
            self.pool =  lambda a: self.fspool(torch.transpose(a, 1,2))[0]
        elif pool == 'sum':
            self.pool = lambda a: torch.sum(a, dim=1)
    def forward(self, x):
        x = self.enc(x)
        x = self.pool(x)
        x = self.dec(x)
        return x


#used for max4, min2max2
class Max4DeepSet(nn.Module):
    def __init__(self, pool="sum"):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=4),
        )
        self.pool = None
        self.fspool = None
        if pool == 'max':
            self.pool = lambda a: torch.max(a, dim = 1)[0]
        if pool=='mean':
            self.pool = lambda a: torch.mean(a, dim=1)
        elif pool =='fspool':
            self.fspool =  Models.FSPool.FSPool(in_channels=64, n_pieces=100)
            self.pool =  lambda a: self.fspool(torch.transpose(a, 1,2))[0]
        elif pool == 'sum':
            self.pool = lambda a: torch.sum(a, dim=1)

    def forward(self, x):
        x = self.enc(x)
        x = self.pool(x)
        x = self.dec(x)
        return x

class MNISTsumDeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_hidden=128):
        super(MNISTsumDeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs)
        #print(X)
        return X


class DeepSetPointCloud(nn.Module):

  def __init__(self, d_dim, x_dim=3, equipool = 'mean', pool='mean'):
    super(DeepSetPointCloud, self).__init__()
    self.d_dim = d_dim
    self.x_dim = x_dim

    self.pool = None
    if pool == 'max':
        self.pool = lambda a: torch.max(a, dim = 1)[0]
    if pool=='mean':
        self.pool = lambda a: torch.mean(a, dim=1)
    elif pool =='fspool':
        self.fspool =  Models.FSPool.FSPool(in_channels=d_dim, n_pieces=100)
        self.pool =  lambda a: self.fspool(a)[0]
    elif pool == 'sum':
        self.pool = lambda a: nn.sum(a, dim=1)

    if equipool == 'max':
        self.phi = nn.Sequential(
          PermEqui2_max(self.x_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
        )
    elif equipool == 'max1':
        self.phi = nn.Sequential(
          PermEqui1_max(self.x_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
        )
    elif equipool == 'mean':
        self.phi = nn.Sequential(
          PermEqui2_mean(self.x_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui2_mean(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui2_mean(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
        )
    elif equipool == 'mean1':
        self.phi = nn.Sequential(
          PermEqui1_mean(self.x_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui1_mean(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
          PermEqui1_mean(self.d_dim, self.d_dim),
          nn.ELU(inplace=True),
        )

    self.ro = nn.Sequential(
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, self.d_dim),
       nn.ELU(inplace=True),
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, 40),
    )
    print(self)

  def forward(self, x):
    phi_output = self.phi(x)
    sum_output = phi_output.mean(1)
    ro_output = self.ro(sum_output)
    return ro_output


class DeepSetPointCloudTanh(nn.Module):

  def __init__(self, d_dim, x_dim=3, pool = 'mean'):
    super(DTanh, self).__init__()
    self.d_dim = d_dim
    self.x_dim = x_dim

    if pool == 'max':
        self.phi = nn.Sequential(
          PermEqui2_max(self.x_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_max(self.d_dim, self.d_dim),
          nn.Tanh(),
        )
    elif pool == 'max1':
        self.phi = nn.Sequential(
          PermEqui1_max(self.x_dim, self.d_dim),
          nn.Tanh(),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui1_max(self.d_dim, self.d_dim),
          nn.Tanh(),
        )
    elif pool == 'mean':
        self.phi = nn.Sequential(
          PermEqui2_mean(self.x_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_mean(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui2_mean(self.d_dim, self.d_dim),
          nn.Tanh(),
        )
    elif pool == 'mean1':
        self.phi = nn.Sequential(
          PermEqui1_mean(self.x_dim, self.d_dim),
          nn.Tanh(),
          PermEqui1_mean(self.d_dim, self.d_dim),
          nn.Tanh(),
          PermEqui1_mean(self.d_dim, self.d_dim),
          nn.Tanh(),
        )

    self.ro = nn.Sequential(
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, self.d_dim),
       nn.Tanh(),
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, 40),
    )
    print(self)

  def forward(self, x):
    phi_output = self.phi(x)
    sum_output, _ = phi_output.max(1)
    ro_output = self.ro(sum_output)
    return ro_output



class PermEqui1_max(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui1_max, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    xm, _ = x.max(1, keepdim=True)
    x = self.Gamma(x-xm)
    return x

class PermEqui1_mean(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui1_mean, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    xm = x.mean(1, keepdim=True)
    x = self.Gamma(x-xm)
    return x

class PermEqui2_max(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui2_max, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

  def forward(self, x):
    xm, _ = x.max(1, keepdim=True)
    xm = self.Lambda(xm)
    x = self.Gamma(x)
    x = x - xm
    return x

class PermEqui2_mean(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui2_mean, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

  def forward(self, x):
    xm = x.mean(1, keepdim=True)
    xm = self.Lambda(xm)
    x = self.Gamma(x)
    x = x - xm
    return x
