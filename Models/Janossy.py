import torch
import torch.nn as nn
import itertools
import numpy as np





#Idea: this should be used as a trainable layer -> f must be trainable

#Important: this layer does not work on the gpu! the apply_() onyl works for cpu tensors!
class JanossyPool(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, janossy_k = 2):
        #in_dim: dim of input elements
        #out_dim: dim of output
        super().__init__()
        self.k = janossy_k
        self.f = nn.Sequential(nn.Linear(in_dim*janossy_k, h_dim),
            nn.ReLU(), nn.Linear(h_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, out_dim))
        self.permute = lambda a: itertools.permutations(a, self.k)


    def forward(self, x):
        #x has dim: (batchsize, n, in_dim)
        #xperm = torch.as_tensor(np.array([torch.as_tensor(list(self.permute(set))) for set in x]), dtype=float).requires_grad()
        xperm = torch.as_tensor([[np.array(x) for x in self.permute(set)] for set in x]).requires_grad()
        #xperm = x.apply_(self.permute) #cant use apply on tensor that requires grad
        xperm = torch.flatten(xperm, start_dim=2)
        return self.f(xperm).sum(dim=1)

        #major problem: all i try seems to require detaching from the gradient, but i need to get the gradient of every step
