import torch
import numpy as np
import os
import torchvision


#n_min = min numbr of pictures in sum
#n_max = max number of pictures in sum
class MNIST_sum_data(object):
    def __init__(self, batchsize=64, n_min=1, n_max=10):
        DATA_DIR = os.path.join(os.getcwd(), "Datasets\MNIST_data\\")
        train_raw= torchvision.datasets.MNIST(DATA_DIR, train=True, download=True)
        test_raw = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True)
        train_len = train_raw.__len__()
        test_len = test_raw.__len__()

        #try to have each image just once, not a problem if that isnt the case tho
        num_train_batches = round(train_len*2/(n_min+n_max)/batchsize)
        num_test_batches = round(test_len*2/(n_min+n_max)/batchsize)

        indexes = np.arange(0, train_raw.__len__())
        np.random.shuffle(indexes)

        self._train_data = []
        index = 0
        for _ in range(num_train_batches):
            n = np.random.randint(n_min, n_max+1) #n of batch
            X = np.zeros((batchsize, n, 784))
            Y = np.zeros(batchsize)
            for i in range(batchsize):
                for j in range(n):
                    img,label = train_raw.__getitem__(indexes[index])
                    img = np.asarray(img).reshape(784)

                    #img = img.reshape()
                    X[i,j,:]=img
                    Y[i] += label #Y will be sum of all y
                    index = (index+1)%train_len
            X,Y = np.expand_dims(X, axis=1), np.expand_dims(Y, axis=1)
            self._train_data.append((X,Y))

        self._test_data = []
        indexes = np.arange(0, test_len)
        np.random.shuffle(indexes)
        index = 0
        for _ in range(num_test_batches):
            n = np.random.randint(n_min, n_max+1) #n of batch
            X = np.zeros((batchsize, n, 784))
            Y = np.zeros(batchsize)
            for i in range(batchsize):
                for j in range(n):
                    img,label = test_raw.__getitem__(indexes[index])
                    img = np.asarray(img).reshape(784)

                    #img = img.reshape()
                    X[i,j,:]=img
                    Y[i] += label #Y will be sum of all y
                    index = (index+1)%test_len
            X, Y = np.expand_dims(X, axis=1), np.expand_dims(Y, axis=1)
            self._test_data.append((X,Y))


    def train_data(self):
        for X,Y in self._train_data:
            #shuffle the points in each element of X
            for array in X:
                np.random.shuffle(array)
            #Y stays the same
        return self._train_data
    def test_data(self):
        for X,Y in self._test_data:
            #shuffle the points in each element of X
            for array in X:
                np.random.shuffle(array)
            #Y stays the same
        return self._test_data
