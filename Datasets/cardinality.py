import numpy as np
import math




class CardinalityData(object):
    def __init__(self, batch_size=64, numberOfElements=20000, max_length = 40):
        self.batch_size = batch_size
        #each element in a batch must have same length
        num_train_batches = round(numberOfElements /batch_size)
        num_test_batches = round(num_train_batches)

        self._train_data = []
        for i in range(num_train_batches):
            length = np.random.randint(5,max_length)
            X = np.random.randint(low=1,high=100, size=(batch_size, length))
            Y = np.ones(batch_size, dtype=int)*length
            X, Y = np.expand_dims(X, axis=2), np.expand_dims(Y, axis=1)
            self._train_data.append((X,Y))

        self._test_data = []
        for i in range(num_test_batches):
            length = np.random.randint(5,max_length)
            X = np.random.randint(low=1,high=100, size=(batch_size, length))
            Y = np.ones(batch_size, dtype=int)*length
            X, Y = np.expand_dims(X, axis=2), np.expand_dims(Y, axis=1)
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
