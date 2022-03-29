import numpy as np
import math

class MaximumData(object):
    def __init__(self, batch_size=64, numberOfElements=20000, max_length = 40):
        self.batch_size = batch_size
        #each element in a batch must have same length
        num_train_batches = round(numberOfElements /batch_size)
        num_test_batches = num_train_batches

        self._train_data = []
        for i in range(num_train_batches):
            length = np.random.randint(1,max_length)
            X = np.random.randint(low=1,high=100, size=(batch_size, length))
            Y = np.max(X, axis=1)
            X, Y = np.expand_dims(X, axis=2), np.expand_dims(Y, axis=1)
            self._train_data.append((X,Y))

        self._test_data = []
        for i in range(num_test_batches):
            length = np.random.randint(1,max_length)
            X = np.random.randint(low=1,high=100, size=(batch_size, length))
            Y = np.max(X, axis=1)
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


class Max4Data(object):
    def __init__(self, batch_size=64, numberOfElements=10000, max_length = 40):
        self.batch_size = batch_size
        min_length = 4
        #each element in a batch must have same length
        num_train_batches = round(numberOfElements /batch_size)
        num_test_batches = num_train_batches

        self._train_data = []
        for i in range(num_train_batches):
            length = np.random.randint(min_length,max_length)
            X = np.random.randint(low=1,high=100, size=(batch_size, length))
            Y = np.sort(X, axis=1)[:,-1:-5:-1].copy()
            X= np.expand_dims(X, axis=2)
            self._train_data.append((X,Y))

        self._test_data = []
        for i in range(num_test_batches):
            length = np.random.randint(min_length,max_length)
            X = np.random.randint(low=1,high=100, size=(batch_size, length))
            Y = np.sort(X, axis=1)[:,-1:-5:-1].copy()
            X= np.expand_dims(X, axis=2)
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


class Min2Max2Data(object):
    def __init__(self, batch_size=64, numberOfElements=10000, max_length = 40):
        self.batch_size = batch_size
        min_length = 4
        #each element in a batch must have same length
        num_train_batches = round(numberOfElements /batch_size)
        num_test_batches = num_train_batches

        self._train_data = []
        for i in range(num_train_batches):
            length = np.random.randint(min_length,max_length)
            X = np.random.randint(low=1,high=100, size=(batch_size, length))
            Y = np.delete(np.sort(X, axis=1), slice(2,-2), axis = 1)
            # X,Y = np.expand_dims(X, axis=2), np.expand_dims(Y, axis=1)
            X= np.expand_dims(X, axis=2)
            self._train_data.append((X,Y))

        self._test_data = []
        for i in range(num_test_batches):
            length = np.random.randint(min_length,max_length)
            X = np.random.randint(low=1,high=100, size=(batch_size, length))
            Y = np.delete(np.sort(X, axis=1), slice(2,-2), axis = 1)
            # X,Y = np.expand_dims(X, axis=2), np.expand_dims(Y, axis=1)
            X= np.expand_dims(X, axis=2)
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

class MeanData(object):
    def __init__(self, batch_size=64, numberOfElements=20000, max_length = 40):
        self.batch_size = batch_size
        #each element in a batch must have same length
        num_train_batches = round(numberOfElements /batch_size)
        num_test_batches = num_train_batches

        self._train_data = []
        for i in range(num_train_batches):
            length = np.random.randint(1,max_length)
            X = np.random.randint(low=1,high=100, size=(batch_size, length))
            Y = np.mean(X, axis=1)
            X, Y = np.expand_dims(X, axis=2), np.expand_dims(Y, axis=1)
            self._train_data.append((X,Y))

        self._test_data = []
        for i in range(num_test_batches):
            length = np.random.randint(1,max_length)
            X = np.random.randint(low=1,high=100, size=(batch_size, length))
            Y = np.mean(X, axis=1)
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

class SumData(object):
    def __init__(self, batch_size=64, numberOfElements=20000, max_length = 40):
        self.batch_size = batch_size
        #each element in a batch must have same length
        num_train_batches = round(numberOfElements /batch_size)
        num_test_batches = num_train_batches

        self._train_data = []
        for i in range(num_train_batches):
            length = np.random.randint(1,max_length)
            X = np.random.randint(low=1,high=100, size=(batch_size, length))
            Y = np.sum(X, axis=1)
            X, Y = np.expand_dims(X, axis=2), np.expand_dims(Y, axis=1)
            self._train_data.append((X,Y))

        self._test_data = []
        for i in range(num_test_batches):
            length = np.random.randint(1,max_length)
            X = np.random.randint(low=1,high=100, size=(batch_size, length))
            Y = np.sum(X, axis=1)
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

#Tests:
# data = SumData(2,6,10)
# train_data = data.train_data()
# for X,Y in train_data:
#     print("hello")
#     print(np.array2string(X),Y)


#does it work the way it should?

# print(data.train_data())
# print(data.train_data())
