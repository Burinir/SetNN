import numpy as np
import math


#this file should contain a few datasets based on clustering of the data.
#Ideas: eq2, how many points needed for every elemnt to be within 5 of a point, normal distribution parameters

class Eq2Data(object):  #Are there two equal objects in the set? yes n0
    def __init__(self, batch_size=64, numberOfElements=20000, max_length = 40):
        self.batch_size = batch_size
        #each element in a batch must have same length
        num_train_batches = round(numberOfElements /batch_size)
        num_test_batches = num_train_batches

        self._train_data = []
        for i in range(num_train_batches):
            length = np.random.randint(1,max_length)
            X = np.random.randint(low=1,high=100, size=(batch_size, length))
            Y = contains2Equal(X)
            X = np.expand_dims(X, axis=2)
            self._train_data.append((X,Y))

        self._test_data = []
        for i in range(num_test_batches):
            length = np.random.randint(1,max_length)
            X = np.random.randint(low=1,high=100, size=(batch_size, length))
            Y = contains2Equal(X)
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

    #for a binary classifier it is important to know how good it is to just always guess the more common label
    def avgY(dataset):
        batchsize = dataset._train_data[0][1].shape[0]
        sum = 0
        amount = 0
        for X,Y in dataset._train_data:
            sum += Y.sum()
            amount += batchsize
        trainavg = sum / amount
        amount=0
        sum = 0
        for X,Y in dataset._test_data:
            sum += Y.sum()
            amount += batchsize
        testavg = sum / amount
        print(f'Train data: {trainavg:.3f} 0\'s \nTest data: {testavg:.3f} 0\'s')



def contains2Equal(batch):
    batchlength = len(batch[:,0])
    setlength = len(batch[0,:])
    Y = np.zeros(batchlength)
    for i, set in enumerate(batch):
        for shift in range(1,setlength):
            if (np.roll(set, shift) ==set).sum() > 0:
                Y[i]=1
                break
    return Y








#Tests:
# data = Eq2Data(2,6,10)
# train_data = data.train_data()
# for X,Y in train_data:
#     print("hello")
#     print(np.array2string(X),Y)
