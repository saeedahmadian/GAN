import numpy as np


class CyberData(object):

    def __init__(self, config):

        self.num_samples = config.num_samples
        self.num_features = config.num_features
        self.num_nodes = config.num_nodes
        mean = 100
        std = 20
        mydata=np.load('traindata.npy')

        # create array for data
        self.X = np.zeros((self.num_samples, self.num_nodes, self.num_features, 1), dtype='float32')
        self.X=mydata

        # generate user data from Gaussian distribution
        #for n in range(self.num_samples):
            #for i in range(self.num_nodes):
                #self.X[n, i, :, 0] = np.random.normal(loc=mean, scale=std, size=self.num_features)

        self.y = np.zeros((self.num_samples, 1)).astype('float32')

    def next_batch(self, batch_size):
        """ get next batch of samples"""

        for i in range(0, self.num_samples, batch_size):
            yield self.X[i: i + batch_size, :, :], self.y[i: i + batch_size]

    def randomize(self):
        """ Randomizes the order of data samples and their corresponding labels"""
        permutation = np.random.permutation(self.X.shape[0])
        self.X = self.X[permutation, :, :, :]
        self.y = self.y[permutation, :]

if __name__ == '__main__':
    data = CyberData()