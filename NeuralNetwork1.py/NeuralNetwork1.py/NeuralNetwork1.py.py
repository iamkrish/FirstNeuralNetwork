import numpy as np

class Neural_Network(object):
    def __init__(self):
        self.inputLayerSize=2
        self.outputLayerSize=1
        self.hiddenLayerSize=3

        # Initialize Weights
        self.W1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize)

    def forward_propogation(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)

        self.z3 =np.dot(self.a2, self.W2)
        yhat = self.sigmoid(self.z3)

        return yhat


    def sigmoid(self, z):
        return 1/(1+np.exp(-z))


def prepare_data():
    X = np.array(([3,5], [5,1], [10,2]), dtype=float)
    y = np.array(([75], [82], [93]), dtype=float)

    # Normalize
    Xnormal = X/np.amax(X,axis=0)
    Ynormal = y/100
    return X,y

NN = Neural_Network()
X,y = prepare_data()
yhat = NN.forward_propogation(X)
print(yhat)