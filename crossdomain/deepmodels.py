import numpy as np
from scipy import sparse as S
from scipy.sparse.csr import csr_matrix 

def sigmoid(x):
    #return x*(x > 0)
    #return numpy.tanh(x)
    return 1.0/(1+numpy.exp(-x)) 

def identity(x):
    return x

def bernoulli(p):
    return (np.random.rand(*p.shape) < p) * 1

class BaseModel(object):
    """docstring for Base class for building layer components"""
    def __init__(self, n_visible, n_hidden, activation = identity):
        super(BaseModel, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = np.random.uniform(-.1, 0.1, (n_visible,  n_hidden)) / np.sqrt(n_visible + n_hidden)
        self.W = numpy.insert(self.W, 0, 0, axis = 0)
        self.activation = activation

    def forward(self, X):
        if isinstance(X, S.csr_matrix):
            bias = S.csr_matrix(numpy.ones((X.shape[0], 1)))
            Z = S.hstack([bias, X]).tocsr()
        else:
            Z = numpy.insert(X, 0, 1, 1)
        return self.activation(Z.dot(self.W))

class RBM(BaseModel):
    """RBM"""
    def __init__(self, n_visible, n_hidden, visible_unit = 'binary'):
        super(RBM, self).__init__(n_visible, n_hidden, sigmoid)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.visible_unit = visible_unit
        self.W = numpy.insert(self.W, 0, 0, axis = 1)
    def backward(self, H):
        if isinstance(H, S.csr_matrix):
            bias = S.csr_matrix(numpy.ones((H.shape[0], 1))) 
            csr = S.hstack([bias, H]).tocsr()
        else:
            csr = numpy.insert(H, 0, 1, 1)









