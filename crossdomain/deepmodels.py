import numpy as np
from scipy import sparse as S
from scipy.sparse.csr import csr_matrix 

def sigmoid(x):
    #return x*(x > 0)
    #return numpy.tanh(x)
    return 1.0/(1+numpy.exp(-x)) 

def d_sigmoid(z):
    return np.multiply(z, 1 - z)

def identity(x):
    return x

def d_identity(x):
    return 1

def softmax(X):
    """
    numpy.array -> numpy.array
    Compute softmax function: exp(X) / sum(exp(X, 1))
    where each row of X is a vector output (e.g., different columns representing 
    outputs for different classes)
    The output of softmax is a matrix, with the sum of each row to be nearly 1.0
    as it is the probabilities that are calculated.
    """
    mx = np.max(X)
    ex = np.exp(X - mx) # prefer zeros over stack overflow - but NOT always useful
    return ex / np.sum(ex, 1).reshape(-1, 1)

def d_softmax(X):
    return np.multiply(X, 1 - X)

def bernoulli(p):
    return (np.random.rand(*p.shape) < p) * 1

derivative = { sigmoid : d_sigmoid,
               identity : d_identity, 
               softmax : d_softmax }

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
        self._Z = Z
        self._A = self.activation(Z.dot(self.W))
        return self._A

    def reset_weights(self):
        self.W = np.random.uniform(-.1, 0.1, self.W.shape) / np.sqrt(self.n_visible + self.n_hidden)


class Layer(BaseModel):
    """docstring for Layer"""
    def __init__(self, n_visible, n_hidden, activation = sigmoid, learning_rate = 0.01, momentum = 0.6, weight_decay = 0.0001):
        super(Layer, self).__init__(n_visible, n_hidden, activation)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._prev_W = np.zeros(self.W.shape)

    def backprop(self, E):
        self.delta = np.multiply(E, derivative[self.activation](self._A))
        self.delta_W = np.divide(np.dot(self._Z.transpose(), self.delta), E.shape[0])
        # self.delta_W = np.dot(self._Z.transpose(), self.delta)
        del self._Z
        del self._A
        self._prev_W = self.momentum * self._prev_W - self.learning_rate * (self.delta_W + self.weight_decay * self.W)
        self.W += self._prev_W
        return np.dot(self.delta, self.W[1:, :].transpose())

class NormalNoise(object):
    """docstring for NormalNoise"""
    def __init__(self, mean=0, sd=0.01):
        super(NormalNoise, self).__init__()
        self.mean = mean
        self.sd = sd
    def corrupt(self, X):
        return X + np.random.normal(self.mean, self.sd, X.shape)

class SaltAndPepper(object):
    """docstring for SaltAndPepper"""
    def __init__(self, p = 0.1):
        super(SaltAndPepper, self).__init__()
        self.p = p
    def corrupt(self, X):
        return X * bernoulli(np.random.uniform(0, 1, X.shape))

class Autoencoder(Layer):
    """
    docstring for Autoencoder
    """
    def __init__(self, n_visible, n_hidden, activation = sigmoid, decoder_transform = identity, learning_rate = 0.01, momentum = 0.6, weight_decay = 0.0001):
        super(Autoencoder, self).__init__(n_visible, n_hidden, activation, learning_rate, momentum, weight_decay)
        self.decoder_transform = decoder_transform
        self.decoder = Layer(n_hidden, n_visible, decoder_transform, learning_rate, momentum, weight_decay)
        self.errors = []

    def reconstruct(self, X, noise = False):
        if noise == False:
            return self.decoder.forward(self.forward(X))
        return self.decoder.forward(self.forward(noise.corrupt(X)))

    def encode(self, X):
        return self.forward(X)

    def fit_pretrain(self, X, batch_size = 2, epochs = 1000, validation = None, noise=False):

        for i in xrange(0, epochs):
            ix = np.random.randint(0, X.shape[0], batch_size)

            reconstructed = self.reconstruct(X[ix], noise)
            E = reconstructed - X[ix]
            _ = self.backprop(self.decoder.backprop(E))
            print "Epoch %s: error = %s" % (i, (np.mean(E**2)) * 0.5)  
            if validation is None:
                self.errors.append((np.mean(E**2)) * 0.5)
            else:
                self.errors.append(np.mean((self.reconstruct(validation) - validation) ** 2) * 0.5)




net = [Autoencoder(1061, 700, learning_rate=0.001, momentum=0.9, decoder_transform=identity),
       Autoencoder(700, 50, learning_rate=0.001, momentum=0.9, decoder_transform=sigmoid),
       Autoencoder(50, 24, learning_rate=0.001, momentum=0.9, decoder_transform=sigmoid)]
       

AE = Autoencoder(1061, 200, learning_rate=0.0001, momentum=0.99, decoder_transform=identity, weight_decay=0)

AE.fit_pretrain(Z[:5000], validation=Z[5000:7000], batch_size=10)

net = [AE,
       Autoencoder(2000, 50, learning_rate=0.001, momentum=0.9, decoder_transform=sigmoid),
       Autoencoder(50, 24, learning_rate=0.001, momentum=0.9, decoder_transform=sigmoid)]

R = net[0].encode(Z)

net[1].fit_pretrain(R[:5000], validation=R[5000:7000])

R = net[1].encode(R)

net[2].fit_pretrain(R[:1000], validation=R[1000:200])


net.append(Layer(24, 3, activation=softmax, learning_rate=0.01, momentum=0.9))



net = [Layer(200, 150, activation=sigmoid, learning_rate=0.001, momentum=0.9, weight_decay=0), 
       Layer(150, 70, activation=sigmoid, learning_rate=0.001, momentum=0.9, weight_decay=0),
       Layer(70, 24, activation=softmax, learning_rate=0.001, momentum=0.9, weight_decay=0)]

new_learning = 0.01
for L in net:
    L.learning_rate = new_learning



def predict(net, data):
    Z = data
    for L in net:
        Z = L.forward(Z)
    return Z



def train(net, data, target):
    Z = predict(net, data)
    E = Z - target
    for L in net[::-1]:
        E = L.backprop(E)
    return net



from sklearn import preprocessing

binarizer = preprocessing.LabelBinarizer()

binarizer.fit(labels)

T = binarizer.transform(labels)

# Z = scaler.fit_transform(X[:10000])


for i in xrange(0, 1000000):
    print i
    ix = np.random.randint(0, Z.shape[0], 20)
    net = train(net, Z[ix], T[ix])



ct = 0


for L in net:
    print 'Pretraining Layer {}'.format(ct + 1)
    ct += 1
    try:
        L.fit_pretrain(X[:120000], validation=X[120000:])
    except KeyboardInterrupt, k:
        continue



class RBM(BaseModel):
    """RBM inherited"""
    def __init__(self, n_visible, n_hidden, visible_unit = 'binary'):
        super(RBM, self).__init__(n_visible, n_hidden, sigmoid)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.visible_unit = visible_unit
        self.W = numpy.insert(self.W, 0, 0, axis = 1)
    def backward(self, H):
        if isinstance(H, S.csr_matrix):
            bias = S.csr_matrix(numpy.ones((H.shape[0], 1))) 
            B = S.hstack([bias, H]).tocsr()
        else:
            B = numpy.insert(H, 0, 1, 1)
        self._B = B
        if visible_unit == 'binary':
            return sigmoid(B.dot(self.W.T)) 
        elif visible_unit == 'gaussian':
            return B.dot(self.W.T)

    def sample_hidden(self, V):
        return bernoulli(forward(V))

    def sample_visible(self, H):
        if visible_unit == 'binary':
            return bernoulli(backward(H))
        else:











