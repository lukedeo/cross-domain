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
    def finalize(self):
        self._W = self.W.copy()



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
    def __init__(self, n_visible=None, n_hidden=None, W=None, learning_rate = 0.1, weight_decay=0.01,cd_steps=1,momentum=0.5, activation = sigmoid):
        super(RBM, self).__init__(n_visible, n_hidden, activation)
        if W == None:
            self.W =  numpy.random.uniform(-.1,0.1,(n_visible,  n_hidden)) / numpy.sqrt(n_visible + n_hidden)
            self.W = numpy.insert(self.W, 0, 0, axis = 1)
            self.W = numpy.insert(self.W, 0, 0, axis = 0)
        else:
            self.W=W 
        self.activation = activation
        self.learning_rate = learning_rate 
        self.momentum = momentum
        self.last_change = 0
        self.last_update = 0
        self.cd_steps = cd_steps
        self.epoch = 0 
        self.weight_decay = weight_decay  
        self.errors = []
        self.n_visible = n_visible
        self.n_hidden = n_hidden
   
    def to_layer(self):
        L = Layer(self.n_visible, self.n_hidden, self.activation, self.learning_rate, self.momentum, self.weight_decay)
        L.W = self.W[:, 1:]
        return L

    def fit_pretrain(self, Input, max_epochs = 1, batch_size=100, holdout = None):  
        if isinstance(Input, S.csr_matrix):
            bias = S.csr_matrix(numpy.ones((Input.shape[0], 1))) 
            csr = S.hstack([bias, Input]).tocsr()
        else:
            csr = numpy.insert(Input, 0, 1, 1)
        for epoch in range(max_epochs): 
            idx = np.random.randint(0, csr.shape[0], batch_size)
            self.V_state = csr[idx] 
            self.H_state = self.activate(self.V_state)
            pos_associations = self.V_state.T.dot(self.H_state) 
  
            for i in range(self.cd_steps):
              self.V_state = self.sample(self.H_state)  
              self.H_state = self.activate(self.V_state)
              
            neg_associations = self.V_state.T.dot(self.H_state) 
            self.V_state = self.sample(self.H_state) 
            
            # Update weights. 
            w_update = self.learning_rate * (((pos_associations - neg_associations) / batch_size) - self.weight_decay * self.W)
            total_change = numpy.sum(numpy.abs(w_update)) 
            self.W += self.momentum * self.last_change  + w_update
            # self.W *= self.weight_decay 
            
            self.last_change = w_update
            if holdout is None:
                RMSE = numpy.mean((csr[idx] - self.V_state)**2)**0.5
            else:
                if isinstance(holdout, S.csr_matrix):
                    bias = S.csr_matrix(numpy.ones((holdout.shape[0], 1))) 
                    h_out = S.hstack([bias, holdout]).tocsr()
                else:
                    h_out = numpy.insert(holdout, 0, 1, 1)
                RMSE = numpy.mean((h_out - self.sample(self.activate(h_out)))**2)**0.5
            self.errors.append(RMSE)
            self.epoch += 1
            print "Epoch %s: RMSE = %s; ||W||: %6.1f; Sum Update: %f" % (self.epoch, RMSE, numpy.sum(numpy.abs(self.W)), total_change)  
        return self 
        
    def learning_curve(self):
        plt.ion()
        #plt.figure()
        plt.show()
        E = numpy.array(self.errors)
        plt.plot(pandas.rolling_mean(E, 50)[50:])  
     
    def activate(self, X):
        if X.shape[1] != self.W.shape[0]:
            if isinstance(X, S.csr_matrix):
                bias = S.csr_matrix(numpy.ones((X.shape[0], 1))) 
                csr = S.hstack([bias, X]).tocsr()
            else:
                csr = numpy.insert(X, 0, 1, 1) 
        else:
            csr = X
        p = sigmoid(csr.dot(self.W)) 
        p[:,0]  = 1.0 
        return p  
        
    def sample(self, H, addBias=True): 
        if H.shape[1] == self.W.shape[0]:
            if isinstance(H, S.csr_matrix):
                bias = S.csr_matrix(numpy.ones((H.shape[0], 1))) 
                csr = S.hstack([bias, H]).tocsr()
            else:
                csr = numpy.insert(H, 0, 1, 1)
        else:
            csr = H
        p = sigmoid(csr.dot(self.W.T)) 
        p[:,0] = 1
        return p

    # def finalize(self):
        # self._W = self.W.copy()
        # self.W = self.W[:, 1:]









