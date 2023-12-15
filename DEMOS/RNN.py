import numpy as np


class RNN:

    def __init__(self, in_size, out_size, hidden_layer_size):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_layer_size = hidden_layer_size

        self.U = np.zeros((self.hidden_layer_size, self.in_size))
        self.V = np.zeros((self.hidden_layer_size, self.hidden_layer_size))
        self.W = np.zeros((self.out_size, self.hidden_layer_size))

        self.U = init_mat(self.U)
        self.V = init_mat(self.V)
        self.W = init_mat(self.W)
        self.b_hidden = np.random.randn(self.hidden_layer_size, 1)
        self.b_out = np.random.randn(self.out_size, 1)
        self.hidden_state = np.zeros((self.hidden_layer_size, 1))
    
    # perform a single forward pass operation
    def forward_pass(self, inputs):
        hidden_s = relu(np.dot(self.U, inputs).reshape(self.hidden_layer_size, 1) + np.dot(self.V, self.hidden_state).reshape(self.hidden_layer_size, 1) + self.b_hidden)
        # print(hidden_s.shape)
        out = sigmoid(np.dot(self.W, hidden_s) + self.b_out)
        self.hidden_state = hidden_s
        return out

def init_mat(par):
    rows, cols = par.shape
    new_par = np.random.randn(rows, cols)
    if rows < cols:
        new_par = new_par.T
    q, r = np.linalg.qr(new_par)
    d = np.diag(r, 0)
    ph = np.sign(d)
    q *= ph
    if rows < cols:
        q = q.T
    new_par = q
    return new_par


def relu(par, derivative=False):
    if derivative:
        new = np.zeros_like(par)
        new[new > 0] = 1
    else:
        new = np.copy(par)
        new[new < 0] = 0
    return new
        


def sigmoid(par, derivative=False):
    par += 1e-12
    f = 1 / (1 + np.exp(-par))
    if derivative:
        return f * (1 - f)
    else:
        return f


def softmax(par, derivative=False):
    par += 1e-12
    f = np.exp(par) / np.sum(np.exp(par))
    if derivative:
        pass
    else:
        return f


def tanh(par, derivative=False):
    par += 1e-12
    f = (np.exp(par)-np.exp(-par)) / (np.exp(par)+np.exp(-par))
    if derivative:
        return 1-f**2
    else:
        return f
