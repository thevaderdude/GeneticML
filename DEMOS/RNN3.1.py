import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import time

np.random.seed(42)

fmins = []
fmaxs = []


# create stock dataset
# inputs should be mem size length and targets should be the same but shifted up one
def create_dataset(stock, mem_size):

    data = pd.read_csv(stock)
    prices = data.iloc[:, 4]
    prices = np.array(prices)
    inputs, targets = np.zeros((data.shape[0], mem_size, 6)), []
    '''
    omax, omin, = np.max(prices), np.min(prices)
    min = omin - .1 * (omax - omin)
    max = omax + .1 * (omax - omin)
    prices = (prices - min) / (max - min)
    '''
    for i in range(data.shape[0]-(mem_size+1)):
        # targets.append(np.array(prices[i + 1:i + mem_size + 1]))
        ie = np.array(prices[i:i+mem_size])
        # normalization
        omin = np.min(ie)
        omax = np.max(ie)
        changes1 = [0]
        changes = []
        # targets[i] = (targets[i] - omin) / (omax - omin)
        for g in range(mem_size):
            ins1 = []
            ins1.append((prices[i+g] - omin) / (omax - omin))
            change = (prices[i+g+1] - prices[i+g]) / prices[i+g]
            changes1.append(change)
            changes.append(change)
            for j in range(7):
                if not (j == 0 or j == 4 or j == 5):
                    ins = data.iloc[i+g, j]
                    ins2 = data.iloc[i:i+mem_size, j]
                    emin = np.min(ins2)
                    emax = np.max(ins2)
                    ins1.append((ins - emin) / (emax - emin + 1e-12))
            ins1.append(changes1[g-1])
            for h in range(6):
                inputs[i, g, h] = ins1[h]
        fmin = np.min(changes)
        fmax = np.max(changes)
        fmins.append(fmin)
        fmaxs.append(fmax)
        targets.append((changes - fmin) / (fmax - fmin))
    inputs, targets = np.array(inputs), np.array(targets)
    return inputs, targets


def de_normalize_outs(par, it):
    return par * (fmaxs[it] - fmins[it]) + fmins[it]


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


def init_rnn(hidden_size):
    # p[0]
    U = np.zeros((hidden_size, 6))
    # w
    V = np.zeros((hidden_size, hidden_size))
    # p[1]
    W = np.zeros((1, hidden_size))

    U = init_mat(U)
    V = init_mat(V)
    W = init_mat(W)
    b_hidden = np.random.randn(hidden_size, 1)
    b_out = np.random.randn(1, 1)
    return U, V, W, b_hidden, b_out


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


def relu(par, derivative=False):
    if derivative:
        new = np.zeros_like(par)
        for i in range(len(par)):
            if par[i] > 0:
                new[i] = 1
            else:
                new[i] = 0
        return new
    else:
        new = np.copy(par)
        for i in range(len(par)):
            if new[i] < 0:
                new[i] = 0
        return new


# h
def forward_pass(inputs, hidden_state, p):
    U, V, W, b_hidden, b_out = p
    outputs, hidden_states = [], []
    for t in range(len(inputs)):
        hidden_state = relu(np.dot(U, inputs[t]).reshape(100, 1) + np.dot(V, hidden_state).reshape(100, 1) + b_hidden)

        out = sigmoid(np.dot(W, hidden_state) + b_out)
        outputs.append(out)
        hidden_states.append(hidden_state.copy())

    return outputs, hidden_states


# ok like this is fine but do I really need this
def clip_gradient_norm(grads, max_norm=0.25):
    max_norm = float(max_norm)
    total_norm = 0

    for grad in grads:
        grad_norm = np.sum(np.power(grad, 2))
        total_norm += grad_norm

    total_norm = np.sqrt(total_norm)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in grads:
            grad *= clip_coef

    return grads


def backward_pass(inputs, outputs, hidden_states, targets, params):
    U, V, W, b_hidden, b_out = params
    d_U, d_V, d_W = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)
    d_b_hidden, d_b_out = np.zeros_like(b_hidden), np.zeros_like(b_out)

    d_h_next = np.zeros_like(hidden_states[0])
    loss = 0

    for t in reversed(range(len(outputs))):
        # loss += -np.mean(np.log(outputs[t]+1e-12) * targets[t])
        loss += ((outputs[t] - targets[t])**2)
        # 1x1
        d_o = outputs[t].copy()
        d_o -= targets[t]
        # 1x100
        d_W += np.dot(d_o.reshape(-1, 1), hidden_states[t].reshape(1, -1))
        d_b_out += d_o
        # 100x1
        d_h = np.dot(W.T, d_o) + d_h_next
        # 100x1
        d_f = relu(hidden_states[t], derivative=True) * d_h
        d_b_hidden += d_f
        # 100x2
        d_U += np.dot(d_f.reshape(-1, 1), inputs[t].reshape(1, -1))
        # 100x100
        d_V += np.dot(d_f, hidden_states[t-1].T)
        # 100x1
        d_h_next = np.dot(V.T, d_f)

    grads = d_U, d_V, d_W, d_b_hidden, d_b_out
    # grads = clip_gradient_norm(grads)
    return loss / len(outputs), grads


def update_pars(params, grads, lr):
    # print(grads)
    for param, grad in zip(params, grads):
        param -= lr * grad

    return params


def train(epochs, lr, x, y):
    global p
    global hidden_state
    for i in range(epochs):
        loss0 = 0
        start = time.time()
        for inputs, targets in zip(x, y):
            hidden_state = np.zeros_like(hidden_state)

            outputs, hidden_states = forward_pass(inputs, hidden_state, p)
            loss, grads = backward_pass(inputs, outputs, hidden_states, targets, p)

            p = update_pars(p, grads, lr)
            # print(p)
            loss0 += loss
        end = time.time()
        print(end-start)

        print(f'Epoch {i}: {loss0/len(x)}')


def save_p(params):
    pd.DataFrame(params).to_csv('p1.csv')


# BIAS BIAS BIAS BIAS BIAS

# create dataset

X, Y = create_dataset('AMD.csv', 20)

# init network
hidden_size = 100
p = init_rnn(hidden_size)
hidden_state = np.zeros((hidden_size, 1))

# test to make sure the network actually works (also to show how the rnn works)
'''
test_input, test_target = X[0], Y[0]
outputs, hidden_states = forward_pass(test_input, hidden_state, p)
print(outputs)
print([idx_to_word[np.argmax(test_inputs)] for test_inputs in test_input])
print([idx_to_word[np.argmax(test_targets)] for test_targets in test_target])
print([idx_to_word[np.argmax(output)] for output in outputs])
loss, _ = backward_pass(X[0], outputs, hidden_states, Y[0], p)
print(loss)
'''
test_input, test_target = X[9], Y[9]
outputs, hidden_states = forward_pass(test_input, hidden_state, p)
print(test_input)
print(test_target)
print(outputs)
hidden_state = np.zeros((hidden_size, 1))
train(1600, 2e-4, X[:600], Y[:600])

for i in range(20):
    hidden_state = np.zeros((hidden_size, 1))
    test_input, test_target = X[120+i], Y[120+i]
    outputs, hidden_states = forward_pass(test_input, hidden_state, p)
    # print(test_input)
    # print(test_target)
    # print(np.array(outputs).reshape(20))
    plt.plot(de_normalize_outs(test_target, 120+i))
    plt.plot(de_normalize_outs(np.array(outputs).reshape(20), 120+i))
    plt.plot(np.zeros(20))
    plt.show()
hidden_state = np.zeros((hidden_size, 1))
correct = 0
for i in range(600):
    test_input, test_target = X[i], Y[i]
    outputs, hidden_states = forward_pass(test_input, hidden_state, p)
    # print(test_input)
    targ = test_target
    out = np.array(outputs).reshape(20)
    if de_normalize_outs(targ[20-1], i) > 0 and de_normalize_outs(out[20-1], i) > 0:
        correct += 1
    elif de_normalize_outs(targ[20-1], i) < 0 and de_normalize_outs(out[20-1], i) < 0:
        correct += 1

print(correct)
save_p(p)


#
# Possibilities:
# Use more information from previous day's price action: high, low, vol, etc.
# use relu activation function for outputs ???
