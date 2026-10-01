import numpy as np

'''
Python numpy implmentation of the Long Short Term Memeory (LSTM) network.

NOTES:
-- h = hidden state, also known as the short term memory
-- c = cell state, also known as the long term memory
'''

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        ## Weight matrices
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size)
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size)
        self.W_c = np.random.randn(hidden_size, input_size + hidden_size)
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size)

        self.b_f = np.zeros((hidden_size, 1))
        self.b_i = np.zeros((hidden_size, 1))
        self.b_c = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))


    def forward(self, x, h_prev, c_prev):
        concat = np.vstack((h_prev, x))

        f = sigmoid(np.dot(self.W_f, concat) + self.b_f)
        i = sigmoid(np.dot(self.W_i, concat) + self.b_i)
        c_bar = tanh(np.dot(self.W_c, concat) + self.b_c)
        c = f * c_prev + i * c_bar
        o = sigmoid(np.dot(self.W_o, concat) + self.b_o)
        h = o * tanh(c)

        return h, c

    def backward(self, dh_next, dc_next, cache, loss):
        dh, dc = dh_next, dc_next
        
    