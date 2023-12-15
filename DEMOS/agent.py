from tokenize import endpats
import numpy as np
from RNN import RNN

class Agent:

    def __init__(self, start_pos, end_pos, num_nodes):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.pos = self.start_pos
        self.is_alive = True
        self.num_moves = 0
        self.dist_to_goal = distance(self.pos, self.end_pos)
        # maybe adjust these parameters later
        self.rnn = RNN(11, 4, num_nodes)
    
    # move based on window and end_pos vector
    def move(self, window, end_pos_vector):
        if self.is_alive and end_pos_vector != (0, 0):
            self.num_moves += 1

            # RNN decision-making
            # clean input data
            window_flat = window.flatten()
            end_pos_vector = np.array(end_pos_vector)
            # print(window_flat.shape)
            out = self.rnn.forward_pass(np.concatenate((window_flat, end_pos_vector)))
            # order: u d l r
            max = np.argmax(out)
            # print(max)
            move_vector = (0, 0)
            if max == 0:
                move_vector = (-1, 0)
            if max == 1:
                move_vector = (1, 0)
            if max == 2:
                move_vector = (0, -1)
            if max == 3:
                move_vector = (0, 1)
            
            self.pos = (self.pos[0]+move_vector[0], self.pos[1]+move_vector[1])
            # print(self.pos)
            if window[1+move_vector[0], 1+move_vector[1]] == 1:
                # print('dead')
                self.is_alive = False
            self.dist_to_goal = distance(move_vector, end_pos_vector)
        return self.pos
    
    def reset(self):
        self.is_alive = True
        self.pos = self.start_pos
        self.num_moves = 0
        self.dist_to_goal = distance(self.pos, self.end_pos)
    

# distance formula
def distance(x, y):
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
        
        
        