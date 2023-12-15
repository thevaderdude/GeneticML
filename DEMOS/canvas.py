import numpy as np
from agent import Agent
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Canvas:
    
    def __init__(self, num_agents, num_nodes, canvas=None, start_pos=None, end_pos=None, graph=True):  
        # initialize default canvas
        if canvas is None:
            # create canvas: array of booleans with padding
            canvas_shape = (10, 10)
            self.canvas = np.zeros((canvas_shape[0]+2, canvas_shape[1]+2), dtype=int)
            # add padding
            self.canvas[0] = 1
            self.canvas[:, 0] = 1
            self.canvas[-1] = 1
            self.canvas[:, -1] = 1
            # add obstacle: 8x8 box in the middle
            self.canvas[3:-3, 3:-3] = 1
            # set start and end positions
            start_pos = (10, 1)
            end_pos = (1, 10)
            self.start_pos = start_pos
            self.end_pos = end_pos

        self.graph = graph
        # create colored array for canvas
        self.canvas_display = np.zeros((self.canvas.shape[0], self.canvas.shape[1], 3))
        for i, _ in np.ndenumerate(self.canvas):
            # if not barrier
            if self.canvas[i] == 0:
                self.canvas_display[i[0], i[1]] = 1
        # add start pos and end pos colors
        self.canvas_display[self.start_pos[0], self.start_pos[1]] = [1, 0, 0]
        self.canvas_display[self.end_pos[0], self.end_pos[1]] = [0, 1, 0]
        # print(self.canvas_display[self.start_pos[0], self.start_pos[1]])

        # create dict to store data 
        self.data = {
            'gen_nums': [],
            'max_fit': [],
            'alive_prop': [],
            'min_dist': [],
            'max_moves': [],
            'agent_moves': []
        }
        
        # initialize a ton of agents
        self.agents = [Agent(self.start_pos, self.end_pos, num_nodes) for i in range(num_agents)]

    # move agents based on what they want
    def move_agents(self):
        # just do move function for each agnet
        agents_x, agents_y = [], []
        for agent in self.agents:
            # only do this if they are alive else risk and index out of bounds error
            # print(agent.is_alive)
            if agent.is_alive:
                # get window and end_pos vector
                window = self.canvas[agent.pos[0]-1:agent.pos[0]+2, agent.pos[1]-1:agent.pos[1]+2]
                end_pos_vector = (self.end_pos[0]-agent.pos[0], self.end_pos[1]-agent.pos[1])
                # print(window.shape)
                # print([agent.pos[0]-1, agent.pos[0]+2, agent.pos[1]-1, agent.pos[1]+2])
                x, y = agent.move(window, end_pos_vector)
                agents_x.append(int(x))
                agents_y.append(int(y))
        return agents_x, agents_y

    # do one selection, take in amount of moves allowed and output generation stats and best few agents stats  
    def genetic_iter(self, generation, moves_allowed=100, sel_prop=0.3, mutate_proba=0.01, mutate_range=0.3):
        # add generation num to data:
        self.data['gen_nums'] += [int(generation)]
        
        # do the moves

        # scatter data for agents
        a_x, a_y = [], []
        for i in range(moves_allowed):
            # print(i)
            agents_x, agents_y = self.move_agents()
            a_x.append(agents_x)
            a_y.append(agents_y)
        # print(a_x)

        # animate moves
        if self.graph:
            fig = plt.figure()
            ax = plt.axes()
            frame = 0
            scatter = ax.scatter(a_x[0], a_y[0])
            plt.imshow(self.canvas_display)
            def update_fig(frame):
                if frame < 20:
                    frame += 1
                else:
                    frame = 0
                    # plt.close()

                data = np.vstack((a_x[frame], a_y[frame])).T
                # print(data)
                scatter.set_offsets(data)
                return scatter,
            anim = animation.FuncAnimation(fig, update_fig)
            writergif = animation.PillowWriter(fps=5)

            anim.save(f'Animations/1/anim_{generation}.gif', writer=writergif)
            plt.title(f'Generation: {generation}')
            plt.show()
        #plt.close()
        

        # find best agents
        best_list = []
        fits = self.fitness()
        for i in range(int(sel_prop*len(self.agents))):
            max = np.argmax(fits)
            # if its top agent
            if i == 0:
                # get moves of best agent
                # each move is a an array with subarrays for each agent
                moves_x = []
                moves_y = []
                # loop thru each move:
                for x_move in a_x:
                    # if it exists (if len is less than max idx)
                    if len(x_move) > max:
                        # print(type(x_move[max]))
                        moves_x.append(x_move[max])
                 # loop thru each move:
                for y_move in a_y:
                    # if it exists (if len is less than max idx)
                    if len(y_move) > max:
                        moves_y.append(y_move[max])
                # add to data 
                self.data['agent_moves'] += [[moves_x, moves_y]]

            best_list.append(self.agents[max])
            fits[max] = -float('inf')
        
        # perform crossover
        idxs = list(range(len(best_list)))
        while len(idxs) > 1:
            p1_idx = np.random.choice(idxs)
            idxs.remove(p1_idx)
            p2_idx = np.random.choice(idxs)
            idxs.remove(p2_idx)
            best_list[p1_idx], best_list[p2_idx] = self.crossover(best_list[p1_idx], best_list[p2_idx])
        
        # create new arr with mutated agents
        new_list = []
        for _ in range(len(self.agents)-len(best_list)):
            agent = copy.deepcopy(best_list[np.random.choice(list(range(len(best_list))))])
            # perform mutate op on each par in agent.rnn

            # U
            for i, _ in np.ndenumerate(agent.rnn.U):
                if np.random.choice([True, False], p=[mutate_proba, 1-mutate_proba]):
                    agent.rnn.U[i] = agent.rnn.U[i] + np.random.uniform(-abs(agent.rnn.U[i])*mutate_range, abs(agent.rnn.U[i])*mutate_range)

            # V
            for i, _ in np.ndenumerate(agent.rnn.V):
                if np.random.choice([True, False], p=[mutate_proba, 1-mutate_proba]):
                    agent.rnn.V[i] = agent.rnn.V[i] + np.random.uniform(-abs(agent.rnn.V[i])*mutate_range, abs(agent.rnn.V[i])*mutate_range)

            # W
            for i, _ in np.ndenumerate(agent.rnn.W):
                if np.random.choice([True, False], p=[mutate_proba, 1-mutate_proba]):
                    agent.rnn.W[i] = agent.rnn.W[i] + np.random.uniform(-abs(agent.rnn.W[i])*mutate_range, abs(agent.rnn.W[i])*mutate_range)

            # b_hidden
            for i, _ in np.ndenumerate(agent.rnn.b_hidden):
                if np.random.choice([True, False], p=[mutate_proba, 1-mutate_proba]):
                    agent.rnn.b_hidden[i] = agent.rnn.b_hidden[i] + np.random.uniform(-abs(agent.rnn.b_hidden[i])*mutate_range, abs(agent.rnn.b_hidden[i])*mutate_range)
            
            # b_out
            for i, _ in np.ndenumerate(agent.rnn.b_out):
                if np.random.choice([True, False], p=[mutate_proba, 1-mutate_proba]):
                    agent.rnn.b_out[i] = agent.rnn.b_out[i] + np.random.uniform(-abs(agent.rnn.b_out[i])*mutate_range, abs(agent.rnn.b_out[i])*mutate_range)
            
            new_list.append(agent)
        
        old_fits = self.fitness()
        dists_to_goal = []
        num_moves = []
        is_alive = []
        for agent in self.agents:
            dists_to_goal.append(agent.dist_to_goal) 
            num_moves.append(agent.num_moves)
            is_alive.append(int(agent.is_alive))
        dists_to_goal = np.array(dists_to_goal)
        num_moves = np.array(num_moves)
        is_alive = np.array(is_alive)
        # finally we have are new agents for the next generation, change agents var to reflect this
        self.agents = best_list + new_list
        # return stats about the generation

        # reset all agents
        for agent in self.agents:
            agent.reset()

        return old_fits, dists_to_goal, num_moves, is_alive

    # perform multiple generation iterations and display stats for each:
    def evolve(self, num_generations=100, moves_allowed=100, sel_prop=0.3, mutate_proba=0.01, mutate_range=0.3):
        for i in range(num_generations):
            old_fits, dists_to_goal, num_moves, is_alive = self.genetic_iter(i, moves_allowed, sel_prop, mutate_proba, mutate_range)
            print(f'Generation {i}: dist {dists_to_goal.min()}, moves {num_moves.min(), num_moves.max()}, alive prop {is_alive.mean()}, fit {max(old_fits)}')
            # add to data
            self.data['max_fit'] += [max(old_fits)]
            self.data['alive_prop'] += [is_alive.mean()]
            self.data['min_dist'] += [dists_to_goal.min()]
            self.data['max_moves'] += [int(num_moves.max())]
            # print(old_fits)
            # print(dists_to_goal)

    # perfrom crossover on two agents inplace
    def crossover(self, agent1, agent2):
        # perform crossover on two numpy arrays (assuming they are the same size)
        def crossover_array(arr1, arr2):
            for i, _ in np.ndenumerate(arr1):
                swap = np.random.choice([True, False])
                if swap:
                    temp = arr1[i]
                    arr1[i] = arr2[i]
                    arr2[i] = temp
            return arr1, arr2
        
        agent1.rnn.U, agent2.rnn.U = crossover_array(agent1.rnn.U, agent2.rnn.U)
        agent1.rnn.W, agent2.rnn.W = crossover_array(agent1.rnn.W, agent2.rnn.W)
        agent1.rnn.V, agent2.rnn.V = crossover_array(agent1.rnn.V, agent2.rnn.V)
        agent1.rnn.b_hidden, agent2.rnn.b_hidden = crossover_array(agent1.rnn.b_hidden, agent2.rnn.b_hidden)
        agent1.rnn.b_out, agent2.rnn.b_out = crossover_array(agent1.rnn.b_out, agent2.rnn.b_out)

        # might be unnessecary
        return agent1, agent2



    # get fitnesses of all agents
    def fitness(self):
        dists_to_goal = []
        num_moves = []
        is_alive = []
        for agent in self.agents:
            dists_to_goal.append(agent.dist_to_goal) 
            num_moves.append(agent.num_moves)
            is_alive.append(int(agent.is_alive))
        dists_to_goal = np.array(dists_to_goal)
        num_moves = np.array(num_moves)
        is_alive = np.array(is_alive)
        #  / (1 -min_max(num_moves) + 1e-12)
        return (1 - min_max(dists_to_goal)) + (1 - min_max(num_moves)) + is_alive
        

def min_max(arr):
    arr = np.array(arr)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-12)
  