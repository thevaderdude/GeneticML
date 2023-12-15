from canvas import Canvas
import json


canvas = Canvas(num_agents=100, num_nodes=20, graph=True)
canvas.evolve(moves_allowed=30, num_generations=20, sel_prop=0.3, mutate_proba=0.3, mutate_range=0.7)
# print(canvas.data)
# with open('maze4.json', 'w') as file:
    # json.dump(canvas.data, file)
# standard preset: 20 nodes, 20 generations, 100 agents, selprop = 0.3, mutateproba = 0.01, mutate_range=0.3
# better preset: 50 nodes, 20 generations, 100 agents, selprop = 0.3, mutateproba = 0.01, mutate_range=0.3
# low_population preset: 20 nodes, 20 generations, 20 agents, selprop = 0.3, mutateproba = 0.01, mutate_range=0.3
# high mutation preset: 20 nodes, 100 generations, 100 agents, selprop = 0.3, mutateproba = 0.3, mutate_range=0.7