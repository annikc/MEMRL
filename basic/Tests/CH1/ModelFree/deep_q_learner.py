import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from basic.modules.Agents.RepresentationLearning import PlaceCells
'''
import modules.Agents.Networks as nets
import modules.Agents.EpisodicMemory as Memory
from modules.Agents import Agent
from modules.Experiments import expt as expt
import matplotlib.pyplot as plt
from modules.Utils import running_mean as rm
import time
import uuid
import csv
import pickle

'''


from basic.modules.Agents import DQ_agent

env = gym.make('gym_grid:gridworld-v1')
print(env.shape)
num_cells = 1
field_size = 0.1

pcs = PlaceCells(env.shape, num_cells, field_size)

def one_hot_state(state):
    vec = np.zeros(env.nstates)
    vec[state] = 1
    return vec

def onehot_state_collection(env):
    collection = []
    for state in range(env.nstates):
        vec = one_hot_state(state)
        collection.append(vec)
    return collection

def twoD_states(env):
    twods = []
    for state in range(env.nstates):
        twod = env.oneD2twoD(state)
        twods.append(twod)
    return twods

gridworld_onehots = onehot_state_collection(env)


## get place cell activities
# get all states as coordinates
two_d_states = twoD_states(env)


# make a collection of place cells
place_cells       = PlaceCells(env.shape,env.nstates, field_size=1/env.shape[0])

# get activities for each state
gridworld_pc_reps = place_cells.get_activities(two_d_states)

# get random indices for place cell fields to plot
def plot_some_place_fields(env, list_of_coords, place_cells):
    states = np.asarray(list_of_coords)
    gridworld_pc_reps = place_cells.get_activities(list_of_coords)

    num_pc_view = 9
    get_rand_cells = np.random.choice(env.nstates,num_pc_view,replace=False)

    fig, axes = plt.subplots(3,3)

    for i, ax in enumerate(axes.flat):
        # for every state, get what the place cell activity is
        ax.scatter(states[:,0],states[:,1], c =gridworld_pc_reps[:,get_rand_cells[i]])
        cell_center = np.round(np.multiply(place_cells.cell_centres[get_rand_cells[i]],env.shape),1)
        print(cell_center)
        ax.set_title(f'{cell_center}')
        ax.set_yticks([])
        ax.set_xticks([])
    plt.show()


input_dims = [8]
n_actions  = env.action_space.n
batch_size = 64
gamma      = 0.98
epsilon    = 1.0
lr         = 0.03


agent = DQ_agent(gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_decay=1e-4,
                 replace_target=0)

