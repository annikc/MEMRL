import numpy as np
import sys
sys.path.append('../../../modules')
from basic.modules.Utils import one_hot_state
from basic.modules.Agents.RepresentationLearning import PlaceCells
import pickle
import os
abspath = os.path.dirname(__file__)

def onehot(env):
    name = 'onehot'
    dim = env.nstates

    oh_state_reps = {}
    for state in env.useable:
        oh_state_reps[env.twoD2oneD(state)] = one_hot_state(env,env.twoD2oneD(state))

    return oh_state_reps, name, dim

def place_cell(env, **kwargs):
    f_size = kwargs.get('field_size', 1/(max(*env.shape)))
    name = f'state-centred pc f{f_size}'
    dim = env.nstates
    # place cells centred at each state of the environment
    # for randomly distributed centres use rand_place_cell
    cell_centres = []
    for i in range(env.nstates):
        cell_centres.append(env.oneD2twoD(i))

    pc_state_reps = {}

    pcs = PlaceCells(env.shape, dim, field_size=f_size, cell_centres=np.asarray(cell_centres))

    for state in env.useable:
        pc_state_reps[env.twoD2oneD(state)] = pcs.get_activities([state])[0]

    return pc_state_reps, name, dim

def rand_place_cell(env, **kwargs):
    f_size = kwargs.get('field_size', 1/(max(*env.shape)))
    pc_state_reps = {}
    name = f'random-centred pc f_{f_size}'
    dim = env.nstates
    pcs = PlaceCells(env.shape, dim, field_size=f_size) # centres of cells are randomly distributed
    for state in env.useable:
        pc_state_reps[env.twoD2oneD(state)] = pcs.get_activities([state])[0]

    return pc_state_reps, name, dim

def sr(env):
    env_name = env.unwrapped.spec.id
    with open(f'{abspath}/Learned_Rep_pickles/SR_{env_name}.p', 'rb') as f:
        sr_ = pickle.load(f)
    ### TODO - update with analytically calculated sr from adjacency matrix
    SR_matrix = np.sum(sr_, axis=0)

    name = 'successor'
    dim = SR_matrix.shape[1]


    sr_reps = {}
    for index in range(SR_matrix.shape[0]):
        sr_reps[index] = SR_matrix[index]

    return sr_reps, name, dim