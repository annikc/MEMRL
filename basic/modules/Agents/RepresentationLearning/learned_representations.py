import numpy as np
import sys
sys.path.append('../../../modules')
#from basic.modules.Utils import one_hot_state
#from basic.modules.Agents.RepresentationLearning import PlaceCells
# for jupyter no call to basic
from modules.Utils import one_hot_state
from modules.Agents.RepresentationLearning import PlaceCells
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
        cell_centres.append(np.divide(env.oneD2twoD(i),env.shape) )

    pc_state_reps = {}

    pcs = PlaceCells(env.shape, dim, field_size=f_size, cell_centres=np.asarray(cell_centres))

    for state in env.useable:
        pc_state_reps[env.twoD2oneD(state)] = pcs.get_activities([state])[0]/max(pcs.get_activities([state])[0])

    return pc_state_reps, name, dim

def rand_place_cell(env, **kwargs):
    f_size = kwargs.get('field_size', 1/(max(*env.shape)))
    pc_state_reps = {}
    name = f'random-centred pc f_{f_size}'
    dim = env.nstates
    pcs = PlaceCells(env.shape, dim, field_size=f_size) # centres of cells are randomly distributed
    for state in env.useable:
        pc_state_reps[env.twoD2oneD(state)] = pcs.get_activities([state])[0]/max(pcs.get_activities([state])[0])

    return pc_state_reps, name, dim

def sr(env, **kwargs):
    discount = kwargs.get('discount', 0.98)
    adj = np.sum(env.P/len(env.action_list),axis=0)
    nstates = adj.shape[0]
    sr_mat = np.linalg.inv(np.eye(nstates) - discount*adj)
    name = 'analytic successor'
    dim = sr_mat.shape[1]

    sr_reps = {}
    for index in range(sr_mat.shape[0]):
        if max(sr_mat[index])==0:
            sr_reps[index] = np.zeros_like(sr_mat[index])
        else:
            sr_reps[index] = sr_mat[index]/max(sr_mat[index])

    return sr_reps, name, dim
