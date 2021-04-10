import numpy as np
import sys
import torch
sys.path.append('../../../modules')
#from basic.modules.Utils import one_hot_state
#from basic.modules.Agents.RepresentationLearning import PlaceCells
#from basic.modules.Agents.Networks import flex_ActorCritic as ac_net
# for jupyter no call to basic
from modules.Utils import one_hot_state
from modules.Agents.RepresentationLearning import PlaceCells
from modules.Agents.Networks import flex_ActorCritic as ac_net
import pickle
import os

abspath = os.path.dirname(__file__)


def onehot(env):
    name = 'onehot'
    dim = env.nstates

    oh_state_reps = {}
    for state in env.useable:
        oh_state_reps[env.twoD2oneD(state)] = one_hot_state(env, env.twoD2oneD(state))

    return oh_state_reps, name, dim, []


def place_cell(env, **kwargs):
    f_size = kwargs.get('field_size', 1 / (max(*env.shape)))
    name = f'state-centred pc f{f_size}'
    dim = env.nstates
    # place cells centred at each state of the environment
    # for randomly distributed centres use rand_place_cell
    cell_centres = []
    for i in range(env.nstates):
        cell_centres.append(np.divide(env.oneD2twoD(i), env.shape))

    pc_state_reps = {}

    pcs = PlaceCells(env.shape, dim, field_size=f_size, cell_centres=np.asarray(cell_centres))

    for state in env.useable:
        pc_state_reps[env.twoD2oneD(state)] = pcs.get_activities([state])[0] / max(pcs.get_activities([state])[0])

    return pc_state_reps, name, dim, pcs.cell_centres


def rand_place_cell(env, **kwargs):
    sort_rands = kwargs.get('sort', True)
    f_size = kwargs.get('field_size', 1 / (max(*env.shape)))
    pc_state_reps = {}
    name = f'random-centred pc f_{f_size}'
    dim = env.nstates
    pcs = PlaceCells(env.shape, dim, field_size=f_size)  # centres of cells are randomly distributed
    if sort_rands:
        centres = pcs.cell_centres
        a = centres[centres[:, 1].argsort()]
        b = a[a[:, 0].argsort()]

        pcs.cell_centres = b

    for state in env.useable:
        pc_state_reps[env.twoD2oneD(state)] = pcs.get_activities([state])[0] / max(pcs.get_activities([state])[0])

    return pc_state_reps, name, dim, pcs.cell_centres


def sr(env, **kwargs):
    discount = kwargs.get('discount', 0.98)
    adj = np.sum(env.P / len(env.action_list), axis=0)
    nstates = adj.shape[0]
    sr_mat = np.linalg.inv(np.eye(nstates) - discount * adj)
    name = 'analytic successor'
    dim = sr_mat.shape[1]

    sr_reps = {}
    for index in range(sr_mat.shape[0]):
        if max(sr_mat[index]) == 0:
            sr_reps[index] = np.zeros_like(sr_mat[index])
        else:
            sr_reps[index] = sr_mat[index] / max(sr_mat[index])

    return sr_reps, name, dim, []


def reward_convs(env, **kwargs):
    name = 'reward_conv'
    dim = env.grid.shape

    grid_array = env.grid
    reward_array = np.zeros_like(env.grid)
    for coord in env.rewards.keys():
        reward_array[coord] = 1

    conv_reps = {}
    for state in env.useable:
        state_array = np.zeros_like(env.grid)
        state_array[state] = 1

        conv_reps[env.twoD2oneD(state)] = np.asarray([[grid_array, reward_array, state_array]])

    return conv_reps, name, dim, []


def convs(env, **kwargs):
    name = 'conv'
    dim = env.grid.shape

    grid_array = env.grid

    conv_reps = {}
    for state in env.useable:
        state_array = np.zeros_like(env.grid)
        state_array[state] = 1

        conv_reps[env.twoD2oneD(state)] = np.asarray([[grid_array, state_array]])

    return conv_reps, name, dim, []


def random(env):
    name = 'random'
    dim = env.nstates

    np.random.seed(1234)
    rand_reps = np.random.random((env.nstates, env.nstates))
    rand_state_reps = {}
    for state in env.useable:
        rand_state_reps[env.twoD2oneD(state)] = rand_reps[env.twoD2oneD(state)]

    return rand_state_reps, name, dim, []


def latents(env, path_to_saved_agent):
    # TODO - shift saved agents to saved weight dicts
    network = torch.load(path_to_saved_agent)

    # states from learned representations
    state_reps, _, __, ___ = convs(env)

    latents = {}
    for index, inp in state_reps.items():
        # do a forward pass
        network(inp)
        # get hidden_layer activity
        latents[index] = network.h_act.detach().numpy()[0]

    name = 'conv_latents'
    dim = latents[0].shape[0]
    return latents, name, dim, []


