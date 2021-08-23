## plot retraining of actor critic head using latent state representations of partially / fully observable states
# conv_head_only_retraing = latent states and loaded output layer weights
# empty_head_only_retraining = latent states and a clean init of ac output layer ("flat ac")
import numpy as np
import matplotlib.pyplot as plt
import pickle

import gym
import pandas as pd
import torch
from modules.Agents.Networks import conv_PO_params, conv_FO_params
from modules.Agents.Networks import flex_ActorCritic, shallow_ActorCritic
from modules.Agents.RepresentationLearning.learned_representations import convs, reward_convs, onehot, sr
from Analysis.analysis_utils import get_avg_std, get_grids, analysis_specs, LINCLAB_COLS
from modules.Utils import running_mean as rm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import matplotlib.cm as cmx

from modules.Utils import softmax
from scipy.stats import entropy

data_dir = '../../Data/'

df = pd.read_csv(data_dir+'conv_head_only_retrain.csv')
envs = ['gridworld:gridworld-v11', 'gridworld:gridworld-v41' ,'gridworld:gridworld-v31', 'gridworld:gridworld-v51']
reps = ['conv_latents', 'rwd_conv_latents']

grids = get_grids(envs)
labels_for_plot = {'conv_latents':'Partially Observable State', 'rwd_conv_latents':'Fully Observable State'} # for empty_head_only_retrain
colors_for_plot = {'conv_latents':LINCLAB_COLS['blue'], 'rwd_conv_latents':LINCLAB_COLS['red']} # for empty_head_only_retrain

def get_conv_agent_pol_valmaps(df, env_name,rep, train=True):
    gb = df.groupby(['env_name','representation'])["save_id"]#.apply(list)
    env = gym.make(env_name)
    plt.close()

    agent_id = list(gb.get_group((env_name, rep)))[0]
    load_id  = list(df.loc[df['save_id']==agent_id]['load_from'])[0]
    # load agent
    if train:
        state_dict = torch.load(data_dir+f'agents/{load_id}.pt')
    else:
        state_dict = torch.load(data_dir+f'agents/{agent_id}.pt')

    # get state reps
    if rep == 'conv_latents':
        params = conv_PO_params
        state_reps, _, __, ___  = convs(env)

    elif rep == 'rwd_conv_latents':
        params = conv_FO_params
        state_reps, _, __, ___  = reward_convs(env)

    network_parameters = params(env)
    net = flex_ActorCritic(network_parameters)
    net.load_state_dict(state_dict)

    policy_map = np.empty(env.shape, dtype=[(x,'f8') for x in env.action_list])
    value_map = np.zeros(env.shape)
    value_map[:] = np.nan
    for key, img in state_reps.items():
        coord = env.oneD2twoD(key)
        p,v = net(img)

        policy = tuple(p.detach().numpy()[0])
        value = v.item()
        policy_map[coord] = policy
        value_map[coord]  = value

    return policy_map, value_map

def get_shallow_fc_agent_pol_valmaps_from_saved(env_name,rep,map_index):
    # df = shallow AC with PV maps
    df = pd.read_csv(data_dir+'shallowAC_withPVmaps.csv')
    gb = df.groupby(['env_name','representation'])["save_id"]
    rep_types = {'onehot': onehot, 'analytic successor':sr }
    env = gym.make(env_name)
    plt.close()

    agent_id = list(gb.get_group((env_name, rep)))[0]
    with open(data_dir+f'results/{agent_id}_data.p', 'rb') as f:
        dats = pickle.load(f)
        pol_maps = dats['P_snap']
        val_maps = dats['V_snap']

    policy_map = pol_maps[map_index]
    value_map  = val_maps[map_index]
    for x in env.obstacle2D:
        value_map[x]=np.nan
    '''
    # load agent
    state_dict = torch.load(data_dir+f'agents/{agent_id}.pt')

    state_reps, _, __, ___  = rep_types[rep](env)

    in_dims = df.loc[df['save_id']==agent_id]['MF_input_dims'].item()
    hi_dims = df.loc[df['save_id']==agent_id]['MF_hidden_dims'].item()

    net = shallow_ActorCritic(input_dims=in_dims,hidden_dims=hi_dims,output_dims=env.action_space.n, lr=0)
    net.load_state_dict(state_dict)

    policy_map = np.empty(env.shape, dtype=[(x,'f8') for x in env.action_list])
    value_map = np.zeros(env.shape)
    value_map[:] = np.nan
    for key, img in state_reps.items():
        coord = env.oneD2twoD(key)
        p,v = net(img)

        policy = tuple(p.detach().numpy())
        value = v.item()
        policy_map[coord] = policy
        value_map[coord]  = value
    
    '''
    return policy_map, value_map

def make_arrows(action, probability):
    '''
    alternate style:
        def make_arrows(action):
        offsets = [(0,0.25),(0,-0.25),(0.25,0),(-0.25,0),(0,0),(0.1,0.1) ] # D U R L J P
        dx,dy = offsets[action]
        head_w, head_l = 0.1, 0.1
        return dx, dy, head_w, head_l
    :param action:
    :param probability:
    :return:
    '''
    if probability == 0:
        dx, dy = 0, 0
        head_w, head_l = 0, 0
    else:
        dxdy = [(0.0, 0.25),  # D
                (0.0, -0.25),  # U
                (0.25, 0.0),  # R
                (-0.25, 0.0),  # L
                (0.1, -0.1),  # points right and up #J
                (-0.1, 0.1),  # points left and down # P
                ]
        dx, dy = dxdy[action]  #dxdy[(action-1)%len(dxdy)] ## use if action-space remapping

        head_w, head_l = 0.1, 0.1

    return dx, dy, head_w, head_l

def plot_pref_pol(maze, policy_array, save=False, **kwargs):
    '''
        :param maze: the environment object
        :param save: bool. save figure in current directory
        :return: None
        '''
    show = kwargs.get('show', True)
    title = kwargs.get('title', 'Policy Entropy')
    directory = kwargs.get('directory', '../figures/')
    filetype = kwargs.get('filetype', 'svg')
    vmax = kwargs.get('upperbound', 2)
    rewards = kwargs.get('rwds', maze.rewards)
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_axes([0, 0, 0.85, 0.85])
    axc = fig.add_axes([0.75, 0, 0.05, 0.85])

    cmap = plt.cm.Spectral_r
    cNorm = colors.Normalize(vmin=0, vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    cb1 = colorbar.ColorbarBase(axc, cmap=cmap, norm=cNorm)
    # make base grid
    ax1.pcolor(maze.grid, vmin=0, vmax=vmax, cmap='bone')
    # add patch for reward location/s (red)
    for rwd_loc in rewards:
        rwd_r, rwd_c = rwd_loc
        ax1.add_patch(plt.Rectangle((rwd_c, rwd_r), width=0.99, height=1, linewidth=1, ec='white', fill=False))

    chance_threshold = kwargs.get('threshold', 0.18)  # np.round(1 / len(maze.actionlist), 6)

    for coord in maze.useable:
        i = coord[0]
        j = coord[1]
        policy = tuple(policy_array[i, j])

        dx, dy = 0.0, 0.0
        for ind, k in enumerate(policy):
            action = ind
            prob = k
            if prob < 0.01:
                pass
            else:
                dx1, dy1, head_w, head_l = make_arrows(action, prob)
                dx += dx1*prob
                dy += dy1*prob
        if dx ==0.0 and dy == 0.0:
            pass
        else:
            colorVal1 = scalarMap.to_rgba(entropy(policy))
            if entropy(policy) > 1.2:
                pass
            else:
                ax1.arrow(j + 0.5, i + 0.5, dx, dy, head_width=0.3, head_length=0.5, color=colorVal1)


    ax1.set_aspect('equal')
    ax1.set_title(title)
    ax1.invert_yaxis()
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    if save:
        plt.savefig(f'{directory}pref_{title}.{filetype}', format=f'{filetype}', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


env_name = envs[1][:-1]
rep_name = 'rwd_conv_latents'
if rep_name == 'conv_latents' or rep_name=='rwd_conv_latents':
    pol, val = get_conv_agent_pol_valmaps(df, env_name+'1', rep_name)
else:
    pol,val = get_shallow_fc_agent_pol_valmaps_from_saved(env_name,rep_name,49)


a= plt.imshow(val,vmin=4, vmax=10)
plt.colorbar(a)
plt.savefig(f'../figures/CH1/pv_maps/VAL{env_name[-2:]}_{rep_name}.svg',format='svg')
plt.show()

env = gym.make(env_name)
plt.close()
plot_pref_pol(env,pol, save=True, directory='../figures/CH1/pv_maps/',title=f'{env_name[-2:]}_{rep_name}')



