## plot retraining of actor critic head using latent state representations of partially / fully observable states
# conv_head_only_retraing = latent states and loaded output layer weights
# empty_head_only_retraining = latent states and a clean init of ac output layer ("flat ac")
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist,squareform
import pickle

import gym
import pandas as pd
import torch
from modules.Agents.Networks import conv_PO_params, conv_FO_params
from modules.Agents.Networks import flex_ActorCritic, shallow_ActorCritic
from modules.Agents.RepresentationLearning.learned_representations import convs, reward_convs, onehot, sr, place_cell, random
from Analysis.analysis_utils import get_avg_std, get_grids, analysis_specs, LINCLAB_COLS, linc_coolwarm, linc_coolwarm_r, make_env_graph, compute_graph_distance_matrix
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

def get_conv_agent_pol_valmaps(df, env_name,rep, type='h0',train=True):
    gb = df.groupby(['env_name','representation'])["save_id"]#.apply(list)
    env = gym.make(env_name)
    plt.close()

    agent_id = list(gb.get_group((env_name, rep)))[0]
    load_id  = list(df.loc[df['save_id']==agent_id]['load_from'])[0]
    print(load_id,"loadID")
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
    latents = {}
    for key, img in state_reps.items():
        coord = env.oneD2twoD(key)
        p,v = net(img)
        # top layer
        if type =='h1':
            latents[key] = net.h_act.detach().numpy()

        # middle layer
        elif type == 'h0':
            latents[key] = net.test_activity.detach().numpy()
        else:
            raise(Exception("wrong type argument"))
        policy = tuple(p.detach().numpy()[0])
        value = v.item()
        policy_map[coord] = policy
        value_map[coord]  = value

    return policy_map, value_map, latents

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
        reward_curve = dats['total_reward']

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
    return policy_map, value_map, reward_curve

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

def get_graph_dist_from_state(envs_to_plot, sim_ind):
    graph_distances = []
    for i,test_env_name in enumerate(envs_to_plot):
        env=gym.make(test_env_name)
        plt.close()
        G = make_env_graph(env)
        gd = compute_graph_distance_matrix(G,env)
        dist_in_state_space = gd[sim_ind]
        graph_distances.append(dist_in_state_space)
    return graph_distances

def plot_learned_val_pol_all():
    fig, axes = plt.subplots(2,2, figsize=(10,10))
    ax = axes.flatten()
    env_name = envs[1][:-1]
    for ind, env_name in enumerate(envs):
        rep_name = 'conv_latents'
        if rep_name == 'conv_latents' or rep_name=='rwd_conv_latents':
            pol, val, state_reps = get_conv_agent_pol_valmaps(df, env_name, rep_name)
        else:
            pol,val = get_shallow_fc_agent_pol_valmaps_from_saved(env_name,rep_name,49)

        a= ax[ind].imshow(val,vmin=4, vmax=10)
        ax[ind].get_xaxis().set_visible(False)
        ax[ind].get_yaxis().set_visible(False)
        plt.savefig(f'../figures/CH1/v_maps.svg',format='svg')
    plt.show()

    for env_name in envs:
        rep_name = 'conv_latents'
        if rep_name == 'conv_latents' or rep_name=='rwd_conv_latents':
            pol, val, state_reps = get_conv_agent_pol_valmaps(df, env_name, rep_name)
        else:
            pol,val = get_shallow_fc_agent_pol_valmaps_from_saved(env_name,rep_name,49)

        env = gym.make(env_name[:-1])
        plt.close()
        plot_pref_pol(env,pol, save=True, directory='../figures/CH1/', title=env_name+'p')

env_name = envs[1]
env =gym.make(env_name)
plt.close()
rep_dict = {'analytic successor':sr, 'onehot':onehot, 'place_cell':place_cell, 'random':random}
rep_name = 'place_cell'
if rep_name == 'conv_latents' or rep_name=='rwd_conv_latents':
    pol, val, state_reps = get_conv_agent_pol_valmaps(df, env_name, rep_name, type='h0')
else:
    #pol,val, rwd = get_shallow_fc_agent_pol_valmaps_from_saved(env_name[:-1],rep_name,49)
    state_reps, _ , __, __ = rep_dict[rep_name](env)
    #fig,ax= plt.subplots(1,2)
    #ax[0].plot(rm(rwd,200))
    #ax[1].imshow(val)
    #plt.show()

for coord in [(5,5),(5,14),(14,5),(14,14)]:
    #coord = (5,5)
    oneDc = env.twoD2oneD(coord)
    print(oneDc)
#plt.imshow(state_reps[oneDc],aspect='auto', cmap=linc_coolwarm)
#plt.savefig('../figures/CH1/latent_state_vec.svg', format='svg')
#plt.show()

def plot_dist_to_neighbours(test_env_name, sim_ind,state_reps, geodesic_dist=True, single_pos=True):
    # make new env to run test in
    env = gym.make(test_env_name)
    plt.close()

    # make graph of env states
    G = make_env_graph(env)
    gd= compute_graph_distance_matrix(G,env)
    dist_in_state_space = np.delete(gd[sim_ind],sim_ind) #distance from sim ind to all other states

    cmap = linc_coolwarm_r #cm.get_cmap('coolwarm')
    try:
        reps_as_matrix = np.zeros((400,state_reps[0].shape[1]))
    except:
        reps_as_matrix = np.zeros((400,state_reps[0].shape[0]))
    reps_as_matrix[:]  = np.nan

    for ind, (k,v) in enumerate(state_reps.items()):
        reps_as_matrix[k] = v
        if k in env.obstacle:
            reps_as_matrix[k,:] = np.nan


    RS = squareform(pdist(reps_as_matrix,metric='chebyshev'))
    for state2d in env.obstacle:
        RS[state2d,:] = np.nan
        RS[:,state2d] = np.nan

    dist_in_rep_space = np.delete(RS[sim_ind],sim_ind)
    print(type(dist_in_rep_space))
    rgba = [cmap(x) for x in dist_in_rep_space/np.nanmax(dist_in_rep_space)]
    if geodesic_dist:
        plt.scatter(dist_in_state_space,dist_in_rep_space,color=LINCLAB_COLS['red'],alpha=0.4,linewidths=0.5 )
        plt.xlabel("D(s,s')")
        plt.ylabel("D(R(s), R(s'))")
        plt.ylim([0.,1.1])
        plt.savefig(f'../figures/CH1/latent_distance{sim_ind}.svg')
    else:
        if single_pos:
            a = plt.imshow(RS[sim_ind].reshape(env.shape)/np.nanmax(RS[sim_ind]), cmap=cmap, vmin=0, vmax=1)
            r,c = env.oneD2twoD(sim_ind)
            plt.gca().add_patch(plt.Rectangle(np.add((c,r),(-0.5,-0.5)), .99, .99, edgecolor='k', fill=False, alpha=1))
            plt.colorbar(a)
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            plt.savefig(f'../figures/CH1/representation_similarity/latent_sim{sim_ind}.svg')
        else:
            sliced = RS.copy()
            for state1d in reversed(env.obstacle):
                sliced = np.delete(sliced,state1d,0)
                sliced = np.delete(sliced,state1d,1)

            a = plt.imshow(sliced/np.nanmax(sliced), vmin=0, vmax=1,  cmap=linc_coolwarm)
            plt.colorbar(a)

    #plt.show()
    plt.close()
print(env.useable)
'''
for gd in [0]:
    for state in [339,79]: #[105,114,285,294]:
        plot_dist_to_neighbours(env_name,state,state_reps,geodesic_dist=gd, single_pos=True)

'''


def plot_grid_of_shit(test_env_name,state_reps, sim_ind, metric):
    # make new env to run test in
    env = gym.make(test_env_name)
    plt.close()

    # make graph of env states
    G = make_env_graph(env)
    gd= compute_graph_distance_matrix(G,env)
    dist_in_state_space = np.delete(gd[sim_ind],sim_ind) #distance from sim ind to all other states

    cmap = linc_coolwarm_r #cmx.get_cmap('Spectral_r') #
    try:
        reps_as_matrix = np.zeros((400,state_reps[0].shape[1]))
    except:
        reps_as_matrix = np.zeros((400,state_reps[0].shape[0]))
    reps_as_matrix[:]  = np.nan

    for ind, (k,v) in enumerate(state_reps.items()):
        reps_as_matrix[k] = v
        if k in env.obstacle:
            reps_as_matrix[k,:] = np.nan


    RS = squareform(pdist(reps_as_matrix,metric=metric))
    for state2d in env.obstacle:
        RS[state2d,:] = np.nan
        RS[:,state2d] = np.nan

    dist_in_rep_space = np.delete(RS[sim_ind],sim_ind)

    a = plt.imshow(RS[sim_ind].reshape(env.shape)/np.nanmax(RS[sim_ind]), cmap=cmap)
    r,c = env.oneD2twoD(sim_ind)
    plt.gca().add_patch(plt.Rectangle(np.add((c,r),(-0.5,-0.5)), .99, .99, edgecolor='k', fill=False, alpha=1))
    plt.colorbar(a)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.savefig(f'../figures/CH1/representation_similarity/distance_metrics/{rep_name}{metric}.svg')
    plt.show()

#metric = 'euclidean'
#plot_grid_of_shit('gridworld:gridworld-v4',state_reps, 169,metric)
for rep_name in ['conv_latents','onehot','random','place_cell','analytic successor']:
    if rep_name == 'conv_latents' or rep_name=='rwd_conv_latents':
        pol, val, state_reps = get_conv_agent_pol_valmaps(df, env_name, rep_name, type='h0')
    else:
        #pol,val, rwd = get_shallow_fc_agent_pol_valmaps_from_saved(env_name[:-1],rep_name,49)
        state_reps, _ , __, __ = rep_dict[rep_name](env)
    
    for e, metric in enumerate(['cityblock','euclidean','chebyshev']):
        plot_grid_of_shit('gridworld:gridworld-v4',state_reps, 169,metric)
