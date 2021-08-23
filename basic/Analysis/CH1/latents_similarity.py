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
from scipy.spatial.distance import euclidean, mahalanobis, pdist, squareform,cdist
import matplotlib.pyplot as plt


data_dir = '../../Data/'

df = pd.read_csv(data_dir+'conv_head_only_retrain.csv')
envs = ['gridworld:gridworld-v11', 'gridworld:gridworld-v41' ,'gridworld:gridworld-v31', 'gridworld:gridworld-v51']
reps = ['conv_latents', 'rwd_conv_latents']

grids = get_grids(envs)
labels_for_plot = {'conv_latents':'Partially Observable State', 'rwd_conv_latents':'Fully Observable State'} # for empty_head_only_retrain
colors_for_plot = {'conv_latents':LINCLAB_COLS['blue'], 'rwd_conv_latents':LINCLAB_COLS['red']} # for empty_head_only_retrain

def get_conv_agent_latents(df, env_name,rep, train=True):
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

    h0 = {}
    h1 = {}
    for key, img in state_reps.items():
        coord = env.oneD2twoD(key)
        p,v = net(img)

        h0[key] = net.test_activity.detach().numpy()
        h1[key] = net.h_act.detach().numpy()

    return h0, h1
env_name = envs[1]

rep_name = reps[0]

env= gym.make(env_name)
plt.close()

h0, h1 = get_conv_agent_latents(df, env_name, rep_name)
for i in env.obstacle:
    empty_array = np.zeros(600)
    empty_array[:] = np.nan
    h0[i] = empty_array
    h1[i] = empty_array[0:400]

h0_array = np.empty((400,600))
for i in range(400):
    h0_array[i] = h0[i]

h1_array = np.empty((400,400))
for i in range(400):
    h1_array[i] = h1[i]

coord = (9,8)
index = env.twoD2oneD((coord[1],coord[0]))
print(index)
metric = 'cosine'
h0_sim = cdist([h0_array[index]],h0_array,metric=metric).reshape(20,20)
h1_sim = cdist([h1_array[index]],h1_array,metric=metric).reshape(20,20)
print(env.oneD2twoD(index))
fig, ax = plt.subplots(1,2)
a = ax[0].imshow(h0_sim, vmin=0, vmax=.5)
ax[0].add_patch(plt.Rectangle((5-.5,5-.5),1,1,edgecolor='white', fill=False))
ax[0].add_patch(plt.Rectangle(np.add(coord,(-.5,-.5)),1,1,edgecolor='red', fill=False))
ax[0].set_title(f'$\psi$')
plt.colorbar(a, ax=ax[0])
b= ax[1].imshow(h1_sim, vmin=0, vmax=.5)
ax[1].add_patch(plt.Rectangle((5-.5,5-.5),1,1,edgecolor='white', fill=False))
ax[1].add_patch(plt.Rectangle(np.add(coord,(-.5,-.5)),1,1,edgecolor='red', fill=False))
ax[1].set_title(f'$\phi$')
plt.colorbar(b, ax=ax[1])
plt.show()