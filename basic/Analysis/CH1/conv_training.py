## plot training of full convolutional neural network using partially / fully observable states

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import gym
import pandas as pd

sys.path.append('../../modules/')
from Analysis.analysis_utils import get_avg_std
from Utils import running_mean as rm

rep_to_plot = 'reward_conv'

data_dir = '../../Data/results/'
df = pd.read_csv('../../Data/head_only_retrain.csv')

master_dict = {}
envs = df.env_name.unique()
reps = df.representation.unique()

for env in envs:
    master_dict[env] = {}
    for rep in reps:
        id_list = list(df.loc[(df['env_name']==env)
         & (df['representation']==rep)]['save_id'])

        master_dict[env][rep]=id_list

env_names = ['gridworld:gridworld-v11', 'gridworld:gridworld-v41','gridworld:gridworld-v31', 'gridworld:gridworld-v51']
grids = []
for ind, environment_to_plot in enumerate(env_names):
    env = gym.make(environment_to_plot)
    plt.close()
    grids.append(env.grid)


plot_all = True

if plot_all:
    fig, axs = plt.subplots(4, 2, sharex='col')
    #rect = plt.Rectangle((5,5), 1, 1, color='r', alpha=0.3)
    for i in range(len(grids)):
        rect = plt.Rectangle((5,5), 1, 1, color='g', alpha=0.3)
        axs[i,0].pcolor(grids[i],cmap='bone_r',edgecolors='k', linewidths=0.1)
        axs[i,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        axs[i,0].set_aspect('equal')
        axs[i,0].add_patch(rect)
        axs[i,0].invert_yaxis()

    for ind, name in enumerate(env_names):
        for rep_to_plot in reps:
            v_list = master_dict[name][rep_to_plot]
            avg_, std_ = get_avg_std(v_list,cutoff=25000)
            axs[ind,1].plot(avg_, label=f'{rep_to_plot}')
            axs[ind,1].fill_between(np.arange(len(avg_)),avg_-std_, avg_+std_, alpha=0.3)
        if ind == len(env_names)-1:
            axs[ind,1].set_xlabel('Episodes')
            axs[ind,1].set_ylabel('Cumulative \nReward')
    plt.savefig('../figures/CH1/conv_testing.png')

plt.show()
