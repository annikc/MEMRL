## plot training of full convolutional neural network using partially / fully observable states

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import gym
import pandas as pd
sys.path.append('../../modules/')
from Analysis.analysis_utils import get_avg_std, get_id_dict, get_grids
from Utils import running_mean as rm


data_dir = '../../Data/'
df = pd.read_csv(data_dir+'conv_mf_training.csv')
envs = df.env_name.unique()
envs.remove('gridworld:gridworld-v2')
reps = df.representation.unique()

master_dict = get_id_dict(df)

grids = get_grids([1,3,4,5])


def plot_all(save=False):
    fig, axs = plt.subplots(4, 2, sharex='col')
    #rect = plt.Rectangle((5,5), 1, 1, color='r', alpha=0.3)
    for i in range(len(grids)):
        rect = plt.Rectangle((15,15), 1, 1, color='g', alpha=0.3)
        axs[i,0].pcolor(grids[i],cmap='bone_r',edgecolors='k', linewidths=0.1)
        axs[i,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        axs[i,0].set_aspect('equal')
        axs[i,0].add_patch(rect)
        axs[i,0].invert_yaxis()

    for ind, name in enumerate(envs):
        for rep_to_plot in reps:
            v_list = master_dict[name][rep_to_plot]
            avg_, std_ = get_avg_std(v_list,cutoff=5000)
            axs[ind,1].plot(avg_, label=f'{rep_to_plot}')
            axs[ind,1].fill_between(np.arange(len(avg_)),avg_-std_, avg_+std_, alpha=0.3)
        if ind == len(envs)-1:
            axs[ind,1].set_xlabel('Episodes')
            axs[ind,1].set_ylabel('Cumulative \nReward')
        axs[0,1].legend(loc='upper center', ncol=2, bbox_to_anchor = (0.1,1.1))
    if save:
        plt.savefig('../figures/CH1/conv_training.svg',format='svg')
    plt.show()

plot_all()