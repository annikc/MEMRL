## plot training of full convolutional neural network using partially / fully observable states

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import gym
import pandas as pd

sys.path.append('../../modules/')
from Analysis.analysis_utils import get_avg_std, get_id_dict
from Utils import running_mean as rm

data_dir = '../../Data/results/'
df = pd.read_csv('../../Data/conv_mf_training.csv')
envs = df.env_name.unique()
reps = df.representation.unique()
print(reps)
envs = np.delete(envs, np.where(envs == 'gridworld:gridworld-v2'))
print('#####', envs)


master_dict = get_id_dict(df)

grids=[]
for ind, environment_to_plot in enumerate(envs):
    env = gym.make(environment_to_plot)
    plt.close()
    grids.append(env.grid)

labels_for_plot = {'conv':'Partially Observable State', 'reward_conv':'Fully Observable State'}
def plot_all():
    fig, axs = plt.subplots(4, 2, sharex='col')
    for i in range(len(grids)):
        rect = plt.Rectangle((5,5), 1, 1, color='g', alpha=0.3)
        axs[i,0].pcolor(grids[i],cmap='bone_r',edgecolors='k', linewidths=0.1)
        axs[i,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        axs[i,0].set_aspect('equal')
        axs[i,0].add_patch(rect)
        axs[i,0].get_xaxis().set_visible(False)
        axs[i,0].get_yaxis().set_visible(False)
        axs[i,0].invert_yaxis()

    for ind, name in enumerate(envs):
        for rep_to_plot in reps:
            v_list = master_dict[name][rep_to_plot]
            for i in v_list:
                print(i[0:8])
                if i[0:8] in ['69aa8807','9ea97939']:
                    v_list.remove(i)
            avg_, std_ = get_avg_std(v_list,cutoff=5000, smoothing=50)
            axs[ind,1].plot(avg_, label=f'{labels_for_plot[rep_to_plot]}')
            axs[ind,1].fill_between(np.arange(len(avg_)),avg_-std_, avg_+std_, alpha=0.3)
        if ind == len(envs)-1:
            axs[ind,1].set_xlabel('Episodes')
            #axs[ind,1].set_ylabel('Cumulative \nReward')
    axs[0,1].legend(loc='upper center', ncol=1, bbox_to_anchor=(0.2,1.1))
    plt.savefig('../figures/CH1/conv_training.svg',format='svg')

    plt.show()

def plot_each(env_name, rep,cutoff=5000, smoothing=50):
    plt.figure()
    list_of_ids = master_dict[env_name][rep]
    for id_num in list_of_ids:
        with open(data_dir+ f'{id_num}_data.p', 'rb') as f:
            dats = pickle.load(f)
            reward_info = dats['total_reward'][0:cutoff]
        processed_rwd = rm(reward_info, smoothing)
        plt.plot(processed_rwd, label=id_num[0:8])
    plt.legend(loc='upper center', bbox_to_anchor=(0.1,1.1))
    plt.show()

#plot_each('gridworld:gridworld-v3','conv')
plot_all()