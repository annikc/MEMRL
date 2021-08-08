## plot retraining of actor critic head using latent state representations of partially / fully observable states
# conv_head_only_retraing = latent states and loaded output layer weights
# empty_head_only_retraining = latent states and a clean init of ac output layer ("flat ac")
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import gym
import pandas as pd

sys.path.append('../../modules/')
from Analysis.analysis_utils import get_avg_std, get_grids
from modules.Utils import running_mean as rm

data_dir = '../../Data/results/'
filename = 'conv_head_only_retrain' #'empty_head_only_retrain'
df = pd.read_csv(f'../../Data/{filename}.csv')
envs = df.env_name.unique()
reps = df.representation.unique()
lrs  = df.MF_lr.unique()
envs = np.delete(envs, np.where(envs == 'gridworld:gridworld-v2'))
print(df)
gb = df.groupby(['env_name','representation','MF_lr'])["save_id"]#.apply(list)

for key, item in gb:
    print(key, list(item))

print(list(gb.get_group(('gridworld:gridworld-v31', 'conv_latents', 0.001))))
grids = get_grids(envs)
labels_for_plot = {'conv_latents':'Partially Observable State', 'rwd_conv_latents':'Fully Observable State'} # for empty_head_only_retrain

def plot_all(save=True, cutoff=25000):
    rep_to_plot = 'conv_latents'
    fig, axs = plt.subplots(4, 3, sharex='col')
    for i in range(len(grids)):
        rect = plt.Rectangle((14,14), 1, 1, color='g', alpha=0.3)
        axs[i,0].pcolor(grids[i],cmap='bone_r',edgecolors='k', linewidths=0.1)
        axs[i,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        axs[i,0].set_aspect('equal')
        axs[i,0].add_patch(rect)
        axs[i,0].get_xaxis().set_visible(False)
        axs[i,0].get_yaxis().set_visible(False)
        axs[i,0].invert_yaxis()

    for ind, name in enumerate(envs):
        for jnd, rep in enumerate(reps):
            for lr in lrs:
                v_list = list(gb.get_group((name, rep, lr)))
                for i in v_list:
                    print(i[0:8])
                    if i[0:8] in ['69aa8807','9ea97939']:
                        v_list.remove(i)
                avg_, std_ = get_avg_std(v_list,cutoff=cutoff, smoothing=500)
                axs[ind,jnd+1].plot(avg_, label=f'{lr}')
                axs[ind,jnd+1].fill_between(np.arange(len(avg_)),avg_-std_, avg_+std_, alpha=0.3)
                axs[ind,jnd+1].set_ylim([-4,12])
            if ind == len(envs)-1:
                axs[ind,jnd+1].set_xlabel('Episodes')
                #axs[ind,1].set_ylabel('Cumulative \nReward')
    axs[0,1].legend(loc='upper center', ncol=1, bbox_to_anchor=(0.2,1.1))
    if save:
        plt.savefig(f'../figures/CH1/{filename}_varied_LR.svg',format='svg')

    plt.show()
plot_all()




def old_plot_all(save=True, cutoff=25000):
    fig, axs = plt.subplots(4, 2, sharex='col')
    for i in range(len(grids)):
        rect = plt.Rectangle((15,15), 1, 1, color='g', alpha=0.3)
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
            avg_, std_ = get_avg_std(v_list,cutoff=cutoff, smoothing=500)
            axs[ind,1].plot(avg_, label=f'{labels_for_plot[rep_to_plot]}')
            axs[ind,1].fill_between(np.arange(len(avg_)),avg_-std_, avg_+std_, alpha=0.3)
            axs[ind,1].set_ylim([-4,12])
        if ind == len(envs)-1:
            axs[ind,1].set_xlabel('Episodes')
            #axs[ind,1].set_ylabel('Cumulative \nReward')
    axs[0,1].legend(loc='upper center', ncol=1, bbox_to_anchor=(0.2,1.1))
    if save:
        plt.savefig(f'../figures/CH1/{filename}.svg',format='svg')

    plt.show()

def plot_each(env_name, rep,cutoff=25000, smoothing=500):
    plt.figure()
    list_of_ids = master_dict[env_name][rep]
    for id_num in list_of_ids:
        with open(data_dir+ f'{id_num}_data.p', 'rb') as f:
            dats = pickle.load(f)
            reward_info = dats['total_reward'][0:cutoff]
        processed_rwd = rm(reward_info, smoothing)
        plt.plot(processed_rwd, label=id_num[0:8])
    plt.legend(loc='upper center', bbox_to_anchor=(0.1,1.1))
    plt.ylim([-4,12])
    plt.show()

#plot_all(cutoff=25000)
#plot_each(envs[2],reps[1])