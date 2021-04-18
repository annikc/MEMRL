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
from Analysis.analysis_utils import get_avg_std, get_id_dict, get_grids, plot_each
from Utils import running_mean as rm

data_dir = '../../Data/'
df = pd.read_csv(data_dir+'ec_testing.csv')
print(df)
envs = df.env_name.unique()
reps = df.representation.unique()
envs = np.delete(envs, np.where(envs == 'gridworld:gridworld-v1'))
reps = np.delete(reps, np.where(reps == 'saved_latents'))
print('#####',reps)
ref = pd.read_csv(data_dir+'ec_empty_test.csv')

ref_dict = get_id_dict(ref,reps=['conv_latents'])
for key in ref_dict.keys():
    print('ref dict', key, ref_dict[key].keys())


master_dict = get_id_dict(df, reps=reps)

print([x[-2:] for x in envs])
grids = get_grids([x[-2:] for x in envs])

convert_rep_to_color = {'analytic successor':'C0', 'onehot':'C1', 'random':'C2','state-centred pc f0.05':'C3'}
labels_for_plot = {'analytic successor':'SR', 'onehot':'onehot', 'random':'random','state-centred pc f0.05':'PC'}

def plot_all(save=False, cutoff=25000, smoothing=5000):
    fig, axs = plt.subplots(4, 3, sharex='col', sharey='col')
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
        for jnd, rep_to_plot in enumerate(reps):
            v_list = master_dict[name][rep_to_plot]
            for i in v_list:
                print(i[0:8])
            avg_, std_ = get_avg_std(v_list,cutoff=cutoff, smoothing=smoothing)
            axs[ind,1].plot(avg_, label=f'{labels_for_plot[rep_to_plot]}',color=convert_rep_to_color[rep_to_plot]) # label=f'n={len(v_list)}'
            axs[ind,1].fill_between(np.arange(len(avg_)),avg_-std_, avg_+std_, alpha=0.1,color=convert_rep_to_color[rep_to_plot])
            axs[ind,1].set_ylim([-4,12])
            axs[ind,2].bar(jnd, np.mean(std_),color=convert_rep_to_color[rep_to_plot])
        if ind == len(envs)-1:
            axs[ind,1].set_xlabel('Episodes')
            #axs[ind,1].set_ylabel('Cumulative \nReward')
        # compare reference:
        v_list = ref_dict[name]['conv_latents']
        avg_, std_ = get_avg_std(v_list,cutoff=cutoff,smoothing=smoothing)
        axs[ind,1].plot(avg_, label=f'latents',color='k') # label=f'n={len(v_list)}'
        axs[ind,1].fill_between(np.arange(len(avg_)),avg_-std_, avg_+std_, alpha=0.1,color='gray')
        axs[ind,2].bar(jnd+1, np.mean(std_),color='k')
    axs[0,1].legend(loc='upper center', ncol=5, bbox_to_anchor=(0.2,1.1))
    axs[0,2].set_ylim([0,6])
    if save:
        plt.savefig('../figures/CH1/ec_alt_rep_testing.svg',format='svg')

    plt.show()

'''def plot_each(env_name, rep,cutoff=25000, smoothing=500):
    plt.figure()
    list_of_ids = master_dict[env_name][rep]
    for id_num in list_of_ids:
        with open(data_dir+ f'results/{id_num}_data.p', 'rb') as f:
            dats = pickle.load(f)
            reward_info = dats['total_reward'][0:cutoff]
        processed_rwd = rm(reward_info, smoothing)
        plt.plot(processed_rwd, label=id_num[0:8])
    plt.legend(loc='upper center', bbox_to_anchor=(0.1,1.1))
    plt.ylim([-4,12])
    plt.show()'''

plot_all(cutoff=5000,smoothing=50,save=False)
env = envs[2]
rep = reps[1]

#plot_each(master_dict[env][rep], data_dir)