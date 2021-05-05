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
from modules.Utils import running_mean as rm

data_dir = '../../Data/'
dfa = pd.read_csv(data_dir+'ec_latent_test.csv')
df = pd.read_csv(data_dir+'ec_empty_test.csv')
comparison_dat1 = pd.read_csv(data_dir+'conv_head_only_retrain.csv')
comparison_dat2 = pd.read_csv(data_dir+'empty_head_only_retrain.csv')
#df = pd.concat([dfa,dfb])
print(df)
envs = df.env_name.unique()
reps = df.representation.unique()
envs = np.delete(envs, np.where(envs == 'gridworld:gridworld-v2'))
print('#####', envs)


master_dict = get_id_dict(df)
slave_dict1 = get_id_dict(comparison_dat1)
slave_dict2 = get_id_dict(comparison_dat2)
print([x[-2:] for x in envs])
grids = get_grids([x[-2:] for x in envs])

#labels_for_plot = {'conv_saved_latents':'Partially Observable State', 'rwd_conv_saved_latents':'Fully Observable State'}
labels_for_plot = {'conv_latents':'Partially Observable State', 'rwd_conv_latents':'Fully Observable State'}
convert_conv_head_to_ec_labels = {'conv_latents':'conv_saved_latents', 'rwd_conv_latents':'rwd_conv_saved_latents'}

convert_rep_to_color = {'conv_latents':'C0', 'rwd_conv_latents':'C1'}

def plot_all(save=False, cutoff=25000, smoothing=5000):
    fig, axs = plt.subplots(4, 3, sharex='col')
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
                if i[0:8] in ['69aa8807','9ea97939']:
                    v_list.remove(i)
            avg_, std_ = get_avg_std(v_list,cutoff=cutoff, smoothing=smoothing)
            axs[ind,jnd+1].plot(avg_, label=f'{labels_for_plot[rep_to_plot]}',color=convert_rep_to_color[rep_to_plot]) # label=f'n={len(v_list)}'
            axs[ind,jnd+1].fill_between(np.arange(len(avg_)),avg_-std_, avg_+std_, alpha=0.1,color=convert_rep_to_color[rep_to_plot])

            # compare w
            v_list = slave_dict1[name][convert_conv_head_to_ec_labels[rep_to_plot]]
            avg_, std_ = get_avg_std(v_list,cutoff=cutoff, smoothing=smoothing)
            axs[ind,jnd+1].plot(avg_, ':', color=convert_rep_to_color[rep_to_plot], alpha=0.5) # label=f'n={len(v_list)}'
            #axs[ind,1].fill_between(np.arange(len(avg_)),avg_-std_, avg_+std_, alpha=0.3)
            v_list = slave_dict2[name][rep_to_plot]
            avg_, std_ = get_avg_std(v_list,cutoff=cutoff, smoothing=smoothing)
            axs[ind,jnd+1].plot(avg_, ':', color='gray', alpha=0.3) # label=f'n={len(v_list)}'

            axs[ind,jnd+1].set_ylim([-4,12])
        #axs[ind,1].legend(loc=0)
        if ind == len(envs)-1:
            axs[ind,1].set_xlabel('Episodes')
            #axs[ind,1].set_ylabel('Cumulative \nReward')
    axs[0,1].legend(loc='upper center', ncol=1, bbox_to_anchor=(0.2,1.1))
    axs[0,2].legend(loc='upper center', ncol=1, bbox_to_anchor=(0.2,1.1))
    if save:
        plt.savefig('../figures/CH1/ec_testing.svg',format='svg')

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