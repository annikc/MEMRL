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
from Analysis.analysis_utils import get_avg_std, get_grids, analysis_specs, LINCLAB_COLS
from modules.Utils import running_mean as rm

data_dir = '../../Data/results/'
filename = 'conv_head_only_retrain' #'empty_head_only_retrain'
df = pd.read_csv(f'../../Data/{filename}.csv')
envs = ['gridworld:gridworld-v11', 'gridworld:gridworld-v41' ,'gridworld:gridworld-v31', 'gridworld:gridworld-v51']
reps = ['conv_latents', 'rwd_conv_latents']

lrs  = df.MF_lr.unique()
envs = np.delete(envs, np.where(envs == 'gridworld:gridworld-v2'))
gb = df.groupby(['env_name','representation'])["save_id"]#.apply(list)

grids = get_grids(envs)
labels_for_plot = {'conv_latents':'Partially Observable State', 'rwd_conv_latents':'Fully Observable State'} # for empty_head_only_retrain
colors_for_plot = {'conv_latents':LINCLAB_COLS['blue'], 'rwd_conv_latents':LINCLAB_COLS['red']} # for empty_head_only_retrain

def get_train_test(env,rep):
    scaling_factor = analysis_specs['avg_max_rwd'][env]
    id_list = list(gb.get_group((env, rep)))
    load_id = list(df.loc[df['save_id']==id_list[0]]['load_from'])[0]
    with open(data_dir+f'{load_id}_data.p','rb') as f:
        training_data = pickle.load(f)['total_reward']
    print(len(training_data))
    tot_rwds =[]
    for save_id in id_list:
        with open(data_dir+f'{save_id}_data.p','rb') as f:
            retrain_data = pickle.load(f)['total_reward']
        train_test_data = training_data+retrain_data
        scaled_  = (np.asarray(train_test_data)+2.5)/(scaling_factor+2.5)
        tot_rwds.append(rm(scaled_,200))

    mean_performance = np.nanmean(tot_rwds,axis=0)
    std_e_mean = np.nanstd(tot_rwds,axis=0)/np.sqrt(len(tot_rwds))
    return mean_performance, std_e_mean

#mean_perf, sem = get_train_test('gridworld:gridworld-v11', 'rwd_conv_latents')



def plot_all(envs, reps, save=True, cutoff=25000):
    fig, axs = plt.subplots(4, 2, sharex='col', sharey='col')
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
            mean_perf, sem = get_train_test(name,rep)
            xvals = np.arange(len(mean_perf))
            axs[ind,1].plot(xvals, mean_perf, color=colors_for_plot[rep])
            axs[ind,1].fill_between(xvals, mean_perf-sem, mean_perf+sem, color=colors_for_plot[rep], alpha=0.2)
            axs[ind,1].set_ylabel('Performance \n(% Optimal)')
    axs[0,1].set_ylim([0,1.1])
    axs[0,1].set_yticks([0,1])
    axs[0,1].set_yticklabels([0,100])
    if save:
        plt.savefig('../figures/CH1/conv_retrain_head_only.svg')
    plt.show()


plot_all(envs, reps)

