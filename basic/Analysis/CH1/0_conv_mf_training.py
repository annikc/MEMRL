## plot training of full convolutional neural network using partially / fully observable states

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
df = pd.read_csv('../../Data/conv_mf_retraining.csv')
#df = pd.read_csv('../../Data/head_only_retrain.csv')
envs = df.env_name.unique()
reps = df.representation.unique()
print(reps)
envs = np.delete(envs, np.where(envs == 'gridworld:gridworld-v2'))
print('#####', envs)

groups_to_split = ['env_name','representation','extra_info']
df_gb = df.groupby(groups_to_split)["save_id"]


envs = ['gridworld:gridworld-v1', 'gridworld:gridworld-v4', 'gridworld:gridworld-v3', 'gridworld:gridworld-v5']

grids = get_grids(envs)
labels_for_plot = {'conv':'Partially Observable State', 'reward_conv':'Fully Observable State','onehot':'Unstructured', 'analytic successor': 'Structured'}
rep_to_col = {'conv':'purple', 'reward_conv':'orange', 'onehot':'blue', 'analytic successor':'red'}
def plot_train_test(df, envs, reps, save=False):
    fig, ax = plt.subplots(4,2, sharex='col')
    ftsz=8
    groups_to_split = ['env_name','representation','extra_info']
    training_df = pd.read_csv('../../Data/conv_mf_training.csv')
    tr_gb = training_df.groupby(groups_to_split)['save_id']
    df_gb = df.groupby(groups_to_split)["save_id"]
    for e, env in enumerate(envs):
        if env[-1] == '5':
            rwd_colrow0 = (3,9)
            rwd_colrow1= (16,9)
        else:
            rwd_colrow0 = (5,5)
            rwd_colrow1=(14,14)

        rect0 = plt.Rectangle(rwd_colrow0, 1, 1, facecolor='gray',edgecolor=None, alpha=0.5)
        rect1 = plt.Rectangle(rwd_colrow1, 1, 1, facecolor='g', edgecolor=None,alpha=0.3)
        ax[e,0].pcolor(grids[e],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[e,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[e,0].set_aspect('equal')
        ax[e,0].add_patch(rect0)
        ax[e,0].add_patch(rect1)
        ax[e,0].get_xaxis().set_visible(False)
        ax[e,0].get_yaxis().set_visible(False)
        ax[e,0].invert_yaxis()

        for r, rep in enumerate(reps):
            print(env,rep)
            train_test_array = []
            id_array = []
            # get training
            test_dummy = np.zeros(25000)
            test_dummy[:] = np.nan
            train_dummy = np.zeros(5000)
            train_dummy[:] = np.nan

            train_ids = list(tr_gb.get_group((env,rep,'x')))
            for i, id_num in enumerate(train_ids):
                with open(data_dir+ f'{id_num}_data.p', 'rb') as f:
                    dats = pickle.load(f)
                    raw_score = dats['total_reward'][0:5000]
                    normalization = analysis_specs['avg_max_rwd'][env+'1']
                    training_transformed = (np.asarray(raw_score)+2.5)/(normalization +2.5)

                run_info = np.concatenate((training_transformed,test_dummy))
                id_array.append(id_num)
                train_test_array.append(run_info)
            print('training')
            # get testing
            test_ids = list(df_gb.get_group((env,rep,'x')))
            for i, id_num in enumerate(test_ids):
                with open(data_dir+ f'{id_num}_data.p', 'rb') as f:
                    dats = pickle.load(f)
                    raw_score = dats['total_reward']
                    normalization = analysis_specs['avg_max_rwd'][env+'1']
                    testing_transformed = (np.asarray(raw_score)+2.5)/(normalization +2.5)

                run_info = np.concatenate((train_dummy,testing_transformed))
                id_array.append(id_num)
                train_test_array.append(run_info)
            print('testing')

            mean_perf = rm(np.nanmean(train_test_array,axis=0),200)
            std_perf = rm(np.nanstd(train_test_array,axis=0),200)/np.sqrt(len(train_test_array))
            mins = mean_perf-std_perf
            maxes = mean_perf+std_perf

            ax[e,1].plot(np.arange(len(mean_perf)),mean_perf, color=LINCLAB_COLS[rep_to_col[rep]], label=labels_for_plot[rep])
            ax[e,1].fill_between(np.arange(len(mean_perf)),mins, maxes, color=LINCLAB_COLS[rep_to_col[rep]], alpha=0.2)
            ax[e,1].set_ylim(0,1.1)
            ax[e,1].set_yticks([0,1])
            ax[e,1].set_yticklabels([0,100])
            ax[e,1].axvline(x=4801, linestyle=":",color='gray')
            ax[e,1].tick_params(axis='both', which='major', labelsize=8)
        ax[e,1].set_ylabel('Performance \n(% Optimal)', fontsize=ftsz)

    ax[e,1].set_xlabel('Episodes',fontsize=ftsz)

    ax[0,1].legend(loc='upper center', bbox_to_anchor = (0.5,1.1))
    plt.savefig('../figures/CH1/conv_net_retrain.svg')
    plt.show()


#plot_train_test(df, envs, ['conv'])

def plot_train_only(df, envs, reps, save=False):
    ftsz =8
    groups_to_split = ['env_name','representation','extra_info']
    df_gb = df.groupby(groups_to_split)["save_id"]
    fig, ax = plt.subplots(len(envs),2, sharex='col')
    for e, env in enumerate(envs):
        if env[-1] == '5':
            rwd_colrow0 = (3,9)
            rwd_colrow1= (16,9)
        else:
            rwd_colrow0 = (5,5)
            rwd_colrow1=(14,14)

        rect0 = plt.Rectangle(rwd_colrow0, 1, 1, facecolor='gray',edgecolor=None, alpha=0.5)
        rect1 = plt.Rectangle(rwd_colrow1, 1, 1, facecolor='g', edgecolor=None,alpha=0.3)
        ax[e,0].pcolor(grids[e],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[e,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[e,0].set_aspect('equal')
        ax[e,0].add_patch(rect0)
        ax[e,0].add_patch(rect1)
        ax[e,0].get_xaxis().set_visible(False)
        ax[e,0].get_yaxis().set_visible(False)
        ax[e,0].invert_yaxis()

        for r, rep in enumerate(reps):
            train_test_array = []
            id_list = list(df_gb.get_group((env,rep,'x')))
            print(env,rep, len(id_list))
            for i, id_num in enumerate(id_list[0:5]):
                # get training data
                with open(data_dir+ f'{id_num}_data.p', 'rb') as f:
                    dats = pickle.load(f)
                    raw_score = dats['total_reward'][0:5000]
                    normalization = analysis_specs['avg_max_rwd'][env+'1']
                    training_transformed = (np.asarray(raw_score)+2.5)/(normalization +2.5)
                '''
                # get testing data
                with open(data_dir+ f'{id_num}_data.p', 'rb') as f:
                    dats = pickle.load(f)
                    raw_score = dats['total_reward']
                    normalization = analysis_specs['avg_max_rwd'][env+'1']
                    testing_transformed = (np.asarray(raw_score)+2.5)/(normalization +2.5)

                train_test_data = rm(np.concatenate((training_transformed,testing_transformed)),200)
                '''
                train_test_data = rm(training_transformed,100)
                train_test_array.append(train_test_data)
                print('done', id_num)

            mean_perf = np.mean(train_test_array,axis=0)
            std_perf = np.std(train_test_array,axis=0)/np.sqrt(len(train_test_array))
            mins = mean_perf-std_perf
            maxes = mean_perf+std_perf

            ax[e,1].plot(np.arange(len(mean_perf)),mean_perf, color=LINCLAB_COLS[rep_to_col[rep]], label=labels_for_plot[rep])
            ax[e,1].fill_between(np.arange(len(mean_perf)),mins, maxes, color=LINCLAB_COLS[rep_to_col[rep]], alpha=0.2)
            ax[e,1].set_ylim(0,1.1)
            ax[e,1].set_yticks([0,1])
            ax[e,1].set_yticklabels([0,100],fontsize=ftsz)

            #ax[e,1].axvline(x=4801, linestyle=":",color='gray')

        ax[e,1].set_ylabel('Performance \n(% Optimal)', fontsize=ftsz)

    ax[e,1].set_xlabel('Episodes',fontsize=ftsz)
    ax[e,1].set_xticks([0,2500,5000])
    ax[e,1].set_xticklabels([0,2500,5000],fontsize=ftsz)
    ax[0,1].legend(loc='upper center', bbox_to_anchor = (0.5,1.1))
    if save:
        plt.savefig('../figures/CH1/conv_net_train_only.svg')
    plt.show()

#df = pd.read_csv('../../Data/conv_mf_training.csv')
#plot_train_only(df, envs[0:2], ['conv'],save =True)
df = pd.read_csv('../../Data/train_test_shallowAC.csv')
plot_train_only(df, envs[0:2], ['onehot', 'analytic successor'],save=True)

def plot_all(save=False, cutoff=25000):
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
            v_list = list(df_gb.get_group((name,rep_to_plot,'x')))
            for i in v_list:
                print(i[0:8])
                if i[0:8] in ['69aa8807','9ea97939']:
                    v_list.remove(i)
            avg_, std_ = get_avg_std(v_list,cutoff=cutoff, smoothing=200)
            axs[ind,1].plot(avg_, label=f'{labels_for_plot[rep_to_plot]}')
            axs[ind,1].fill_between(np.arange(len(avg_)),avg_-std_/np.sqrt(len(v_list)), avg_+std_/np.sqrt(len(v_list)), alpha=0.3)
            axs[ind,1].set_ylim([-4,12])
        if ind == len(envs)-1:
            axs[ind,1].set_xlabel('Episodes')
            #axs[ind,1].set_ylabel('Cumulative \nReward')
    axs[0,1].legend(loc='upper center', ncol=1, bbox_to_anchor=(0.2,1.1))
    if save:
        plt.savefig('../figures/CH1/conv_retraining.svg',format='svg')

    plt.show()



#plot_all(cutoff=25000)