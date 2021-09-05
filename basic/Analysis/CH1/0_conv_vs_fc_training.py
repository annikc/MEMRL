## plot training of full convolutional neural network using partially / fully observable states

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import gym
import pandas as pd

sys.path.append('../../modules/')
from Analysis.analysis_utils import get_avg_std, get_grids, analysis_specs, LINCLAB_COLS, structured_unstructured
from modules.Utils import running_mean as rm

data_dir = '../../Data/results/'
groups_to_split = ['env_name','representation']

conv_df = pd.read_csv('../../Data/conv_mf_retraining.csv')
conv_df_gb = conv_df.groupby(groups_to_split)["save_id"]
labels_for_plot = {'conv':'Partially Observable State', 'reward_conv':'Fully Observable State'}
rep_to_col = {'unstructured':'blue', 'structured':'red'}

sh_df = pd.read_csv('../../Data/train_test_shallowAC.csv')
sh_df['representation'] = sh_df['representation'].apply(structured_unstructured)
sh_df_gb = sh_df.groupby(groups_to_split)["save_id"]






envs = ['gridworld:gridworld-v1', 'gridworld:gridworld-v4', 'gridworld:gridworld-v3', 'gridworld:gridworld-v5']
grids = get_grids(envs)

def compare_conv_fc(env, conv_gb,fc_gb):
    fig, ax = plt.subplots(1,2, sharex='col')
    if env[-1] == '5':
        rwd_colrow0 = (3,9)
        rwd_colrow1= (16,9)
    else:
        rwd_colrow0 = (5,5)
        rwd_colrow1=(14,14)

    rect0 = plt.Rectangle(rwd_colrow0, 1, 1, facecolor='b',edgecolor=None, alpha=0.3)
    rect1 = plt.Rectangle(rwd_colrow1, 1, 1, facecolor='g', edgecolor=None,alpha=0.3)
    ax[0].pcolor(grids[envs.index(env)],cmap='bone_r',edgecolors='k', linewidths=0.1)
    ax[0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
    ax[0].set_aspect('equal')
    ax[0].add_patch(rect0)
    #ax[0].add_patch(rect1)
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].invert_yaxis()


    # show CONV Results
    id_list = list(conv_gb.get_group((env,'reward_conv')))
    train_array = []
    for i, id_num in enumerate(id_list):
        with open(data_dir+ f'{id_num}_data.p', 'rb') as f:
            dats = pickle.load(f)

            raw_score = dats['total_reward']#[0:5000]
            print(len(raw_score))
            normalization = analysis_specs['avg_max_rwd'][env+'1']
            training_transformed = rm((np.asarray(raw_score)+2.5)/(normalization +2.5),200)
            train_array.append(training_transformed)
            print('done', id_num)
    
    mean_perf = np.mean(train_array,axis=0)
    std_perf = np.std(train_array,axis=0)/np.sqrt(len(train_array))
    mins = mean_perf-std_perf
    maxes = mean_perf+std_perf

    ax[1].plot(np.arange(len(mean_perf)),mean_perf, color="gray", label='Image (Conv)')
    ax[1].fill_between(np.arange(len(mean_perf)),mins, maxes, color='gray', alpha=0.2)
    

    # show FC Results
    for rep in ['unstructured','structured']:
        id_list = list(fc_gb.get_group((env,rep)))
        train_array = []
        filler = np.zeros(10000)
        filler[:] = np.nan
        for i, id_num in enumerate(id_list):
            with open(data_dir+ f'{id_num}_data.p', 'rb') as f:
                dats = pickle.load(f)
                raw_score = dats['total_reward'][5000:-1]
                if len(raw_score)==9999:
                    raw_score+= list(filler)
                print(len(raw_score))
                normalization = analysis_specs['avg_max_rwd'][env+'1']
                training_transformed = rm((np.asarray(raw_score)+2.5)/(normalization +2.5),200)
                train_array.append(training_transformed)
                print('done', id_num)

        mean_perf = np.nanmean(train_array,axis=0)
        std_perf = np.nanstd(train_array,axis=0)/np.sqrt(len(train_array))
        mins = mean_perf-std_perf
        maxes = mean_perf+std_perf

        ax[1].plot(np.arange(len(mean_perf)),mean_perf, color=LINCLAB_COLS[rep_to_col[rep]], label=f'{rep} (FC)')
        ax[1].fill_between(np.arange(len(mean_perf)),mins, maxes, color=LINCLAB_COLS[rep_to_col[rep]], alpha=0.2)



    ax[1].set_ylim(0,1.1)
    ax[1].set_yticks([0,1])
    ax[1].set_yticklabels([0,100])
    ax[1].set_ylabel('Performance \n(% Optimal)')
    ax[1].set_xlabel('Episodes')
    ax[1].legend(loc='upper center', bbox_to_anchor = (0.5,1.1))
    plt.savefig(f'../figures/CH1/compare_conv_fc{env[-2:]}.svg')
    plt.show()

compare_conv_fc(envs[1],conv_df_gb,sh_df_gb)


def plot_train_test(df, df_gb, envs, reps, save=False):
    fig, ax = plt.subplots(len(envs),2, sharex='col')
    for e, env in enumerate(envs):
        if env[-1] == '5':
            rwd_colrow0 = (3,9)
            rwd_colrow1= (16,9)
        else:
            rwd_colrow0 = (5,5)
            rwd_colrow1=(14,14)

        rect0 = plt.Rectangle(rwd_colrow0, 1, 1, facecolor='b',edgecolor=None, alpha=0.3)
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
            id_list = list(df_gb.get_group((env,rep)))
            print(env,rep)
            for i, id_num in enumerate(id_list):
                # get training data
                train_dat_id = list(df.loc[df['save_id']==id_num]['load_from'])[0]
                with open(data_dir+ f'{train_dat_id}_data.p', 'rb') as f:
                    dats = pickle.load(f)
                    raw_score = dats['total_reward'][0:5000]
                    normalization = analysis_specs['avg_max_rwd'][env+'1']
                    training_transformed = (np.asarray(raw_score)+2.5)/(normalization +2.5)
                # get testing data
                with open(data_dir+ f'{id_num}_data.p', 'rb') as f:
                    dats = pickle.load(f)
                    raw_score = dats['total_reward']
                    normalization = analysis_specs['avg_max_rwd'][env+'1']
                    testing_transformed = (np.asarray(raw_score)+2.5)/(normalization +2.5)

                train_test_data = rm(np.concatenate((training_transformed,testing_transformed)),200)
                train_test_array.append(train_test_data)
                print('done', id_num)

            mean_perf = np.mean(train_test_array,axis=0)
            std_perf = np.std(train_test_array,axis=0)/np.sqrt(len(train_test_array))
            mins = mean_perf-std_perf
            maxes = mean_perf+std_perf

            ax[e,1].plot(np.arange(len(mean_perf)),mean_perf, color=LINCLAB_COLS[rep_to_col[rep]], label=labels_for_plot[rep])
            ax[e,1].fill_between(np.arange(len(mean_perf)),mins, maxes, color=LINCLAB_COLS[rep_to_col[rep]], alpha=0.2)
            ax[e,1].set_ylim(0,1.1)
            ax[e,1].axvline(x=4801, linestyle=":",color='gray')
        ax[e,1].set_ylabel('Performance \n(% Optimal)')

    ax[e,1].set_xlabel('Episodes')
    ax[0,1].legend(loc='upper center', bbox_to_anchor = (0.5,1.1))
    plt.savefig('../figures/CH1/conv_net_retrain.svg')
    plt.show()




def plot_all(save=True, cutoff=25000):
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
            v_list = list(df_gb.get_group((name,rep_to_plot)))
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