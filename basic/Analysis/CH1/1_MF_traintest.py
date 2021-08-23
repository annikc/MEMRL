import numpy as np
import pandas as pd
from Analysis.analysis_utils import structured_unstructured, LINCLAB_COLS, plot_specs, analysis_specs
from Analysis.analysis_utils import get_grids
from modules.Utils import running_mean as rm
from Analysis.analysis_utils import avg_performance_over_envs, avg_perf_over_envs_lines, avg_performance_over_envs_violins
import pickle
import matplotlib.pyplot as plt
cache_limits = analysis_specs['cache_limits']

# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'train_test_shallowAC.csv')
#df = pd.read_csv(parent_path+'test_perceptron_AC.csv')

df['representation'] = df['representation'].apply(structured_unstructured)
groups_to_split = ['env_name','representation','num_trials']
df_gb = df.groupby(groups_to_split)["save_id"]

envs_to_plot = ['gridworld:gridworld-v1','gridworld:gridworld-v4','gridworld:gridworld-v3','gridworld:gridworld-v5']
pcts_to_plot = [100,75,50,25]
reps_to_plot = ['unstructured','structured']
col_to_plot = {'unstructured':LINCLAB_COLS['blue'], 'structured':LINCLAB_COLS['red']}
grids = get_grids(envs_to_plot)

def plot_perceptron(df, envs_to_plot, reps_to_plot):
    grids = get_grids(envs_to_plot)
    fig, ax = plt.subplots(len(envs_to_plot),2,sharey='col', sharex='col')
    for e, env in enumerate(envs_to_plot):
        scaling_factor = analysis_specs['avg_max_rwd'][env+'1']
        if env[-1] == '5':
            rwd_colrow0 = (3,9)
            rwd_colrow1= (16,9)
        else:
            rwd_colrow0 = (5,5)
            rwd_colrow1=(14,14)

        rect0 = plt.Rectangle(rwd_colrow0, 1, 1, facecolor='gray',edgecolor=None, alpha=0.3)
        rect1 = plt.Rectangle(rwd_colrow1, 1, 1, facecolor='g', edgecolor=None,alpha=0.3)
        ax[e,0].pcolor(grids[e],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[e,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[e,0].set_aspect('equal')
        ax[e,0].add_patch(rect0)
        ax[e,0].add_patch(rect1)
        ax[e,0].get_xaxis().set_visible(False)
        ax[e,0].get_yaxis().set_visible(False)
        ax[e,0].invert_yaxis()

        for r, rep in enumerate(reps_to_plot):
            id_list = list(df_gb.get_group((env+'1',rep,10000)))
            print(env, rep, len(id_list))
            total_avg_reward = []
            for i, id_num in enumerate(id_list):
                # get training data
                train_dat_id = list(df.loc[df['save_id']==id_num]['load_from'])[0]
                with open(parent_path+ f'results/{train_dat_id}_data.p', 'rb') as f:
                    dats = pickle.load(f)
                    raw_score = dats['total_reward'][0:5000]
                    training_transformed = (np.asarray(raw_score)+2.5)/(scaling_factor +2.5)

                with open(parent_path+ f'results/{id_num}_data.p', 'rb') as f:
                    dats = pickle.load(f)
                    scaled_  = (np.asarray(dats['total_reward'])+2.5)/(scaling_factor+2.5)
                train_test= np.concatenate((training_transformed,scaled_))
                total_avg_reward.append(rm(train_test,200))
            mean = np.mean(total_avg_reward,axis=0)
            stand = np.std(total_avg_reward,axis=0)/np.sqrt(len(total_avg_reward))
            print(len(mean))
            #for j in total_avg_reward:
                #ax[e,r].plot(j)
            ax[e,1].set_ylim([0,1.1])
            ax[e,1].set_yticks([0,1])
            ax[e,1].set_yticklabels([0,100])
            ax[e,1].set_ylabel('Performance \n(% Optimal)')
            ax[e,1].plot(mean,color=col_to_plot[rep])
            ax[e,1].fill_between(np.arange(len(mean)), mean-stand,mean+stand, color=col_to_plot[rep] ,alpha=0.2)
            #ax[0,r].set_title(rep)
    plt.savefig('../figures/CH1/perceptron_FC.svg')
    plt.show()

#plot_perceptron(df,envs_to_plot,reps_to_plot)

def plot_shallow(df,envs_to_plot, reps_to_plot,):
    grids = get_grids(envs_to_plot)
    fig, ax = plt.subplots(len(envs_to_plot),2,sharey='col', sharex='col')
    for e, env in enumerate(envs_to_plot):
        scaling_factor = analysis_specs['avg_max_rwd'][env+'1']
        if env[-1] == '5':
            rwd_colrow0 = (3,9)
            rwd_colrow1= (16,9)
        else:
            rwd_colrow0 = (5,5)
            rwd_colrow1=(14,14)

        rect0 = plt.Rectangle(rwd_colrow0, 1, 1, facecolor='gray',edgecolor=None, alpha=0.3)
        rect1 = plt.Rectangle(rwd_colrow1, 1, 1, facecolor='g', edgecolor=None,alpha=0.3)
        ax[e,0].pcolor(grids[e],cmap='bone_r',edgecolors='k', linewidths=0.1)
        ax[e,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[e,0].set_aspect('equal')
        ax[e,0].add_patch(rect0)
        ax[e,0].add_patch(rect1)
        ax[e,0].get_xaxis().set_visible(False)
        ax[e,0].get_yaxis().set_visible(False)
        ax[e,0].invert_yaxis()

        for r, rep in enumerate(reps_to_plot):
            id_list = list(df_gb.get_group((env,rep,25000)))
            print(env, rep, len(id_list))
            total_avg_reward = []
            for i, id_num in enumerate(id_list):
                with open(parent_path+ f'results/{id_num}_data.p', 'rb') as f:
                    dats = pickle.load(f)
                    scaled_  = (np.asarray(dats['total_reward'])+2.5)/(scaling_factor+2.5)
                    total_avg_reward.append(rm(scaled_,200))
            mean = np.mean(total_avg_reward,axis=0)
            stand = np.std(total_avg_reward,axis=0)/np.sqrt(len(total_avg_reward))
            print(len(mean))
            ax[e,1].set_ylim([0,1.1])
            ax[e,1].set_yticks([0,1])
            ax[e,1].set_yticklabels([0,100],fontsize=8)
            ax[e,1].set_ylabel('Performance \n(% Optimal)',fontsize=8)
            ax[e,1].plot(mean,color=col_to_plot[rep])
            ax[e,1].fill_between(np.arange(len(mean)), mean-stand,mean+stand, color=col_to_plot[rep] ,alpha=0.2)
    ax[e,1].set_xlabel('Episodes')
    plt.savefig('../figures/CH1/shallow_FC.svg')
    plt.show()


plot_shallow(df, envs_to_plot,reps_to_plot)