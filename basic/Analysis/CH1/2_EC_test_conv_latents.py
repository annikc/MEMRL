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
df = pd.read_csv(parent_path+'train_test_ec.csv')
df['representation'] = df['representation'].apply(structured_unstructured)

ref_df = pd.read_csv(parent_path+'conv_mf_retraining.csv')

def chop_(arr):
    smoothing = 20
    start = 5000-smoothing+1
    return np.concatenate((arr[0:start],arr[5000:]))


envs_to_plot = ['gridworld:gridworld-v1','gridworld:gridworld-v4','gridworld:gridworld-v3','gridworld:gridworld-v5']
pcts_to_plot = [100,75,50,25]
reps_to_plot = ['unstructured','structured']
grids = get_grids(envs_to_plot)
col_to_plot = {'unstructured':LINCLAB_COLS['blue'], 'structured':LINCLAB_COLS['red'], 'conv':LINCLAB_COLS['purple']}

print(df.representation.unique())

def shit_to_plot(df, env, rep, end_index=-1):
    df_gb = df.groupby(['env_name','representation','extra_info'])["save_id"]
    total_avg_reward = []

    id_list = list(df_gb.get_group((env,rep,'x')))
    print(env,rep,len(id_list))
    for i, id_num in enumerate(id_list):
        with open(parent_path+ f'results/{id_num}_data.p', 'rb') as f:
            dats = pickle.load(f)
            raw_score = dats['total_reward'][:end_index]
            normalization = analysis_specs['avg_max_rwd'][env[0:22]]
            scaled_ = (np.asarray(raw_score)+2.5)/(normalization +2.5)
            total_avg_reward.append(rm(scaled_,smoothing))

    mean  = np.nanmean(total_avg_reward,axis=0)
    print(len(mean),'meannnnn')
    maxes = mean+np.nanstd(total_avg_reward,axis=0)/np.sqrt(len(total_avg_reward))
    mins  = mean-np.nanstd(total_avg_reward,axis=0)/np.sqrt(len(total_avg_reward))

    return mean, maxes, mins

rep = 'conv'
fig, ax = plt.subplots(len(envs_to_plot),2,sharex='col', sharey='col')
for e, env in enumerate(envs_to_plot):
    smoothing = 20
    upper_limit= 30000
    ftsz=8

    if env[-1] == '5':
        rwd_colrow0 = (3,9)
        rwd_colrow1= (16,9)
    else:
        rwd_colrow0 = (5,5)
        rwd_colrow1=(14,14)

    rect0 = plt.Rectangle(rwd_colrow0, 1, 1, facecolor='gray',edgecolor=None, alpha=0.6)
    rect1 = plt.Rectangle(rwd_colrow1, 1, 1, facecolor='g', edgecolor=None,alpha=0.3)
    ax[e,0].pcolor(grids[envs_to_plot.index(env)],cmap='bone_r',edgecolors='k', linewidths=0.1)
    ax[e,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
    ax[e,0].set_aspect('equal')
    ax[e,0].add_patch(rect0)
    ax[e,0].add_patch(rect1)
    ax[e,0].get_xaxis().set_visible(False)
    ax[e,0].get_yaxis().set_visible(False)
    ax[e,0].invert_yaxis()

    ## training means
    train_df = pd.read_csv(parent_path+'conv_mf_training.csv')
    print('Training Data')
    mean, maxes, mins = shit_to_plot(train_df,env,rep,end_index=5000)
    ax[e,1].plot(np.arange(len(mean)),mean,'k',alpha=0.7)
    ax[e,1].fill_between(np.arange(len(mean)),mins,maxes,color='k', alpha=0.2)
    end_of_training = mean[-1]
    print('testing MF')
    mean, maxes, mins = shit_to_plot(ref_df,env+'1',rep)
    ax[e,1].axvline(x=5000-smoothing+1, linestyle=":",color='gray')
    ax[e,1].plot(np.arange(len(mean))+5000-smoothing-1,mean,'k',alpha=0.7)
    ax[e,1].fill_between(np.arange(len(mean))+5000-smoothing-1,mins,maxes,color='k', alpha=0.2)
    start_of_testing = mean[0]
    ax[e,1].plot([5000-smoothing-1,5000-smoothing],[end_of_training,start_of_testing],color='k',alpha=0.7)
    ax[e,1].set_ylim(0,1.1)
    ax[e,1].set_yticks([0,1])
    ax[e,1].set_yticklabels([0,100],fontsize=ftsz)

    print('EC data')
    mean, maxes, mins = shit_to_plot(df,env+'1',rep)
    ax[e,1].axvline(x=5000-smoothing+1, linestyle=":",color='gray')
    ax[e,1].plot(np.arange(len(mean))+5000-smoothing-1,mean,col_to_plot[rep])
    ax[e,1].fill_between(np.arange(len(mean))+5000-smoothing-1,mins,maxes,color=col_to_plot[rep], alpha=0.2)

    ax[e,1].set_ylim(0,1.1)
    ax[e,1].set_yticks([0,1])
    ax[e,1].set_yticklabels([0,100],fontsize=ftsz)
    ax[e,1].set_ylabel('Performance \n(% Optimal)',fontsize=ftsz)


ax[e,1].set_xlim([5000-smoothing-50,5000+100])
#ax[e,1].set_xlabel('Episodes', fontsize=ftsz)
#ax[e,1].set_xticks([0,10000,20000,30000])
#ax[e,1].set_xticklabels([0,10000,20000,30000],fontsize=ftsz)
plt.savefig(f'../figures/CH1/EC_latents_inset.svg')
plt.show()

