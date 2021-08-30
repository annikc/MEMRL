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
def chop_(arr):
    smoothing = 20
    start = 5000-smoothing+1
    return np.concatenate((arr[0:start],arr[5000:]))


envs_to_plot = ['gridworld:gridworld-v1','gridworld:gridworld-v4','gridworld:gridworld-v3','gridworld:gridworld-v5']
pcts_to_plot = [100,75,50,25]
reps_to_plot = ['unstructured','structured']
grids = get_grids(envs_to_plot)
col_to_plot = {'unstructured':LINCLAB_COLS['blue'], 'structured':LINCLAB_COLS['red'], 'conv_latents':LINCLAB_COLS['purple']}

print(df.representation.unique())

fig, ax = plt.subplots(len(envs_to_plot),2,sharex=True, sharey=True)
for e, env in enumerate(envs_to_plot):
    smoothing = 200
    upper_limit= 30000
    ftsz=8

    df_gb = df.groupby(['env_name','representation'])["save_id"]
    total_avg_reward = []
    id_list = list(df_gb.get_group((env+'1','conv_latents')))
    for i, id_num in enumerate(id_list):
        with open(parent_path+ f'results/{id_num}_data.p', 'rb') as f:
            dats = pickle.load(f)
            raw_score = dats['total_reward']
            normalization = analysis_specs['avg_max_rwd'][env+'1']
            scaled_ = (np.asarray(raw_score)+2.5)/(normalization +2.5)
            total_avg_reward.append(rm(scaled_,smoothing))

    mean  = np.nanmean(total_avg_reward,axis=0)
    print(len(mean),'meannnnn')
    maxes = mean+np.nanstd(total_avg_reward,axis=0)/np.sqrt(len(total_avg_reward))
    mins  = mean-np.nanstd(total_avg_reward,axis=0)/np.sqrt(len(total_avg_reward))

    ax[e,1].axvline(x=5000-smoothing+1, linestyle=":",color='gray')
    ax[e,1].plot(np.arange(len(mean))+5000-smoothing-1,mean,col_to_plot['conv_latents'])
    ax[e,1].fill_between(np.arange(len(mean))+5000-smoothing-1,mins,maxes,color=col_to_plot['conv_latents'], alpha=0.2)

    ax[e,1].set_ylim(0,1.1)
    ax[e,1].set_yticks([0,1])
    ax[e,1].set_yticklabels([0,100],fontsize=ftsz)
    ax[e,1].set_ylabel('Performance \n(% Optimal)',fontsize=ftsz)

#ax[e,0].set_xlim([5000-smoothing-50,5000+100])
for i in range(2):
    ax[e,i].set_xlabel('Episodes', fontsize=ftsz)
    ax[e,i].set_xticks([0,10000,20000,30000])
    ax[e,i].set_xticklabels([0,10000,20000,30000],fontsize=ftsz)
plt.savefig(f'../figures/CH1/EC_latents.svg')
plt.show()

