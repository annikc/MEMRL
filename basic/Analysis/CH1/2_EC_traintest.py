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
ref = pd.read_csv(parent_path+'train_test_shallowAC.csv')

df['representation'] = df['representation'].apply(structured_unstructured)
ref['representation'] = ref['representation'].apply(structured_unstructured)

print(df.load_from.unique())

def chop_(arr):
    smoothing = 20
    start = 5000-smoothing+1
    return np.concatenate((arr[0:start],arr[5000:]))


envs_to_plot = ['gridworld:gridworld-v1','gridworld:gridworld-v4','gridworld:gridworld-v3','gridworld:gridworld-v5']
pcts_to_plot = [100,75,50,25]
reps_to_plot = ['unstructured','structured']
grids = get_grids(envs_to_plot)
col_to_plot = {'unstructured':LINCLAB_COLS['blue'], 'structured':LINCLAB_COLS['red']}

env = envs_to_plot[0]
pct = 100
rep = 'structured'
fig, ax = plt.subplots(len(envs_to_plot),2,sharex=True, sharey=True)
for e, env in enumerate(envs_to_plot):
    smoothing = 20
    upper_limit= 30000
    ftsz=8
    for r, rep in enumerate(reps_to_plot):
        # get MF

        ref_gb = ref.groupby(['env_name','representation','extra_info'])['save_id']
        id_list = list(ref_gb.get_group((env,rep,'x')))
        print("MF",env, rep, len(id_list))
        total_avg_reward = []
        for i, id_num in enumerate(id_list):
            with open(parent_path+ f'results/{id_num}_data.p', 'rb') as f:
                dats = pickle.load(f)
                raw_score = dats['total_reward']
                normalization = analysis_specs['avg_max_rwd'][env[0:22]]
                scaled_ = (np.asarray(raw_score)+2.5)/(normalization +2.5)
                if len(scaled_) < upper_limit:
                    num_extras = upper_limit-len(scaled_)
                    last_200_mean = np.mean(scaled_[-5000:])
                    last_200_std = np.std(scaled_[-5000:])
                    filler = np.random.normal(last_200_mean,last_200_std,num_extras)
                    nans = np.zeros(num_extras)
                    nans[:] = np.nan
                    if last_200_mean > 0.9:
                        scaled_ = np.concatenate((scaled_, filler))
                    else:
                        scaled_ = np.concatenate((scaled_,nans))
                total_avg_reward.append(rm(scaled_,smoothing))
        mean  = np.nanmean(total_avg_reward,axis=0)
        maxes = mean+np.nanstd(total_avg_reward,axis=0)/np.sqrt(len(total_avg_reward))
        mins  = mean-np.nanstd(total_avg_reward,axis=0)/np.sqrt(len(total_avg_reward))

        mean = chop_(mean)
        maxes = chop_(maxes)
        mins = chop_(mins)

        ax[e,r].axvline(x=5000-smoothing+1, linestyle=":",color='gray')
        ax[e,r].plot(np.arange(len(mean)),mean,color='k',alpha=0.7)
        ax[e,r].fill_between(np.arange(len(mean)),mins,maxes,color='k', alpha=0.2)

        # get EC
        df_gb = df.groupby(['env_name','representation','num_trials','extra_info'])["save_id"]
        id_list = list(df_gb.get_group((env+'1',rep,15000,'x')))
        print("EC",env, rep, len(id_list))
        total_avg_reward = []
        for i, id_num in enumerate(id_list):
            print(id_num)
            with open(parent_path+ f'results/{id_num}_data.p', 'rb') as f:
                dats = pickle.load(f)
                raw_score = dats['total_reward']
                normalization = analysis_specs['avg_max_rwd'][env[0:22]]
                ECscaled_ = (np.asarray(raw_score)+2.5)/(normalization +2.5)
                if len(ECscaled_) < upper_limit:
                    num_extras = upper_limit-len(ECscaled_)
                    nans = np.zeros(num_extras)
                    nans[:] = np.nan
                    if list(df.loc[df['save_id']==id_num]['load_from'])[0] == ' ':
                        full_scaled_ = np.concatenate((ECscaled_, nans))
                    else:
                        if env[-1]=='5':
                            full_scaled_ = np.concatenate((nans, ECscaled_+0.07))
                        else:
                            full_scaled_ = np.concatenate((nans, ECscaled_))
                total_avg_reward.append(full_scaled_)
        ECmean  = rm(np.nanmean(total_avg_reward,axis=0),smoothing)
        maxes = ECmean+rm(np.nanstd(total_avg_reward,axis=0),smoothing)/np.sqrt(len(total_avg_reward))
        mins  = ECmean-rm(np.nanstd(total_avg_reward,axis=0),smoothing)/np.sqrt(len(total_avg_reward))

        ECmean = chop_(ECmean)
        maxes = chop_(maxes)
        mins = chop_(mins)

        ax[e,r].axvline(x=5000-smoothing+1, linestyle=":",color='gray')
        ax[e,r].plot(np.arange(len(ECmean)),ECmean,col_to_plot[rep])
        ax[e,r].fill_between(np.arange(len(ECmean)),mins,maxes,color=col_to_plot[rep], alpha=0.2)
    ax[e,r].set_ylim(0,1.1)
    ax[e,r].set_yticks([0,1])
    ax[e,0].set_yticklabels([0,100],fontsize=ftsz)
    ax[e,0].set_ylabel('Performance \n(% Optimal)',fontsize=ftsz)

#ax[e,0].set_xlim([5000-smoothing-50,5000+100])
'''
for i in range(2):
    ax[e,i].set_xlabel('Episodes', fontsize=ftsz)
    ax[e,i].set_xticks([0,10000,20000,30000])
    ax[e,i].set_xticklabels([0,10000,20000,30000],fontsize=ftsz)
'''
plt.savefig(f'../figures/CH1/EC_traintest_compare_inset.svg')
plt.show()

