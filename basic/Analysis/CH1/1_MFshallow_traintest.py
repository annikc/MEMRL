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

df['representation'] = df['representation'].apply(structured_unstructured)
groups_to_split = ['env_name','representation','num_trials']
df_gb = df.groupby(groups_to_split)["save_id"]

envs_to_plot = ['gridworld:gridworld-v1','gridworld:gridworld-v4','gridworld:gridworld-v3','gridworld:gridworld-v5']
pcts_to_plot = [100,75,50,25]
reps_to_plot = ['unstructured','structured']
grids = get_grids(envs_to_plot)

env = envs_to_plot[0]
pct = 100
rep = 'structured'
fig, ax = plt.subplots(len(envs_to_plot),2,sharex=True, sharey=True)
for e, env in enumerate(envs_to_plot):
    for r, rep in enumerate(reps_to_plot):
        id_list = list(df_gb.get_group((env,rep,15000)))
        print(env, rep, len(id_list))
        total_avg_reward = []
        ax[0,r].set_title(f'{rep}')
        for i, id_num in enumerate(id_list):
            with open(parent_path+ f'results/{id_num}_data.p', 'rb') as f:
                dats = pickle.load(f)
                raw_score = dats['total_reward']
                normalization = analysis_specs['avg_max_rwd'][env+'1']
                transformed = rm((np.asarray(raw_score)+2.5)/(normalization +2.5),200)
                total_avg_reward.append(transformed)
                #ax[e,r].plot(transformed,label=f'{id_num[0:8]}')

        mean  = np.mean(total_avg_reward,axis=0)
        maxes = mean+np.std(total_avg_reward,axis=0)/np.sqrt(len(total_avg_reward))
        mins  = mean-np.std(total_avg_reward,axis=0)/np.sqrt(len(total_avg_reward))
        ax[e,r].axvline(x=5000, linestyle=":",color='gray')
        ax[e,r].plot(np.arange(len(mean)),mean,LINCLAB_COLS['green'])
        ax[e,r].fill_between(np.arange(len(mean)),mins,maxes,color=LINCLAB_COLS['green'], alpha=0.2)
    ax[e,r].set_ylim(0,1.1)

#plt.savefig(f'../figures/CH1/MFshallow_traintest.svg')
plt.show()

