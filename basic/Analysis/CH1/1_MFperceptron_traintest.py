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
df_train = pd.read_csv(parent_path+'train_perceptron_AC.csv')
df_test  = pd.read_csv(parent_path+'test_perceptron_AC.csv')

df_train['representation'] = df_train['representation'].apply(structured_unstructured)
df_test['representation'] = df_test['representation'].apply(structured_unstructured)
groups_to_split = ['env_name','representation','num_trials']

train_gb = df_train.groupby(groups_to_split)["save_id"]
test_gb = df_test.groupby(groups_to_split)["save_id"]

envs_to_plot = ['gridworld:gridworld-v1','gridworld:gridworld-v4','gridworld:gridworld-v3','gridworld:gridworld-v5']
grids = get_grids(envs_to_plot)

fig, ax = plt.subplots(len(envs_to_plot),2,sharex=True, sharey=True)
for e, env in enumerate(envs_to_plot):
    for r, rep in enumerate(['unstructured','structured']):
        train_avg_reward = []
        test_avg_reward = []
        train_ids = list(train_gb.get_group((env,rep,5000)))
        test_ids = list(test_gb.get_group((env+'1',rep,10000)))
        print(env,rep,len(train_ids),len(test_ids))
        ax[0,r].set_title(f'{rep}')
        for i, id_num in enumerate(train_ids):
            with open(parent_path+ f'results/{id_num}_data.p', 'rb') as f:
                dats = pickle.load(f)
                raw_score = dats['total_reward']
                normalization = analysis_specs['avg_max_rwd'][env+'1']
                transformed = (np.asarray(raw_score)+2.5)/(normalization +2.5)
                train_avg_reward.append(transformed)
        train_mean  = np.mean(train_avg_reward,axis=0)
        train_maxes = train_mean+np.std(train_avg_reward,axis=0)/np.sqrt(len(train_avg_reward))
        train_mins  = train_mean-np.std(train_avg_reward,axis=0)/np.sqrt(len(train_avg_reward))

        for i, id_num in enumerate(test_ids):
            with open(parent_path+ f'results/{id_num}_data.p', 'rb') as f:
                dats = pickle.load(f)
                raw_score = dats['total_reward']
                normalization = analysis_specs['avg_max_rwd'][env+'1']
                transformed = (np.asarray(raw_score)+2.5)/(normalization +2.5)
                test_avg_reward.append(transformed)
        test_mean  = np.mean(test_avg_reward,axis=0)
        test_maxes = test_mean+np.std(test_avg_reward,axis=0)/np.sqrt(len(test_avg_reward))
        test_mins  = test_mean-np.std(test_avg_reward,axis=0)/np.sqrt(len(test_avg_reward))

        mean = rm(np.concatenate((train_mean,test_mean)),200)
        print(len(mean))
        maxes = rm(np.concatenate((train_maxes,test_maxes)),200)
        mins  = rm(np.concatenate((train_mins,test_mins)),200)

        ax[e,r].axvline(x=5000, linestyle=":",color='gray')
        ax[e,r].plot(np.arange(len(mean)),mean,LINCLAB_COLS['blue'])
        ax[e,r].fill_between(np.arange(len(mean)),mins,maxes,color=LINCLAB_COLS['blue'], alpha=0.2)
    ax[e,r].set_ylim(0,1.1)
plt.savefig(f'../figures/CH1/MF_perceptron_traintest.svg')
plt.show()

