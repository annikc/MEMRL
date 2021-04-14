









import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import gym
import pandas as pd

sys.path.append('../../modules/')
from Utils import running_mean as rm
from Analysis.analysis_utils import get_avg_std, get_id_dict

data_dir = '../../Data/results/'
df2 = pd.read_csv('../../Data/ec_testing.csv')
ids2 = get_id_dict(df2)


env_names = ['gridworld:gridworld-v1','gridworld:gridworld-v4','gridworld:gridworld-v3', 'gridworld:gridworld-v5']
test_env_names = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']

fig, ax = plt.subplots(4,1)
env_number = 1
print(ids2[test_env_names[env_number]])
list_of_representations = ['saved_latents']
for ind, rep_str in enumerate(list_of_representations):
    print(rep_str)
    av, sd = get_avg_std(ids2[test_env_names[env_number]][rep_str], cutoff=5000)
    ax[0].plot(av, label=rep_str)
    ax[0].fill_between(np.arange(len(av)),av-sd, av+sd, alpha=0.3)
    #ax[1].bar(ind,np.mean(sd))

ax[0].legend(loc=0)
#plt.ylim([-4,12])
plt.show()



df = pd.read_csv('../../Data/head_only_retrain.csv')

master_dict = {}
envs = df.env_name.unique()
reps = df.representation.unique()

for env in envs:
    master_dict[env] = {}
    for rep in reps:
        id_list = list(df.loc[(df['env_name']==env)
         & (df['representation']==rep)]['save_id'])

        master_dict[env][rep]=id_list

env_names = ['gridworld:gridworld-v11', 'gridworld:gridworld-v41','gridworld:gridworld-v31', 'gridworld:gridworld-v51']
grids = []
for ind, environment_to_plot in enumerate(env_names):
    env = gym.make(environment_to_plot)
    plt.close()
    grids.append(env.grid)


plot_all = True
if plot_all:
    fig, axs = plt.subplots(4, 1, sharex=True)

    for ind, name in enumerate(env_names):
        for rep_to_plot in reps:
            v_list = master_dict[name][rep_to_plot]
            avg_, std_ = get_avg_std(v_list,cutoff=5000,smoothing=50)
            axs[ind].plot(avg_, label=f'{rep_to_plot}')
            axs[ind].fill_between(np.arange(len(avg_)),avg_-std_, avg_+std_, alpha=0.3)
            axs[ind].set_ylim(-4,12)

            list_of_representations = ['saved_latents']
            for rep_str in list_of_representations:
                print(rep_str)
                av, sd = get_avg_std(ids2[test_env_names[ind]][rep_str], cutoff=5000,smoothing=50)
                axs[ind].plot(av, label=rep_str)
                axs[ind].fill_between(np.arange(len(av)),av-sd, av+sd, alpha=0.3)
        if ind == len(env_names)-1:
            axs[ind].set_xlabel('Episodes')
            axs[ind].set_ylabel('Cumulative \nReward')
    #plt.savefig('../figures/CH1/compare_mf_ec.png')

plt.show()
