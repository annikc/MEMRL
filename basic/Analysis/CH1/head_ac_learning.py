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
df1 = pd.read_csv('../../Data/flat_ac_training.csv')
df2 = pd.read_csv('../../Data/ec_testing.csv')

ids1 = get_id_dict(df1)
ids2 = get_id_dict(df2)


env_names = ['gridworld:gridworld-v1','gridworld:gridworld-v4','gridworld:gridworld-v3', 'gridworld:gridworld-v5']
test_env_names = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
'''
grids = []
for ind, environment_to_plot in enumerate(env_names):
    env = gym.make(environment_to_plot)
    plt.close()
    grids.append(env.grid)
'''
fig, ax = plt.subplots(1,2)
env_number = 1
print(ids2[test_env_names[env_number]])

for ind, rep_str in enumerate(df2.representation.unique()):
    print(rep_str)
    av, sd = get_avg_std(ids2[test_env_names[env_number]][rep_str], cutoff=25000)
    ax[0].plot(av, label=rep_str)
    ax[0].fill_between(np.arange(len(av)),av-sd, av+sd, alpha=0.3)
    ax[1].bar(ind,np.mean(sd))

ax[0].legend(loc=0)
#plt.ylim([-4,12])
plt.show()





### JUNKYARD
'''
reps = df1.representation.unique()
print(reps)
fig, ax = plt.subplots(1,2, sharey=True)
env_number = 1
for rep_str in reps:

    av, sd = get_avg_std(ids1[env_names[env_number]][rep_str], cutoff=25000)
    ax[0].plot(av, label=rep_str)
    ax[0].fill_between(np.arange(len(av)),av-sd, av+sd, alpha=0.3)

    av, sd = get_avg_std(ids2[test_env_names[env_number]][rep_str], cutoff=25000)
    ax[1].plot(av, label=rep_str)
    ax[1].fill_between(np.arange(len(av)),av-sd, av+sd, alpha=0.3)

ax[0].legend(loc=0)
ax[1].legend(loc=0)
ax[0].set_ylim([-4,12])
plt.show()

''''''
reps = df1.representation.unique()
print(reps)
fig, ax = plt.subplots(1,2, sharey=True)
env_number = 1
for rep_str in reps:

    av, sd = get_avg_std(ids1[env_names[env_number]][rep_str], cutoff=25000)
    ax[0].plot(av, label=rep_str)
    ax[0].fill_between(np.arange(len(av)),av-sd, av+sd, alpha=0.3)

    av, sd = get_avg_std(ids2[test_env_names[env_number]][rep_str], cutoff=25000)
    ax[1].plot(av, label=rep_str)
    ax[1].fill_between(np.arange(len(av)),av-sd, av+sd, alpha=0.3)

ax[0].legend(loc=0)
ax[1].legend(loc=0)
ax[0].set_ylim([-4,12])
plt.show()

'''

'''
plot_all = True

if plot_all:
    fig, axs = plt.subplots(4, 2, sharex='col')
    #rect = plt.Rectangle((5,5), 1, 1, color='r', alpha=0.3)
    for i in range(len(grids)):
        rect = plt.Rectangle((5,5), 1, 1, color='g', alpha=0.3)
        axs[i,0].pcolor(grids[i],cmap='bone_r',edgecolors='k', linewidths=0.1)
        axs[i,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        axs[i,0].set_aspect('equal')
        axs[i,0].add_patch(rect)
        axs[i,0].invert_yaxis()

    for ind, name in enumerate(env_names):
        for rep_to_plot in ['conv']: #reps:
            v_list = master_dict[name][rep_to_plot]
            avg_, std_ = get_avg_std(v_list)
            axs[ind,1].plot(avg_, label=f'{rep_to_plot}')
            axs[ind,1].fill_between(np.arange(len(avg_)),avg_-std_, avg_+std_, alpha=0.3)
        if ind == len(env_names)-1:
            axs[ind,1].set_xlabel('Episodes')
            axs[ind,1].set_ylabel('Cumulative \nReward')
    plt.savefig('../figures/CH1/conv_training.png')

plt.show()
'''