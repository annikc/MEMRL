import gym
import pickle
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt

from basic.modules.Utils import running_mean as rm
from basic.modules.Utils.gridworld_plotting import plot_polmap, plot_pref_pol, plot_valmap, plot_world
from basic.Analysis.vis_bootstrap_pol_maps import daves_idea, plot_pol_evol, trajectories, plot_maps, plot_rewards

filename = '../Data/linear_track.csv'
df = pd.read_csv(filename)
rewards = {'mf':{}, 'ecmf':{}}
for x in range(len(df)):
    run_id = df['run_id'].loc[x]
    lr = df['lr'].loc[x]
    if df['expt_type'].loc[x][0:9] == 'Bootstrap':
        dkey = 'ecmf'
    elif df['expt_type'].loc[x][0:9] == 'gridworld':
        dkey = 'mf'
    with open(f'../Data/results/{run_id}_data.p', 'rb') as f:
        data = pickle.load(f)

    rewards[dkey][str(lr)] = data['total_reward']

smoothing = 30
fig, ax = plt.subplots(2,1,sharex=True)
for i in rewards['mf'].keys():
    ax[0].plot(rm(rewards['mf'][i],smoothing), label = i)
for i in rewards['ecmf'].keys():
    ax[1].plot(rm(rewards['ecmf'][i],smoothing), label = i)
ax[0].legend(loc=0)
ax[1].legend(loc=0)
plt.show()


