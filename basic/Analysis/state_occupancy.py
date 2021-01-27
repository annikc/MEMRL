import gym
import pickle
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import colorsys
import os
from basic.modules.Utils import running_mean as rm
from basic.modules.Utils.gridworld_plotting import plot_polmap, plot_pref_pol, plot_valmap, plot_world

run_id = 'f6b73222-00f1-4b68-823f-b0304e9327ca'
env_id = 'gym_grid:gridworld-v1'

env = gym.make(env_id)
plt.close()

with open(f'../Data/results/{run_id}_data.p', 'rb') as f:
    data = pickle.load(f)
for key in data.keys():
    print(key, len(data[key]))

states_visited = np.array(data['mf_visits'])
n_trials = states_visited.shape[0]


all_visits = np.zeros((n_trials, env.r, env.c))


def trial_map(trial):
    visit_map = np.zeros((env.shape))
    for i in range(env.nstates):
        coord = env.oneD2twoD(i)
        visit_map[coord[0],coord[1]] = states_visited[trial,i]
    return visit_map


for i in range(n_trials):
    trialmap = trial_map(i)
    if i == 0:
        all_visits[i] = trialmap.copy()
    else:
        all_visits[i] = trialmap.copy()

def plot_traj():
    for i in range(n_trials):
        plt.figure()
        m = plt.imshow(all_visits[i])
        plt.colorbar(m)
        #plt.savefig(f'./figures/maps/occupancy/{i}.svg',format='svg')
        plt.close()

index = 1
plt.figure()
m = plt.imshow(all_visits[index])
plt.colorbar(m)
plt.show()
plt.close()
