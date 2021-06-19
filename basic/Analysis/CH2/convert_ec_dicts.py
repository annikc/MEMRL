import gym
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from modules.Agents.RepresentationLearning.learned_representations import sr, onehot
from Analysis.analysis_utils import analysis_specs
from modules.Utils.gridworld_plotting import plot_pref_pol, plot_polmap


# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'ec_track_pols.csv')
groups_to_split = ['env_name','representation','EC_cache_limit']
gb = df.groupby(groups_to_split)["save_id"]

rep_dict = {'analytic successor':sr, 'onehot':onehot}
cache_limits = analysis_specs['cache_limits']
env_name = 'gridworld:gridworld-v31'
reps = ['onehot', 'analytic successor']
rep = reps[1]
pct = 25

env = gym.make(env_name)
plt.close()

state_reps, _, __, ___ = rep_dict[rep](env)
if rep == 'analytic successor':
    for s1 in env.obstacle:
        state_reps.pop(s1)

run_id = list(gb.get_group((env_name,rep,cache_limits[env_name][pct])))[0]
print(run_id)

with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
    data = pickle.load(f)

def get_ec_policy_map(data, trial_num, full=True):
    ec_pol_grid = np.zeros(env.shape, dtype=[(x, 'f8') for x in env.action_list])

    blank_mem = Memory(cache_limit=400, entry_size=4)
    blank_mem.cache_list = data['ec_dicts'][trial_num]

    if full:
        for k in state_reps.keys():
            twoD = env.oneD2twoD(k)
            sr_rep = state_reps[k]
            pol = blank_mem.recall_mem(sr_rep)

            ec_pol_grid[twoD] = tuple(pol)
    else:
        for k in blank_mem.cache_list.keys():
            oneDstate = blank_mem.cache_list[k][2]
            twoD = env.oneD2twoD(oneDstate)
            sr_rep = state_reps[oneDstate]
            pol = blank_mem.recall_mem(sr_rep)

            ec_pol_grid[twoD] = tuple(pol)

    return ec_pol_grid

ep_map = []
for x in range(10,20):#range(len(data['ec_dicts'])):
    print(x)
    ecpm = get_ec_policy_map(data,x,full=False)
    ep_map.append(ecpm)
    plot_polmap(env,ecpm)



print('hello')