import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Analysis.analysis_utils import structured_unstructured, LINCLAB_COLS, plot_specs, analysis_specs, convert_pol_array_to_pref_dir, fade
from modules.Utils import running_mean as rm
from modules.Utils.gridworld_plotting import plot_pref_pol, plot_valmap
import torch
import gym
import numpy as np
from modules.Agents.RepresentationLearning.learned_representations import onehot, random, place_cell, sr, latents
import modules.Agents.Networks as nets
import modules.Agents.EpisodicMemory as Memory

from modules.Agents import Agent


# get data frame
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'bootstrapped_retrain_shallow_AC.csv')
df['representation'] = df['representation'].apply(structured_unstructured)
groups_to_split = ['env_name','representation','EC_cache_limit','num_trials']
gb = df.groupby(groups_to_split)["save_id"]

cache_limits = analysis_specs['cache_limits']

## get agent ids matching specs

env_name = 'gridworld:gridworld-v41'
env      = gym.make(env_name)
plt.close()
rep      = 'structured'
pct      = 100
id_list  = list(gb.get_group((env_name, rep, int(cache_limits[env_name][100]*(pct/100)),15000)))
print(id_list)
load_from = np.random.choice(id_list)
print(load_from)
nums = {25:'0c9cf76e-8994-4052-b2e7-3fa59d53a6c5',
        50:'0219483c-54b9-4879-82e9-a7f05879262a',
        75:'6c7df218-8f3a-47a3-b9c2-00c3b6820e2b',
        100:'6d41b20a-056d-4286-b631-17f81896b83f'
        }
load_from = nums[pct]
pct  = 'mf'
load_from = 'b79ef920-1c53-4f49-ae74-153f71be5126'

with open(f'../../Data/results/{load_from}_data.p', 'rb') as f:
    data = pickle.load(f)

def plot_pref_dir(data):
    indices = [5000, 10000, 14999]
    colors = {100:LINCLAB_COLS['red'], 75: LINCLAB_COLS['orange'], 50:LINCLAB_COLS['green'], 25:LINCLAB_COLS['purple'], 'mf':'black'}
    fig, ax = plt.subplots(2,len(indices))
    gs = ax[0,0].get_gridspec()
    for a in ax[0,:]:
        a.remove()
    big_ax = fig.add_subplot(gs[0,:])
    if pct =='mf':
        raw_score = data['total_reward'][5000:20000]
    else:
        raw_score = data['bootstrap_reward'][0:15000]
    normalization = analysis_specs['avg_max_rwd'][env_name[0:22]]
    transformed = rm((np.asarray(raw_score)+2.5)/(normalization +2.5) , 100)
    big_ax.plot(transformed,color=colors[pct])
    big_ax.set_ylim([-0.1,1.1])
    for i, index in enumerate(indices):
        if pct =='mf':
            print(len(data['P_snap']))
            index= [75, 100, 149][i]
            print(index)
            pol_array = data['P_snap'][index].flatten()
        else:
            pol_array = data['P_snap'][index].flatten()
        for x in env.obstacle:
            pol_array[x] = tuple(np.zeros(4))
        dir_map = convert_pol_array_to_pref_dir(pol_array.flatten())
        a = ax[1,i].imshow(dir_map,cmap=fade,vmin=0, vmax=360)

        old_loc = plt.Rectangle((4.49,4.47), width=1, height=1, edgecolor='w',fill=False, linestyle='--')
        new_loc = plt.Rectangle((13.49,13.47), width=1, height=1, edgecolor='w',fill=False, linestyle='-')
        ax[1,i].add_patch(old_loc)
        ax[1,i].add_patch(new_loc)
        ax[1,i].get_xaxis().set_visible(False)
        ax[1,i].get_yaxis().set_visible(False)
    plt.colorbar(a, ax=ax[1,2])
    plt.savefig(f'../figures/CH3/pref_dir_{rep}_{pct}.svg')
    plt.show()

def plot_values(data):
    '''
    VERY QUESTIONABLE RESULTS
    :param data:
    :return:
    '''
    indices = [5000, 10000, 14999]
    fig, ax = plt.subplots(1,len(indices))
    for i, index in enumerate(indices):
        val_array = data['V_snap'][index]
        for item in env.obstacle:
            coord = env.oneD2twoD(item)
            val_array[coord] = np.nan
        a = ax[i].imshow(val_array,cmap='viridis',vmin=-2.49, vmax=5)

        old_loc = plt.Rectangle((4.49,4.47), width=1, height=1, edgecolor='w',fill=False, linestyle='--')
        new_loc = plt.Rectangle((13.49,13.47), width=1, height=1, edgecolor='w',fill=False, linestyle='-')
        ax[i].add_patch(old_loc)
        ax[i].add_patch(new_loc)
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
    plt.colorbar(a, ax=ax[2])
    plt.savefig(f'../figures/CH3/val_maps_{rep}_{pct}.svg')
    plt.show()

#plot_values(data)
plot_pref_dir(data)

## generate pol maps by getting policies for each state in state reps
#pol_array = np.zeros((20,20), dtype=[(x,'f8') for x in env.action_list])
#val_array = np.zeros((20,20))

#plot_pref_pol(env, pol_array,save=True,directory='../figures/CH3/',title=f'{env_name[-2:]}_{rep}_{pct}_5000')
#plot_valmap(env,val_array, v_range=[0,10],save=True, directory='../figures/CH3/',title=f'{env_name[-2:]}_{rep}_{pct}_5000')
