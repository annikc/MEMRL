import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Analysis.analysis_utils import structured_unstructured, LINCLAB_COLS, plot_specs, analysis_specs, fade, convert_pol_array_to_pref_dir
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
rep      = 'structured'
pct      = 100
index    = 5000
id_list  = list(gb.get_group((env_name, rep, int(cache_limits[env_name][100]*(pct/100)),index)))
print(id_list)
load_from = np.random.choice(id_list)
print(load_from)
#load_from = 'f287c6b6-7404-41ad-85d8-2a5ca30bee5f'
nums = {25:'0c9cf76e-8994-4052-b2e7-3fa59d53a6c5',
        50:'0219483c-54b9-4879-82e9-a7f05879262a',
        75:'6c7df218-8f3a-47a3-b9c2-00c3b6820e2b',
        100:'6d41b20a-056d-4286-b631-17f81896b83f'
        }
load_from = nums[pct]

reps_for_reads = {'onehot':'onehot', 'sr':'analytic successor', 'place_cell':'place_cell','random':'random'}

# make gym environment
env = gym.make(env_name)
plt.close()
print(env.rewards)
rep_types = {'unstructured':onehot, 'random':random, 'place_cell':place_cell, 'structured':sr, 'latent':latents}
state_reps, representation_name, input_dims, _ = rep_types[rep](env)
print(env.obstacle)

# load weights to head_ac network from previously learned agent
AC_head_agent = nets.shallow_ActorCritic(input_dims, hidden_dims=200, output_dims=env.action_space.n, lr=0.005)


if load_from != None:
    AC_head_agent.load_state_dict(torch.load(parent_path+f'agents/{load_from}.pt'))
    print(f"weights loaded from {load_from}")

memory = Memory.EpisodicMemory(cache_limit=400, entry_size=env.action_space.n)
agent = Agent(AC_head_agent, memory=memory, state_representations=state_reps)


## generate pol maps by getting policies for each state in state reps
pol_array = np.zeros((20,20), dtype=[(x,'f8') for x in env.action_list])
val_array = np.zeros((20,20))
for s in state_reps:
    p,v = AC_head_agent(state_reps[s])
    coord = env.oneD2twoD(s)
    print(coord)
    pol_array[coord] = tuple(p.detach().numpy())
    val_array[coord] = v.item()

#plot_pref_pol(env, pol_array,save=True,directory='../figures/CH3/',title=f'{env_name[-2:]}_{rep}_{pct}_15000')
#plot_valmap(env,val_array, v_range=[-2,10],save=True, directory='../figures/CH3/bits_and_bobs/',title=f'{env_name[-2:]}_{rep}_{pct}',cmap=plt.cm.viridis)

pref_dir = convert_pol_array_to_pref_dir(pol_array.flatten())
plt.imshow(pref_dir,cmap=fade)

#plt.savefig(f'../figures/CH3/bits_and_bobs/pref_dir_MF_{index}.svg')
plt.show()