import gym
import pickle
import numpy as np
import matplotlib.pyplot as plt
from modules.Utils.gridworld_plotting import plot_pref_pol, plot_polmap
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from modules.Agents.RepresentationLearning.learned_representations import sr
from scipy.special import rel_entr
env = gym.make('gridworld:gridworld-v41')
plt.close()

run_id = 'c34add2d-7809-45b7-92e8-ab3af5673499'#'f50acce3-d20a-4eda-9d71-1817b2cb8056'#'5f850bfd-74c7-479c-9a64-b1b3ed4b624f'# v3 = 'f25f91b9-e641-4d2d-918e-2e29f4d742c0'#'7e7d324b-5605-44fd-a8bb-656f4f3eae89'

with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
    data = pickle.load(f)

n_visits = np.zeros(env.shape)
blank_mem = Memory(cache_limit=400, entry_size=4)
print(len(data['ec_dicts']))
probe_key = list(data['ec_dicts'][0].keys())[0]

ec_pol_grid = np.zeros(env.shape, dtype=[(x, 'f8') for x in env.action_list])
def get_avg_of_memories():
    n_visits = np.zeros(env.shape)
    ec_pol_grid = np.zeros(env.shape, dtype=[(x, 'f8') for x in env.action_list])
    for i in range(len(data['ec_dicts'])):
        print(i)
        blank_mem = Memory(cache_limit=400, entry_size=4)
        blank_mem.cache_list = data['ec_dicts'][i]
        states = []
        for k, key in enumerate(blank_mem.cache_list.keys()):
            twoD = env.oneD2twoD(blank_mem.cache_list[key][2])
            old_policy = ec_pol_grid[twoD]
            current_policy = blank_mem.recall_mem(key)

            average = []
            for x,y in zip(old_policy, current_policy):
                z = x + (y-x)/(k+1)
                average.append(z)
            ec_pol_grid[twoD] = tuple(average)

            n_visits[twoD]+=1


state_reps, _, __, __ =sr(env)
def get_KLD(probe_state,trial_num):
    probe_rep = state_reps[probe_state]

    KLD_array = np.zeros(env.shape)
    KLD_array[:] = np.nan
    ec_pol_grid = np.zeros(env.shape, dtype=[(x, 'f8') for x in env.action_list])

    blank_mem = Memory(cache_limit=400, entry_size=4)
    blank_mem.cache_list = data['ec_dicts'][trial_num]

    probe_pol = blank_mem.recall_mem(probe_rep)
    for k in state_reps.keys():
        print(k, 'state is ')
        sr_rep = state_reps[k]
        pol = blank_mem.recall_mem(sr_rep)
        twoD = env.oneD2twoD(k)
        KLD_array[twoD] = sum(rel_entr(list(probe_pol),list(pol)))
        ec_pol_grid[twoD] = tuple(pol)

    return KLD_array,ec_pol_grid

kld, ec_pols = get_KLD(env.twoD2oneD((5,10)), 400)
a = plt.imshow(kld)
plt.colorbar(a)
plt.show()

plot_pref_pol(env, ec_pols,threshold=0)
'''
for i in range(len(data['ec_dicts'])):
    blank_mem = Memory(cache_limit=400, entry_size=4)
    blank_mem.cache_list = data['ec_dicts'][i]
    ec_pol_grid = np.zeros(env.shape, dtype=[(x, 'f8') for x in env.action_list])
    for key in blank_mem.cache_list.keys():
        readable_state = blank_mem.cache_list[key][2]
        twoD = env.oneD2twoD(readable_state)
        ec_pol_grid[twoD] = tuple(blank_mem.recall_mem(key))
        n_visits[twoD] += 1 
    plot_pref_pol(env,ec_pol_grid)

'''