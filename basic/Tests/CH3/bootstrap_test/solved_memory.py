import torch
import gym
import numpy as np
import modules.Agents.Networks as nets
import modules.Agents.EpisodicMemory as Memory
from modules.Agents import Agent
from modules.Experiments import gridworldBootstrap as expt
import matplotlib.pyplot as plt
from basic.modules.Utils.gridworld_plotting import plot_polmap, plot_pref_pol, plot_valmap, plot_world

env_name   = 'gridworld:gridworld-v111'
network_id = None # '97b5f281-a60e-4738-895d-191a04edddd6'
ntrials    = 1000

def EC_pass():
    pass

def correct_actions(coord):
    actions = []
    rwd_loc = list(env.rewards)[0]
    rwd_r, rwd_c = rwd_loc
    coord_r, coord_c = coord

    r_dif = rwd_r - coord_r
    c_dif = rwd_c - coord_c

    if r_dif < 0:
        actions.append(1)
    elif r_dif > 0:
        actions.append(0)

    if c_dif < 0:
        actions.append(3)
    elif c_dif > 0:
        actions.append(2)

    return actions


# create environment
env = gym.make(env_name)
plt.close()

# generate network
if network_id == None:
    # generate parameters for network from environment observation shape
    params = nets.fc_params(env)
    params.lr = 0.1
    network = nets.ActorCritic(params)
else:
    network = torch.load(f=f'./Data/agents/load_agents/{network_id}.pt')



## build a memory module that knows all the right actions
memory = Memory.EpisodicMemory(cache_limit=400, entry_size=env.action_space.n, mem_temp=1)



agent = Agent(network, memory=memory)
agent.EC_storage = EC_pass

run = expt(agent, env)


for coord, rep in zip(run.sample_states, run.sample_reps):
    actions = correct_actions(coord)
    if len(actions)==0:
        item= {}
        item['activity'] = tuple(rep)
        item['action']   = 0
        item['delta']    = 0
        item['timestamp']= 0
        item['trial']    = 0
        item['readable'] = coord
        run.agent.EC.add_mem(item)
    else:
        for action in actions:
            item= {}
            item['activity'] = tuple(rep)
            item['action']   = action
            item['delta']    = 10
            item['timestamp']= 0
            item['trial']    = 0
            item['readable'] = coord

            run.agent.EC.add_mem(item)

print(len(agent.EC.cache_list))
'''
EC_pols = np.zeros(env.shape, dtype=[(x, 'f8') for x in env.action_list])
for rep, s in zip(run.sample_reps, run.sample_states):
    ec_p = memory.recall_mem(tuple(rep), timestep=0)
    EC_pols[s[0],s[1]] = tuple(ec_p)

plot_pref_pol(env,EC_pols)
'''
print(run.agent.EC.mem_temp)

run.run(NUM_EVENTS=100, NUM_TRIALS=1000, printfreq=100)
run.record_log(expt_type='fixed_mem', env_name=env_name, n_trials=ntrials, dir='../../../Data/', file='debug_bootstrap.csv')