## get an episodic controller to produce successful behaviour in an environment
## use trivial representation i.e. one-hot / direct state observation
## test experiment class
# using gridworld and actor critic network architecture
import gym
import sys
sys.path.append('../../../basic/')
from basic.modules.Agents.Networks import fc_params as basic_agent_params
import numpy as np
import basic.modules.Agents.Networks as nets
from basic.modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from basic.modules.Agents import Agent
from basic.modules.Experiments import gridworldBootstrap as ex
import basic.modules.Utils.gridworld_plotting as gp
import matplotlib.pyplot as plt
import torch



# Make Environment to Test Agent in
env_name = 'gym_grid:gridworld-v11'
expt_type = 'EC'
env = gym.make(env_name)

# get parameters to specific Model Free Network
params = basic_agent_params(env)
# build network & memory modules
network = torch.load('../../Data/agents/30f9805c-6a45-434c-b774-569ab833ee68.pt') #nets.ActorCritic(params)
memory = Memory(entry_size=params.action_dims, cache_limit=400, mem_temp=0.05)

# construct agent instance with network and memory modules
agent = Agent(network, memory=memory)

# choose action controller
if expt_type == 'EC':
    agent.get_action = agent.EC_action
elif expt_type == 'MF':
    agent.get_action = agent.MF_action

# instantiate experiment
run = ex(agent,env)
total_reward = []
for __ in range(1):
    run.run(7000,250, printfreq=100, render=False)
    total_reward += run.data['total_reward']
#run.record_log(expt_type, env_name, n_trials=5000, dir='../../Data/', file='reward_switch_bootstrap.csv')

fig, ax = plt.subplots(2,1,sharex=False)

ax[0].plot(run.data['total_reward'])
ax[0].plot(run.data['bootstrap_reward'], c='r')
ax[1].plot(run.data['loss'][0], label='p')
ax[1].plot(run.data['loss'][1], label='v')
ax[1].legend(bbox_to_anchor=(1.05, 0.95))
plt.show()
'''
# show examples from
state_obs = []
states = []
for i in range(env.nstates):
    state = np.zeros(env.nstates)
    state[i] = 1
    states.append(env.oneD2twoD(i))
    state_obs.append(state)

val_grid = np.empty(env.shape)
pol_grid = np.zeros(env.shape, dtype=[(x, 'f8') for x in env.action_list])

mem_grid = np.zeros(env.shape, dtype=[(x, 'f8') for x in env.action_list])

for obs, s in zip(state_obs, states):
    p,v = agent.MFC(obs)
    pol_grid[s] = tuple(p.data.numpy())
    val_grid[s] = v.item()

    m = agent.EC.recall_mem(tuple(obs))
    mem_grid[s] = tuple(m)

gp.plot_pref_pol(env, mem_grid)
gp.plot_polmap(env,mem_grid)

gp.plot_pref_pol(env,pol_grid)
gp.plot_polmap(env,pol_grid)



'''