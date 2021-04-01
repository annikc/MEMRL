## get an episodic controller to produce successful behaviour in an environment
## use trivial representation i.e. one-hot / direct state observation
## test experiment class
# using gridworld and actor critic network architecture
import gym
from basic.modules.Agents.Networks import fc_params as basic_agent_params
import numpy as np
import basic.modules.Agents.Networks as nets
from basic.modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from basic.modules.Agents import Agent
from basic.modules.Experiments import gridworldExperiment as ex
import basic.modules.Utils.gridworld_plotting as gp
import matplotlib.pyplot as plt
import torch
# Make Environment to Test Agent in
env_name = 'gridworld:gridworld-v1'
expt_type = 'MF'
env = gym.make(env_name)


# get parameters to specific Model Free Network
params = basic_agent_params(env)
# build network & memory modules
network = nets.ActorCritic(params)
memory = Memory(entry_size=params.action_dims, cache_limit=400)

# construct agent instance with network and memory modules
agent = Agent(network, memory=memory)

# choose action controller
if expt_type == 'EC':
    agent.get_action = agent.EC_action
elif expt_type == 'MF':
    agent.get_action = agent.MF_action

# instantiate experiment
run = ex(agent,env)
print(env.obstacles_list)
#run.run(5000,250, printfreq=10, render=False)
#run.record_log(expt_type, env_name, n_trials=5000, dir='../../Data/', file='test_environments.csv')

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


fig, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(run.data['total_reward'])
ax[1].plot(run.data['loss'][0], label='p')
ax[1].plot(run.data['loss'][1], label='v')
ax[1].legend(bbox_to_anchor=(1.05, 0.95))
plt.show()
'''