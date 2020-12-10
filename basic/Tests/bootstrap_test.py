## test experiment class
# using gridworld and actor critic network architecture
import gym

import basic.Agents.Networks as nets
from basic.Agents.Networks import fc_params as basic_agent_params
from basic.Agents.EpisodicMemory import EpisodicMemory as Memory
from basic.Agents import Agent as Agent
from basic.Experiments import Bootstrap as ex

import matplotlib.pyplot as plt
import numpy as np
import basic.Utils.gridworld_plotting as gp

# Make Environment to Test Agent in
env = gym.make('gym_grid:gridworld-v1')

# here using fully connected network
params = basic_agent_params(env)

# build fully connected network
network = nets.ActorCritic(params)
# pass through test observation
test_state = np.zeros(400)
test_state[0] = 1
print(network(test_state))


for x in range(10):
    # build fully connected network
    network = nets.ActorCritic(params)
    memory = Memory(entry_size=params.action_dims, cache_limit=400)
    memory.key_sim = memory.cosine_sim
    agent = Agent(network, memory=memory)
    # instantiate new experiment
    run = ex(agent,env)
    run.run(NUM_TRIALS=1000,NUM_EVENTS=250, printfreq=10, render=False)
    run.record_log(' ', file='test_bootstrap.csv' )


'''
fig, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(run.data['total_reward'])
ax[1].plot(run.data['loss'][0], label='p')
ax[1].plot(run.data['loss'][1], label='v')
ax[1].legend(bbox_to_anchor=(1.05, 0.95))
plt.show()
'''