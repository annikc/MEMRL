## test experiment class
# using gridworld and actor critic network architecture
import gym
from basic.modules.Agents.Networks import params as basic_agent_params

import basic.modules.Agents.Networks as nets
from basic.modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from basic.modules.Agents import Agent
from basic.modules.Experiments import Experiment as ex

import matplotlib.pyplot as plt
import basic.modules.Utils.gridworld_plotting as gp

# Make Environment to Test Agent in
env = gym.make('gym_grid:gridworld-v1')

params = basic_agent_params(env)
network = nets.ActorCritic(params)
memory = Memory(entry_size=params.action_dims, cache_limit=400)
agent = Agent(network, memory=memory)
agent.get_action = agent.MF_action

run = ex(agent,env)

run.run(100,250, printfreq=10, render=False)

fig, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(run.data['total_reward'])
ax[1].plot(run.data['loss'][0], label='p')
ax[1].plot(run.data['loss'][1], label='v')
ax[1].legend(bbox_to_anchor=(1.05, 0.95))
plt.show()