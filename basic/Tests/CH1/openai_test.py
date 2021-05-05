## test experiment class
# using gridworld and actor critic network architecture
import gym
import torch
from modules.Agents.Networks import fc_params as basic_agent_params

import modules.Agents.Networks as nets
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from modules.Agents import Agent
from modules.Experiments import cont_state_Experiment as ex
from modules.Utils import running_mean as rm

import matplotlib.pyplot as plt
import modules.Utils.gridworld_plotting as gp

# Make Environment to Test Agent in
#env = gym.make('FrozenLake-v0', is_slippery=False)
env = gym.make('CartPole-v0')
s_ = env.reset()
a = env.action_space.sample()
s, r, done, __ = env.step(a)
print(s_, a, s,r,done)

## write network parameters
params = basic_agent_params(env)
print(params.__dict__)
params.hidden_types = ['linear', 'linear']
params.hidden_dims = [50, 50]
params.lr = 0.001

network = nets.ActorCritic(params)
print(network)
memory = None #Memory(entry_size=params.action_dims, cache_limit=400)
agent = Agent(network, memory=memory)
agent.get_action = agent.MF_action
run = ex(agent,env)

run.run(5000,250, printfreq=100, render=False)
#run.record_log(expt_type='test',env_name='FrozenLake-V0',n_trials = 0, dir='../Data/', file='test_environments.csv')

fig, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(rm(run.data['total_reward'], 100))
#ax[0].set_ylim([0,1])
ax[1].plot(run.data['loss'][0], label='p')
ax[1].plot(run.data['loss'][1], label='v')
ax[1].legend(bbox_to_anchor=(1.05, 0.95))
plt.show()