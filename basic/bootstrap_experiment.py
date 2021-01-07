import torch
print('gpu:', torch.cuda.is_available())
import gym
import numpy as np
import modules.Agents.Networks as nets
import modules.Agents.EpisodicMemory as Memory
from modules.Agents import Agent
from modules.Experiments import gridworldBootstrap as expt

import matplotlib.pyplot as plt

# create environment
env_name = 'gym_grid:gridworld-v1'
env = gym.make(env_name)
plt.close()
# generate parameters for network from environment observation shape
params = nets.fc_params(env)
# generate network
network = nets.ActorCritic(params)

memory = Memory.EpisodicMemory(cache_limit=400, entry_size=env.action_space.n)

agent = Agent(network, memory=memory)

run = expt(agent, env)
ntrials=5000

#run.run(NUM_TRIALS=ntrials, NUM_EVENTS=250)
#run.record_log('bootstrap', env_name, n_trials=ntrials)
#plt.plot(run.data['total_reward'])
#plt.plot(run.data['bootstrap_reward'])

#plt.show()
