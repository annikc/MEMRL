import torch
import gym
import numpy as np
import modules.Agents.Networks as nets
import modules.Agents.EpisodicMemory as Memory
from modules.Agents import Agent
from modules.Experiments import gridworldExperiment as expt

import matplotlib.pyplot as plt

# action selection type set at beginning 
actor = 'MF'
# create environment
env_name = 'gym_grid:gridworld-v11'
env = gym.make(env_name)
plt.close()
print(env.rewards)
# generate parameters for network from environment observation shape
params = nets.fc_params(env)
# generate network
network = torch.load('./Data/agents/load_agents/97b5f281-a60e-4738-895d-191a04edddd6.pt') #nets.ActorCritic(params)

memory = Memory.EpisodicMemory(cache_limit=400, entry_size=env.action_space.n)


agent = Agent(network, memory=memory)

if actor == 'MF':
    agent.get_action = agent.MF_action
elif actor == 'EC':
    agent.get_action = agent.EC_action

run = expt(agent, env)
ntrials=10000

run.run(NUM_TRIALS=ntrials, NUM_EVENTS=250)
run.record_log(f'movedR_{actor}', env_name, n_trials=ntrials)
#plt.plot(run.data['total_reward'])
#plt.plot(run.data['bootstrap_reward'])

#plt.show()

