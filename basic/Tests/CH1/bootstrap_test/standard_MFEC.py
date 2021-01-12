import torch
import gym
import numpy as np
import modules.Agents.Networks as nets
import modules.Agents.EpisodicMemory as Memory
from modules.Agents import Agent
from modules.Experiments import gridworldExperiment as expt
import matplotlib.pyplot as plt

# set up parameters
env_name   = 'gym_grid:gridworld-v1' #v1 novel #v11 moved reward
network_id = None # '97b5f281-a60e-4738-895d-191a04edddd6'
actor      = 'EC'
ntrials    = 5000


# create environment
env = gym.make(env_name)
plt.close()

# generate network
if network_id == None:
    # generate parameters for network from environment observation shape
    params = nets.fc_params(env)
    network = nets.ActorCritic(params)
else:
    network = torch.load(f'./Data/agents/load_agents/{network_id}.pt')

memory = Memory.EpisodicMemory(cache_limit=400, entry_size=env.action_space.n)

agent = Agent(network, memory=memory)

if actor == 'MF':
    agent.get_action = agent.MF_action
elif actor == 'EC':
    agent.get_action = agent.EC_action

run = expt(agent, env)
run.run(NUM_TRIALS=ntrials, NUM_EVENTS=250)
run.record_log(f'{actor}', env_name, n_trials=ntrials)


