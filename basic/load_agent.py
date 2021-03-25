import torch
import gym
import numpy as np
import modules.Agents.Networks as nets
import modules.Agents.EpisodicMemory as Memory
from modules.Agents import Agent
from modules.Experiments import expt
import matplotlib.pyplot as plt
from basic.modules.Utils.gridworld_plotting import plot_polmap, plot_pref_pol, plot_valmap, plot_world

env_name   = 'gym_grid:gridworld-v1'
network_id = None# 'd02971b6-a9e7-4c43-a280-05a6b0ef09bc'
ntrials    = 1000

# create environment
env = gym.make(env_name)
plt.close()

# generate network
if network_id == None:
    # generate parameters for network from environment observation shape
    params = nets.fc_params(env)
    print(params.__dict__)
    network = nets.ActorCritic(params)
else:
    network = torch.load(f=f'./Data/agents/{network_id}.pt')

memory = Memory.EpisodicMemory(cache_limit=400, entry_size=env.action_space.n)

agent = Agent(network, memory=memory)
print(len(agent.EC.cache_list))

run = expt(agent, env)
print(network)
