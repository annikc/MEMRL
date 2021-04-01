import torch
import gym
import numpy as np
import modules.Agents.Networks as nets
import modules.Agents.EpisodicMemory as Memory
from modules.Agents import Agent
from modules.Experiments import gridworldBootstrap as expt
import matplotlib.pyplot as plt
from modules.Utils import running_mean as rm

env_name   = 'gridworld:gridworld-v1'
network_id = None # '97b5f281-a60e-4738-895d-191a04edddd6'
ntrials    = 1000

# create environment
env = gym.make(env_name)
plt.close()

# generate network
if network_id == None:
    # generate parameters for network from environment observation shape
    params = nets.fc_params(env)
    params.lr = 0.001
    params.temp = 1.1
    print(params.__dict__)
    network = nets.ActorCritic(params)
else:
    network = torch.load(f=f'./Data/agents/load_agents/{network_id}.pt')
memtemp = 0.05
memory = Memory.EpisodicMemory(cache_limit=400, entry_size=env.action_space.n, mem_temp=memtemp)

agent = Agent(network, memory=memory)

run = expt(agent, env)
ntrials = 1000
run.run(NUM_TRIALS=ntrials, NUM_EVENTS=100)
run.record_log(f'mf_ec_t{memtemp}', env_name, n_trials=ntrials)

smoothing=10
plt.figure()
plt.plot(rm(run.data['total_reward'],smoothing), c='k', alpha=0.5)
if 'bootstrap_reward' in run.data.keys():
    plt.plot(rm(run.data['bootstrap_reward'],smoothing), c='r')
plt.show()