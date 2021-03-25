import torch
import gym
import numpy as np
import modules.Agents.Networks as nets
import modules.Agents.EpisodicMemory as Memory
from modules.Agents import Agent
from modules.Experiments import Bootstrap_interleaved as expt
import matplotlib.pyplot as plt
from modules.Utils import running_mean as rm

env_name   = 'gym_grid:gridworld-v1'
network_id = None # '97b5f281-a60e-4738-895d-191a04edddd6'

# create environment
env = gym.make(env_name)
plt.close()

# generate network
if network_id == None:
    # generate parameters for network from environment observation shape
    params = nets.fc_params(env)
    params.lr = 0.001
    params.temp = 1
    print(params.__dict__)
    network = nets.ActorCritic(params)
else:
    network = torch.load(f=f'./Data/agents/load_agents/{network_id}.pt')

memtemp = 1
memory = Memory.EpisodicMemory(cache_limit=400, entry_size=env.action_space.n, mem_temp=memtemp)

agent = Agent(network, memory=memory)

run = expt(agent, env)

ntrials = 1000
nevents = 250
run.run(NUM_TRIALS=ntrials, NUM_EVENTS=nevents)
run.record_log(file='MFtraining.csv', expt_type=f'{type(run).__name__}', env_name=env_name, n_trials=ntrials, n_steps=nevents)
smoothing = 10
fig, ax = plt.subplots(3,1,sharex=True)
ax[0].plot(rm(run.data['total_reward'],smoothing), 'k', alpha=0.5)
ax[0].plot(rm(run.data['bootstrap_reward'],smoothing),'r')

ax[1].plot(rm(run.data['loss'][0], smoothing), label='ec_p')
ax[1].plot(rm(run.data['mf_loss'][0], smoothing), label='mf_p')
ax[1].legend(loc=0)

ax[2].plot(rm(run.data['loss'][1], smoothing), label='ec_v')
ax[2].plot(rm(run.data['mf_loss'][1], smoothing), label='mf_v')
ax[2].legend(loc=0)

plt.show()