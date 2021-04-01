import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from modules.Utils import running_mean as rm

# network
import modules.Agents.Networks as nets
# episodic memory
import modules.Agents.EpisodicMemory as Memory
# agent
from modules.Agents import Agent

## try most basic experiment
from modules.Experiments import Bootstrap_viewMF as expt

# build environment
linear_track = 'gridworld:gridworld-v112' # 1x20 gridworld with reward at (0,19)
mini_grid = 'gridworld:gridworld-v111' # 7x7 gridworld with reward at (5,5)

env_name = linear_track

env = gym.make(env_name)
plt.show()


# make agent
# set up parameters for network
learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
#learning_rates = [0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045]
for l in learning_rates:
    params = nets.fc_params(env)
    params.lr = l
    print(params.__dict__)

    network = nets.ActorCritic(params)

    memtemp = 1
    memory = Memory.EpisodicMemory(cache_limit=400, entry_size=env.action_space.n, mem_temp=memtemp)

    agent = Agent(network, memory)
    #agent.get_action = agent.MF_action

    opt_values = np.zeros(env.nstates)
    reward_loc = env.twoD2oneD(list(env.rewards.keys())[0])
    opt_values[reward_loc] = list(env.rewards.values())[0]

    for ind in reversed(range(len(opt_values)-1)):
        opt_values[ind] = env.step_penalization + agent.gamma*opt_values[ind+1]

    ntrials = 1000
    nevents = 50
    run = expt(environment=env, agent=agent)

    run.run(ntrials,nevents,printfreq=10)
    run.data['opt_values'] = opt_values

    run.record_log(dir='../../../Data/' ,file='linear_track.csv', expt_type=f'{type(run).__name__}_lr{params.lr}', env_name=env_name, n_trials=ntrials, n_steps=nevents, extra=[params.lr])
#plt.plot(rm(run.data['total_reward'],10))
#plt.show()