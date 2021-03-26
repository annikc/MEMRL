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
from modules.Experiments import gridworldExperiment as expt

# build environment
linear_track = 'gym_grid:gridworld-v112' # 1x20 gridworld with reward at (0,19)
env_name = linear_track

env = gym.make(env_name)
plt.show()

params = nets.fc_params(env)
params.lr = 0.5
print(params.__dict__)

network = nets.ActorCritic(params)

memtemp = 1
memory = Memory.EpisodicMemory(cache_limit=400, entry_size=env.action_space.n, mem_temp=memtemp)

agent = Agent(network, memory)
agent.get_action = agent.MF_action

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

run.record_log(dir='../../../Data/' ,file='linear_track_vpi.csv', expt_type=f'{type(run).__name__}', env_name=env_name, n_trials=ntrials, n_steps=nevents, extra=[params.lr])
#plt.plot(rm(run.data['total_reward'],10))
#plt.show()