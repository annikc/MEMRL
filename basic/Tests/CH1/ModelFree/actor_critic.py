import torch
import gym
import numpy as np
import modules.Agents.Networks as nets
import modules.Agents.EpisodicMemory as Memory
from modules.Agents import Agent
from modules.Experiments import expt as expt
import matplotlib.pyplot as plt
from modules.Utils import running_mean as rm
import time
import uuid
import csv
import pickle

def pref_ac_action(psnap):
    action_table = np.zeros(env.shape)
    for state in range(env.nstates):
        state2d = env.oneD2twoD(state)
        action_table[state2d] = np.argmax(list(psnap[state2d]))

    return action_table

env_name   = 'gym_grid:gridworld-v1'

# create environment
env = gym.make(env_name)
plt.close()

network_params = nets.fc_params(env)
network_params.lr = 0.005
print(network_params.__dict__)
network = nets.ActorCritic(network_params)


# generate agent
ntrials = 10000
nevents = 250
agent = Agent(network, None)
run = expt(agent, env)

run.run(NUM_TRIALS=ntrials, NUM_EVENTS=nevents, printfreq=10,snap=False)
run.record_log(dir='../../../Data/',file='actor_critic_learning.csv',expt_type='MFonly',env_name=env_name,n_trials=ntrials,n_steps=nevents)
plt.figure()
plt.plot(run.data['total_reward'])


plt.figure()
ac_pref = pref_ac_action(run.data['P_snap'][-1])
plt.imshow(ac_pref)
plt.show()