####
# File uses episodic memory module to select actions in the environment
# a basic fully connected actor critic network is set as the Agent.MFC;
# this is trained on the trajectories collected by the episodic module
#
# State inputs are one-hot vectors of discrete grid states -- similar experiments are done in CH2/ for different
# styles of representation of state (i.e. place cell, successor representation)
# Episodic module uses L1 norm to determine distance between keys - in the onehot case all states are orthogonal to
# all other states
#
# Data can be logged by calling the .record_log() function of the experiment class
#
#####
import numpy as np
import gym
import matplotlib.pyplot as plt
import sys
sys.path.append('../../modules/')
from modules.Agents.Networks import ActorCritic as Network
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from modules.Agents import Agent
from modules.Experiments import expt
from modules.Utils import one_hot_state

# get environment
env_name = 'gym_grid:gridworld-v4'
env = gym.make(env_name)
plt.close()

# make collection of one-hot state representations
oh_state_reps = {}
for state in env.useable:
    oh_state_reps[env.twoD2oneD(state)] = one_hot_state(env,env.twoD2oneD(state))

input_dims = len(oh_state_reps[list(oh_state_reps.keys())[0]])

network = Network(input_dims=[input_dims],fc1_dims=200,fc2_dims=200,output_dims=env.action_space.n, lr=0.0005)
memory = Memory(entry_size=env.action_space.n, cache_limit=env.nstates)

agent = Agent(network, memory, state_representations=oh_state_reps)

ex = expt(agent,env)

num_trials = 2000

num_events = 250

ex.run(num_trials, num_events)

# print results of training
fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(ex.data['total_reward'])
ax[1].plot(ex.data['loss'][0], label='P_loss')
ax[1].plot(ex.data['loss'][1], label='V_loss')
ax[1].legend(bbox_to_anchor=(1.05,0.95))
plt.show()