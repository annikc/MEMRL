# import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import sys
import torch
import pickle
sys.path.insert(0, '../../../modules/')

from modules.Utils import running_mean as rm
from modules.Utils import one_hot_state, onehot_state_collection, twoD_states
# import actor critic network
from modules.Agents.Networks import ActorCritic as Network

# import memory module
from modules.Agents.EpisodicMemory import EpisodicMemory

# import agent class wrapper to handle behaviour
from modules.Agents import Agent

# import representation type
from modules.Agents.RepresentationLearning import PlaceCells

# import experiment class to handle run and logging
from modules.Experiments import expt

# get environment
import gym


# make env
env_name = 'gym_grid:gridworld-v4'
env = gym.make(env_name)
#env.set_reward({(5,5):2})
plt.close()

# 1. Tabular Q
# 2. SARSA
# 3. DQN
# 4. ActorCritic
input_dims = 400

data_dir = '../../../Data/'
load_id = 'd80ea92c-422c-436a-b0ff-84673d43a30d'

memory = EpisodicMemory(env.action_space.n, cache_limit=env.nstates)

oh_state_reps = {}
for state in env.useable:
	oh_state_reps[env.twoD2oneD(state)] = one_hot_state(env.twoD2oneD(state))




oh_network = Network(input_dims=[input_dims],fc1_dims=200,fc2_dims=200,output_dims=env.action_space.n, lr=0.0005)
oh_network = torch.load(data_dir+f'agents/{load_id}.pt')
oh_agent = Agent(oh_network, state_representations=oh_state_reps, memory=memory)
oh_agent.get_action = oh_agent.EC_action


# retraining
env.set_reward({(15,15):10})

ex = expt(oh_agent, env)
ntrials = 2000
nsteps = 250
ex.run(ntrials, nsteps, printfreq=1)
ex.record_log('oh_episodic',env_name,ntrials,nsteps, dir=data_dir,file='ac_representation.csv')
# save place cells





