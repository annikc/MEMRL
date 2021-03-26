####
# Determine similarity between states from a given representation type
# Onehot / place cells / SR
#
#####

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
env.set_reward({(15,15):2})
env.reset_viewer()
plt.show()

# 1. Tabular Q
# 2. SARSA
# 3. DQN
# 4. ActorCritic
input_dims = 400

data_dir = '../../../Data/'
load_id = 'b6f51c73-ebc0-467a-b5e5-5b51a5a3208d'

memory = EpisodicMemory(env.action_space.n, cache_limit=env.nstates)
with open(data_dir+ f'/ec_dicts/dc126211-0af0-4fc1-8788-3f1b8567cdc2_EC.p', 'rb') as f:
    memory.cache_list = pickle.load(f)

### place cell representations
#place_cells = PlaceCells(env.shape, input_dims, field_size=0.25)
# load place cells
with open(data_dir+ f'results/{load_id}_data.p', 'rb') as f:
    place_cells = (pickle.load(f))['place_cells']

pc_state_reps = {}
oh_state_reps = {}
for state in env.useable:
	oh_state_reps[env.twoD2oneD(state)] = one_hot_state(env.twoD2oneD(state))
	pc_state_reps[env.twoD2oneD(state)] = place_cells.get_activities([state])[0]


place_cells.plot_placefields(env_states_to_map=env.useable)

#oh_network = Network(input_dims=[input_dims],fc1_dims=200,fc2_dims=200,output_dims=env.action_space.n, lr=0.0005)
#oh_network = torch.load(data_dir+f'agents/{load_id}.pt')
#oh_agent = Agent(oh_network, state_representations=oh_state_reps)

#pc_network = Network(input_dims=[input_dims],fc1_dims=200,fc2_dims=200,output_dims=env.action_space.n, lr=0.0005)
pc_network = torch.load(data_dir+f'agents/{load_id}.pt')
pc_agent = Agent(pc_network, state_representations=pc_state_reps, memory=memory)
pc_agent.get_action = pc_agent.EC_action

# retraining
env.set_reward({(15,15):10})

ex = expt(pc_agent, env)
ntrials = 2000
nsteps = 250
#ex.run(ntrials, nsteps, printfreq=1)
#ex.data['place_cells'] = place_cells
#ex.record_log('pc_episodic',env_name,ntrials,nsteps, dir=data_dir,file='ac_representation.csv')
# save place cells





