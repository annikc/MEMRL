import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle
import torch
import sys
sys.path.append('../../modules')
from modules.Agents.Networks import flat_ActorCritic as head_AC
from modules.Agents.Networks import flex_ActorCritic as Network
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from modules.Agents.RepresentationLearning.learned_representations import onehot, random, place_cell, sr
from modules.Agents import Agent
from modules.Experiments import flat_expt
sys.path.append('../../../')
import argparse

# set up arguments to be passed in and their defauls
parser = argparse.ArgumentParser()
parser.add_argument('-v', type=int, default=1)
parser.add_argument('-rep', default='onehot')
parser.add_argument('-load', type=bool, default=True)
parser.add_argument('-lr', default=0.0005)
parser.add_argument('-cache', type=int, default=400)
args = parser.parse_args()

# parameters set with command line arugments
version       = args.v
rep_type      = args.rep
load_weights  = args.load # load learned weights from conv net or use new init head weights
learning_rate = args.lr
cache_size    = args.cache


# parameters set for this file
relative_path_to_data = '../../Data/' # from within Tests/CH1
write_to_file         = 'ec_rep_test.csv'
training_env_name     = f'gridworld:gridworld-v{version}'
test_env_name         = training_env_name+'1'
num_trials = 5000
num_events = 250

# make new env to run test in
env = gym.make(test_env_name)
plt.close()


rep_types = {'onehot':onehot, 'random':random, 'place_cell':place_cell, 'sr':sr}
state_reps, representation_name, input_dims, _ = rep_types[rep_type](env)

AC_head_agent = head_AC(input_dims, test_env.action_space.n, lr=learning_rate)

memory = Memory(entry_size=env.action_space.n, cache_limit=cache_size)

agent = Agent(AC_head_agent, memory=memory, state_representations=state_reps)

ex = flat_expt(agent, env)
ex.run(num_trials,num_events,snapshot_logging=False)
ex.record_log(env_name=test_env_name, representation_type=representation_name,
              n_trials=num_trials, n_steps=num_events,
              dir=relative_path_to_data, file=write_to_file)
