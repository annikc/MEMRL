import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle
import torch
import sys
sys.path.append('../../modules')
from basic.modules.Agents.Networks import flat_ActorCritic as head_AC
from basic.modules.Agents.RepresentationLearning.learned_representations import convs, onehot, place_cell, random, sr

from basic.modules.Agents import Agent

from basic.modules.Experiments import expt
sys.path.append('../../../')


# network from torch.load
version = 1
env_name = f'gridworld:gridworld-v{version}'
num_trials = 25000
num_events = 250
write_to_file = 'flat_ac_training.csv'

# make gym environment
env = gym.make(env_name)
plt.close()

# get representations
state_reps, representation_name, dim, _ = random(env)


for _ in range(5):
    empty_net = head_AC(dim, env.action_space.n, lr=0.0005)

    agent = Agent(empty_net, state_representations=state_reps)
    test_env = gym.make(f'gridworld:gridworld-v{version}1')

    ex = expt(agent, test_env)
    ex.run(num_trials,num_events,snapshot_logging=True)
    ex.flat_record_log(env_name=env_name, representation_type=representation_name,
                  n_trials=num_trials, n_steps=num_events,
                  dir='../../Data/', file=write_to_file)
