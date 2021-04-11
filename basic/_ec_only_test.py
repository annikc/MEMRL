import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle
import torch
import sys
sys.path.append('../../modules')
from modules.Agents.Networks import flat_ActorCritic as head_AC
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from modules.Agents.RepresentationLearning.learned_representations import onehot, random, place_cell, sr, saved_latents
rep_types = {'onehot':onehot, 'random':random, 'place_cell':place_cell, 'sr':sr, 'latent':saved_latents}
from modules.Agents import Agent
from modules.Experiments import flat_expt
sys.path.append('../../../')


write_to_file = 'ec_latent_testing.csv'
version = 5
train_env_name = f'gridworld:gridworld-v{version}'
test_env_name = train_env_name+'1'

representation_type = 'sr'
num_trials = 5000
num_events = 250


# make gym environment
train_env = gym.make(train_env_name)
plt.close()

state_reps, representation_name, input_dims, _ = rep_types[representation_type](train_env)
env = gym.make(test_env_name)



for _ in range(1):
    empty_net = head_AC(input_dims, env.action_space.n, lr=0.0005)
    memory = Memory(entry_size=4, cache_limit=400)
    agent = Agent(empty_net, memory, state_representations=state_reps)

    ex = flat_expt(agent, env)
    ex.run(num_trials,num_events,snapshot_logging=False)
    ex.record_log(env_name=test_env_name, representation_type=representation_name,
                  n_trials=num_trials, n_steps=num_events,
                  dir='./Data/', file=write_to_file)
