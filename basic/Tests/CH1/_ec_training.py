import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle
import torch
import sys
sys.path.append('../../modules')
from basic.modules.Agents.Networks import flat_ActorCritic as head_AC
from basic.modules.Agents.RepresentationLearning.learned_representations import onehot, random, place_cell, sr, saved_latents
rep_types = {'onehot':onehot, 'random':random, 'place_cell':place_cell, 'sr':sr, 'latent':saved_latents}
from basic.modules.Agents import Agent
from basic.modules.Experiments import flat_expt
sys.path.append('../../../')


write_to_file = 'flat_ac_training.csv'
version = 1
env_name = f'gridworld:gridworld-v{version}'
representation_type = 'latent'
num_trials = 25000
num_events = 250


# make gym environment
env = gym.make(env_name)
plt.close()

state_reps, representation_name, input_dims, _ = rep_types[representation_type](env)



for _ in range(1):
    empty_net = head_AC(input_dims, env.action_space.n, lr=0.0005)
    memory = Mem
    agent = Agent(empty_net, state_representations=state_reps)

    ex = flat_expt(agent, env)
    ex.run(num_trials,num_events,snapshot_logging=True)
    ex.record_log(env_name=env_name, representation_type=representation_name,
                  n_trials=num_trials, n_steps=num_events,
                  dir='../../Data/', file=write_to_file, mock_log=True)
