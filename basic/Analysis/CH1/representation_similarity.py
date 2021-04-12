import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle
import torch
import sys
sys.path.append('../../modules')
from basic.modules.Agents.RepresentationLearning.learned_representations import onehot, random, place_cell, sr, saved_latents
rep_types = {'onehot':onehot, 'random':random, 'place_cell':place_cell, 'sr':sr, 'latent':saved_latents}

sys.path.append('../../../')


version = 1
env_name = f'gridworld:gridworld-v{version}'
representation_type = 'latent'


# make gym environment
env = gym.make(env_name)
plt.close()

state_reps, representation_name, input_dims, _ = rep_types[representation_type](env)