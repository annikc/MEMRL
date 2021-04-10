import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle
import torch
import sys
sys.path.append('../../modules')
from modules.Agents.Networks import flat_ActorCritic as head_AC
from modules.Agents.RepresentationLearning.learned_representations import convs, onehot, place_cell, latents
from modules.Agents import Agent
from modules.Experiments import expt
sys.path.append('../../../')


save = False
version = 1
env_id = f'gridworld:gridworld-v{version}'
example_run_ids = ['c34544ac-45ed-492c-b2eb-4431b403a3a8',
                   '9a12edd8-a978-4e6b-a9f9-e09e0e35c534',
                   '32301262-cd74-4116-b776-57354831c484',
                   'b50926a2-0186-4bb9-81ec-77063cac6861',
                   '15b5e27b-444f-4fc8-bf25-5b7807df4c7f'] # example of each gridworld - v1, v2, v3, v4, v5 using conv representations (no rewards in input tensor)

run_id = example_run_ids[version-1]


# make gym environment
env = gym.make(env_id)
plt.close()

reps, name, dim, _ = latents(env, f'./../../Data/agents/{run_id}.pt' )

latent_array = np.zeros((env.nstates,env.nstates))
for i in reps.keys():
    latent_array[i] = reps[i]

if save:
    with open(f'../../modules/Agents/RepresentationLearning/Learned_Rep_pickles/conv_{env_id[-12:]}.p', 'wb') as f:
        pickle.dump(file=f, obj=latent_array.copy())
