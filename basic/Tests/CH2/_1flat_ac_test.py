import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle
import torch
import sys
sys.path.append('../../modules')
from modules.Agents.Networks import flat_ActorCritic as head_AC
from modules.Agents.RepresentationLearning.learned_representations import onehot, random, place_cell, sr, latents
from modules.Agents import Agent
from modules.Experiments import flat_expt
sys.path.append('../../../')


write_to_file = 'flat_ac_training.csv'
version = 1
env_name = f'gridworld:gridworld-v{version}1'
representation_type = 'latent'
num_trials = 25000
num_events = 250
relative_path_to_data = '../../Data' # ../../Data if you are in Tests/CH2


# make gym environment
env = gym.make(env_name)
plt.close()


rep_types = {'onehot':onehot, 'random':random, 'place_cell':place_cell, 'sr':sr, 'latent':latents}
state_reps, representation_name, input_dims, _ = rep_types[representation_type](env)


# load weights to head_ac network from previously learned agent
AC_head_agent = head_AC(input_dims, test_env.action_space.n, lr=0.0005)

agent = Agent(AC_head_agent, state_representations=state_reps)

ex = flat_expt(agent, test_env)
ex.run(num_trials,num_events,snapshot_logging=False)
ex.record_log(env_name=test_env_name, representation_type=representation_name,
              n_trials=num_trials, n_steps=num_events,
              dir=relative_path_to_data, file=write_to_file)