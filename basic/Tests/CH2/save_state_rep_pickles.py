####
# File uses basic fully connected actor critic architecture to train in a gridworld environment
# State inputs are one-hot vectors of discrete grid states -- similar experiments are done in CH2/ for different
# styles of representation of state (i.e. place cell, successor representation)
# No memory module is used in this run
# Data can be logged by calling the .record_log() function of the experiment class
#
#####
import numpy as np
import gym
import sys
sys.path.append('../../modules/')
from basic.modules.Agents.RepresentationLearning.learned_representations import onehot, place_cell, rand_place_cell, sr
rep_types = {'onehot':onehot, 'place_cell':place_cell, 'rand_place_cell':rand_place_cell, 'sr':sr}
import pickle


## set parameters for run -- environment, representation type
env_name = 'gridworld:gridworld-v4'
representation_type = 'place_cell'
num_trials = 5000
num_events = 250


## generate the environment object
env = gym.make(env_name)

## get state representations to be used
state_reps, representation_name, input_dims, _ = rep_types[representation_type](env)

rep_matrix = np.zeros((env.nstates,env.nstates))
for key in state_reps.keys():
    rep = state_reps[key]
    rep_matrix[key] = rep

print(rep_matrix.shape)

with open(f'../../modules/Agents/RepresentationLearning/Learned_Rep_pickles/{representation_type}_{env_name[-12:]}.p', 'wb') as f:
    pickle.dump(rep_matrix,f)


