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
import matplotlib.pyplot as plt
import sys
sys.path.append('../../modules/')
from modules.Agents.Networks import flex_ActorCritic as Network
from modules.Agents.Networks import conv_PO_params, conv_FO_params
from modules.Agents import Agent
from modules.Agents.RepresentationLearning.learned_representations import convs, reward_convs
from modules.Experiments import conv_expt
import argparse

# set up arguments to be passed in and their defauls
parser = argparse.ArgumentParser()
parser.add_argument('-v', type=int, default=1)
parser.add_argument('-rep', default='conv')

args = parser.parse_args()

# parameters set with command line arugments
version             = args.v
representation_type = args.rep


## set parameters for run
write_to_file         = 'conv_mf_training_narrow.csv'
relative_path_to_data = './Data/' # from within Tests/CH1
env_name              = f'gridworld:gridworld-v{version}'
num_trials            = 5000
num_events            = 250

# valid representation types for this experiment
rep_types = {'conv':convs, 'rwd_conv':reward_convs}
param_set = {'conv': conv_PO_params, 'rwd_conv': conv_FO_params}

# instantiate the environment for the experiment
env = gym.make(env_name)
plt.close()

# get representation type, associated parameters to specify the network dimensions
state_reps, representation_name, input_dims, _ = rep_types[representation_type](env)
params = param_set[representation_type]
network_parameters = params(env)

# make a new network instance
network = Network(network_parameters)
# reinitalize agent with new network
agent = Agent(network, state_representations=state_reps)

# expt - redefines logging function to keep track of network details
ex = conv_expt(agent, env)
ex.run(num_trials,num_events,printfreq=10)
ex.record_log(env_name=env_name, representation_type=representation_name,
                  n_trials=num_trials, n_steps=num_events,
                  dir=relative_path_to_data, file=write_to_file)

