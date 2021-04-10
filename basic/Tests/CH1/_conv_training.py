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
from basic.modules.Agents.Networks import flex_ActorCritic as Network
from basic.modules.Agents.Networks import conv_PO_params, conv_FO_params
from basic.modules.Agents import Agent
from basic.modules.Agents.RepresentationLearning.learned_representations import convs, reward_convs
from basic.modules.Experiments import conv_expt

# valid representation types for this experiment
rep_types = {'convs':convs, 'reward_convs':reward_convs}
param_set = {'convs': conv_PO_params, 'reward_convs': conv_FO_params}

## set parameters for run
write_to_file = 'conv_mf_training.csv'
env_name      = 'gridworld:gridworld-v5'
representation_type = 'reward_convs'
num_trials = 5000
num_events = 250

# instantiate the environment for the experiment
env = gym.make(env_name)
plt.close()

# get representation type, associated parameters to specify the network dimensions
state_reps, representation_name, input_dims, _ = rep_types[representation_type](env)
params = param_set[representation_type]
network_parameters = params(env)


for _ in range(5):
    # make a new network instance
    network = Network(network_parameters)
    # reinitalize agent with new network
    agent = Agent(network, state_representations=state_reps)

    # expt - redefines logging function to keep track of network details
    ex = conv_expt(agent, env)
    ex.run(num_trials,num_events,printfreq=10)
    ex.record_log(env_name=env_name, representation_type=representation_name,
                      n_trials=num_trials, n_steps=num_events,
                      dir='../../Data/', file=write_to_file)

