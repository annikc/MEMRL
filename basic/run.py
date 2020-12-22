## write an example agent and show that it does stuff
# import statements
import torch
import gym
import numpy as np
import modules.Agents.Networks as nets
from modules.Agents import Agent
from Tests.agent_test import agent_test
from Tests.representation_learn_test import rep_learning

import matplotlib.pyplot as plt


# create environment
env = gym.make('gym_grid:gridworld-v1')
# generate parameters for network from environment observation shape
params = nets.params(env)
# generate network
network = nets.ActorCritic(params)



agent = Agent(network, memory=None)

#agent_test(env, agent)
rep_learning('onehot', env, n_samples=100, training_cycles=1000)



### JUNKYARD
#network = nets.FC(params)
#network = nets.CNN_AC(params)
#network = nets.CNN_2N(params)
#network = nets.FC2N()