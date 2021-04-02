####
# Training data collected with reward at (5,5) for gridworlds 1-4 --> testing data moves reward to (15,15)
# Gridworld 5 (tunnel environment) has reward at (3,10) during training --> testing moves reward to (16,10)
#
#####
# import packages
import numpy as np
import sys
sys.path.append('/../../../modules')

from modules.Utils import running_mean as rm
from modules.Utils import one_hot_state, onehot_state_collection, twoD_states

# import representation type
from modules.Agents.RepresentationLearning import PlaceCells
# import actor critic network
from modules.Agents.Networks import ActorCritic as Network
# import agent class wrapper to handle behaviour
from modules.Agents import Agent
# import experiment class to handle run and logging
from modules.Experiments import expt

from modules.Agents.EpisodicMemory import EpisodicMemory as Memory

# get environment
import gym