# import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import sys
sys.path.insert(0, '../../../modules/')

from modules.Utils import running_mean as rm

# import actor critic agent
from modules.Agents.tabular_agents import Q_Agent, q_navigate

# get environment
import gym
env = gym.make('gym_grid:gridworld-v4')
plt.close()


q_agent = Q_Agent(env, learning_rate=0.1, discount=0.95, epsilon=1.0)

num_episodes = 25
total_reward = q_navigate(env, q_agent, num_episodes, random_start=True)

plt.figure()
plt.plot(total_reward)
plt.show()