import gym
import numpy as np
import matplotlib.pyplot as plt

figtype = 'png'

# v0/v1: basic gridworld v0 = 6 actions, v1 = 4 actions
env = gym.make('gym_grid:gridworld-v0')
plt.savefig(f'gw_basic.{figtype}', format=f'{figtype}')
plt.close()

# v2: gridworld w random obstacles rho = 0.1
env = gym.make('gym_grid:gridworld-v2')
plt.savefig(f'gw_2020_obs.{figtype}', format=f'{figtype}')

# v3: gridworld 4 rooms task
env = gym.make('gym_grid:gridworld-v3')
plt.savefig(f'gw_2020_4rooms.{figtype}', format=f'{figtype}')

# v4: gridworld bar task
env = gym.make('gym_grid:gridworld-v4')
plt.savefig(f'gw_2020_bar.{figtype}', format=f'{figtype}')