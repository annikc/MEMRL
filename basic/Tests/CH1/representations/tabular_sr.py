# import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import sys
import torch
import pickle
sys.path.insert(0, '../../../modules/')
from modules.Utils import running_mean as rm
from modules.Utils import one_hot_state, onehot_state_collection, twoD_states

# get environment
import gym

from modules.Agents.RepresentationLearning.tabular_sr import Tabular_SR_Agent

env_id = 'gym_grid:gridworld-v6'
env = gym.make(env_id)

n_episodes = 50000

n_steps = 300
learning_rate = 1e-3
gamma = 0.99

agent = Tabular_SR_Agent(env,gamma,learning_rate,epsilon=1) ## random walk agent

for i in range(n_episodes):
    env.random_start=True
    state = env.reset()
    for j in range(n_steps):
        state = agent.step_in_env(state)
        if j>1:
            td_sr, td_rwd = agent.update()
        if env.done:
            break
    if i % 200==0:
        print(f'Episode {i}')

plt.figure()
plt.imshow(agent.M[0,:,:])

state = 1
action = 2 # right
sr = agent.M[action,state,:]
sr_env = sr.reshape(*env.shape)
plt.figure()
plt.imshow(sr_env)


plt.show()
x = agent.M.copy()
with open(f'../../../modules/Agents/RepresentationLearning/SR_{env_id}.p', 'wb') as savedata:
    pickle.dump(x, savedata)



