import numpy as np
import gym
import matplotlib.pyplot as plt
import sys
sys.path.append('../modules/')
from modules.Agents.Networks import ActorCritic as Network
from modules.Agents import Agent
from modules.Experiments import expt
from modules.Utils import one_hot_state

# get environment
env_name = 'gym_grid:gridworld-v4'
env = gym.make(env_name)
plt.close()

# make collection of one-hot state representations
oh_state_reps = {}
for state in env.useable:
    oh_state_reps[env.twoD2oneD(state)] = one_hot_state(env,env.twoD2oneD(state))

input_dims = len(oh_state_reps[list(oh_state_reps.keys())[0]])

network = Network(input_dims=[input_dims],fc1_dims=200,fc2_dims=200,output_dims=env.action_space.n, lr=0.0005)

agent = Agent(network, state_representations=oh_state_reps)

ex = expt(agent,env)

num_trials = 2000

num_events = 250

ex.run(num_trials, num_events)

# print results of training
fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(ex.data['total_reward'])
ax[1].plot(ex.data['loss'][0], label='P_loss')
ax[1].plot(ex.data['loss'][1], label='V_loss')
ax[1].legend(bbox_to_anchor=(1.05,0.95))
plt.show()