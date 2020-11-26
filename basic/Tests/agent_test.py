## write an example agent and show that it does stuff
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt

import basic.Agents.Networks as nets
from basic.Agents import Agent, DualNetwork

# Make Environment to Test Agent in
env = gym.make('gym_grid:gridworld-v1')
# for actor critic agent
class basic_agent_params(object):
    def __init__(self, env):
        self.input_dims = env.observation_space.shape
        self.action_dims = env.action_space.n
        self.hidden_types = ['conv', 'pool', 'conv', 'pool', 'linear', 'linear']
        self.hidden_dims = [None, None, None, None, 1000, 1000]
        self.lr = 5e-4

params = basic_agent_params(env)
network = nets.ActorCritic(params)
print(network)
#network = nets.FC(params)
#network = nets.CNN_AC(params)
#network = nets.CNN_2N(params)
#network = nets.FC2N()

agent = Agent(network, memory=None)


def get_action(s):
    action, logprob, value = agent.get_action(s)
    return action

maxsteps = 100
for step in range(maxsteps):
    s = torch.Tensor(np.expand_dims(env.get_observation(), axis=0))

    action = get_action(s)

    s_prime, r, done, __ = env.step(action)

    env.render()

    if step == maxsteps - 1 or done:
        plt.pause(2)
        plt.close()

    if done:
        break

