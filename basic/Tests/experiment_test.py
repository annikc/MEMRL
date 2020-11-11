## test experiment class
# using gridworld and actor critic network architecture
import gym
from basic.Utils import basic_agent_params

import basic.Agents.Networks as nets
from basic.Agents.EpisodicMemory import EpisodicMemory as Memory
from basic.Agents import Agent
from basic.Experiments import Experiment as ex

import matplotlib.pyplot as plt

# Make Environment to Test Agent in
env = gym.make('gym_grid:gridworld-v1')

params = basic_agent_params(env)
network = nets.ActorCritic(params)
memory = Memory(entry_size=params.action_dims, cache_limit=400)
agent = Agent(network, memory=memory)
agent.get_action = agent.EC_action

run = ex(agent,env)
run.run(100,100, printfreq=10, render=True)
