import gym
from basic.modules.Agents.Networks import ActorCritic, params

env = gym.make('gym_grid:gridworld-v1')
params = params(env)

agent = ActorCritic(params)
