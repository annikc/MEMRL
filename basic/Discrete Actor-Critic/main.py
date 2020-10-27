import numpy as np
import gym
from agent import Agent
import matplotlib.pyplot as plt
from utils import plot_learning_curve
from gym import wrappers


if __name__ == '__main__':
    # first we create our agent
    agent = Agent(alpha=0.00001, beta=0.0005, input_dims=[4], gamma=0.99, 
                    n_actions=2, l1_size=32, l2_size=32)

    env = gym.make('CartPole-v1')
    score_history = []
    n_episodes = 2500

    for i in range(n_episodes):
        done = False
        score = 0 
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation) # choose action to take
            observation_, reward, done, info = env.step(action) # get info from taking that action
            score += reward # add reward to the score for the episode
            agent.learn(observation, reward, observation_, done) 
            observation = observation_
        print(f"Episode: {i}, Score: {score}")
        score_history.append(score)

    filename = 'cartpole.png'
    x = [i+1 for i in range(n_episodes)]
    plot_learning_curve(x, score_history, figure_file)
    