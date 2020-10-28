import gym
import numpy as np

import matplotlib.pyplot as plt

env = gym.make('gym_grid:gridworld-v1')
# available actions in gridworld
acts = [x[0] for x in env.action_list]

# check functions of gridworld gym env
env.reset()

def get_action(s):
    return np.random.choice(acts)


maxsteps = 100
for step in range(maxsteps):
    s = env.get_state()

    action = get_action(s)

    s_prime, r, done, __ = env.step(action)

    print(s, action, s_prime, r)

    env.render(0.05)

    if step == maxsteps-1 or done:
        plt.show(block=True)

    if done:
        break