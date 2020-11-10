import gym
import numpy as np

import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

def get_action():
    return env.action_space.sample()# np.random.choice(env.action_space.shape)


maxsteps = 100
s = env.reset()
print(s)
for step in range(maxsteps):

    action = get_action()
    print(action)

    s_prime, r, done, __ = env.step(action)

    print(s, action, s_prime, r)

    env.render(0.05)

    s = s_prime

    if step == maxsteps-1 or done:
        plt.show(block=True)

    if done:
        break