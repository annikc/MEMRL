## write an example agent and show that it does stuff
import numpy as np
import torch
import matplotlib.pyplot as plt

def get_action(agent, s):
    action, logprob, value = agent.get_action(s)
    return action

def agent_test(env, agent):
    maxsteps = 100
    for step in range(maxsteps):
        s = torch.Tensor(np.expand_dims(env.get_observation(), axis=0))

        action = get_action(agent, s)

        s_prime, r, done, __ = env.step(action)

        env.render()

        if step == maxsteps - 1 or done:
            plt.pause(2)
            plt.close()

        if done:
            break