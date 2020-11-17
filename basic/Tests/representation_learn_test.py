import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from basic.RepresentationLearning import SRNet
import matplotlib.pyplot as plt




def get_action(s):
    return np.random.choice(len(env.action_list))
def get_samples(maxsteps):
    data_col = [[],[],[]]

    for step in range(maxsteps):
        s = env.get_state()
        state = env.get_observation()


        action = get_action(s)

        s_prime, r, done, __ = env.step(action)
        next_state = env.get_observation()

        data_col[0].append(state*10)
        data_col[1].append(action)
        data_col[2].append(next_state*10)

        #env.render(0.05)

        if step == maxsteps-1 or done:
            #plt.show(block=True)
            pass

        if done:
            env.reset()
            #break
    return data_col

# Make Environment to Test Agent in
env = gym.make('gym_grid:gridworld-v1')
# check functions of gridworld gym env
env.reset()
data_col = [[],[],[]]

testsr = SRNet()
loss = nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(testsr.parameters(), lr = 0.001)

training_cycles = 100
for train in range(training_cycles):
    # get samples from env
    [states, actions, n_states] = get_samples(1000)
    # get guesses from network
    phi, psi, reconst = testsr(states)

    # compute loss
    optimizer.zero_grad()
    output = loss(reconst, torch.Tensor(states))
    output.backward()
    optimizer.step()

    print(f'Training Cycle:{train} Loss:{output}')