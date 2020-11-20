import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import basic.RepresentationLearning as rep
import matplotlib.pyplot as plt

def train_network(network, data, training_cycles, **kwargs):
    lr = kwargs.get('lr',0.001)
    print_freq = kwargs.get('print_freq',10)

    loss = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    [states, actions, n_states] = data
    loss_tracker = []
    for train in range(training_cycles):
        # get guesses from network
        phi, psi, reconst = network(states, actions)

        # compute loss
        optimizer.zero_grad()
        output = loss(reconst, torch.Tensor(n_states))
        loss_tracker.append(output)
        output.backward()
        optimizer.step()

        if train%print_freq==0:
            print(f'Training Cycle:{train} Loss:{output}')
    return loss_tracker

env = gym.make('gym_grid:gridworld-v1')

n_samples = 1000
conv_data = rep.get_conv_samples(env, n_samples)
ohot_data = rep.get_onehot_samples(env, n_samples)

conv_autoencoder = rep.Conv_OSFM()

conv_loss = train_network(conv_autoencoder, conv_data, training_cycles=10)
plt.plot(conv_loss)


