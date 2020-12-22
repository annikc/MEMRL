import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('../modules/')
from modules.Agents import RepresentationLearning as rep


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
        loss_tracker.append(output.item())
        output.backward()
        optimizer.step()

        if train%print_freq==0:
            print(f'Training Cycle:{train} Loss:{output}')
    return loss_tracker

def rep_learning(type, env, n_samples, training_cycles, load=False):
    if type == 'conv':
        data = rep.get_conv_samples(env, n_samples)
        if load:
            autoencoder = torch.load(f='../Data/conv_autoencoder.pt')
        else:
            autoencoder = rep.Conv_OSFM()
    elif type == 'onehot':
        data = rep.get_onehot_samples(env, n_samples)
        if load:
            autoencoder = torch.load(f='../Data/onehot_autoencoder.pt')
        else:
            autoencoder = rep.FC_OSFM()

    loss = train_network(autoencoder, data, training_cycles=training_cycles)

    if plot:
        # show the loss curve
        plt.figure(0)
        plt.plot(loss)

        # get an example state and action to pass through network; get latent state, next state prediction
        index = np.random.choice(len(data[0]))
        sample_state = data[0][index]
        sample_action = data[1][index]
        latent_state, __, reconstructed_state = autoencoder(sample_state, sample_action)

        actions = ['D', 'U', 'R', 'L']
        s = np.where(sample_state[0]==1)[0][0]
        a = actions[np.where(sample_action[0]==1)[0][0]]
        sprime = np.argmax(reconstructed_state.detach().numpy())

        # plot state, latent state, predicted next state
        plt.figure(1)
        plt.plot(loss)

        fig, ax = plt.subplots(3,1,sharex=True)
        #ax0 = state
        #ax1 = latent
        #ax2 = reconstruction
        ax[0].imshow(sample_state, aspect='auto')
        ax[0].set_ylabel('State')
        ax[1].imshow(latent_state.detach().numpy(), aspect='auto')
        ax[1].set_ylabel('Latent')
        ax[2].imshow(reconstructed_state.detach().numpy(), aspect='auto')
        ax[2].set_ylabel('Reconst.')
        ax[0].set_title(f'S:{s}, A:{a}, predicted S\':{sprime} ')


        plt.show()

    return loss







