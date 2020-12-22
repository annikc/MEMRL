import gym

import numpy as np

import torch
import torch.nn as nn

import basic.modules.Agents.RepresentationLearning as rep
import matplotlib.pyplot as plt


env = gym.make('gym_grid:gridworld-v1')
plt.close() ## fix this so that env doesn't automatically make a plot

ohot_autoencoder = torch.load(f='../Data/onehot_autoencoder.pt')# rep.FC_OSFM()

test_states = []
test_actions = []
for i in range(env.nstates):
    state= np.zeros(env.nstates)
    state[i] =1
    test_states.append(state)
    dummy_action = np.zeros(env.action_space.n)
    test_actions.append(dummy_action)


latent_states, __, recs = ohot_autoencoder(test_states, test_actions)

fig, ax = plt.subplots(10,10,sharex=True, sharey=True)

for x in range(100):
    test_state0 = latent_states[x].detach().numpy()
    distances = np.zeros((20,20))
    for ind, i in enumerate(latent_states):
        l_state = i.detach().numpy()
        distance = np.linalg.norm(l_state - test_state0)
        coord = env.oneD2twoD(ind)
        distances[coord[0], coord[1]] = distance


plt.imshow(distances, aspect='auto')

plt.show()



'''
# get random sample of data
index = np.random.choice(len(ohot_data[0]))
sample_state = ohot_data[0][index]
sample_action = ohot_data[1][index]
latent_state, __, reconstructed_state = ohot_autoencoder(sample_state, sample_action)



actions = ['D', 'U', 'R', 'L']
s = np.where(sample_state[0]==1)[0][0]
a = actions[np.where(sample_action[0]==1)[0][0]]
sprime = np.argmax(reconstructed_state.detach().numpy())
## plotting
plt.figure(0)
plt.plot(ohot_loss)

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
'''