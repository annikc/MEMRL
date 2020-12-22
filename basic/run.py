## write an example agent and show that it does stuff
# import statements
import torch
import gym
import numpy as np
import modules.Agents.Networks as nets
from modules.Agents import Agent
from Tests.agent_test import agent_test
from Tests.representation_learn_test import rep_learning, latent_space_distance

import matplotlib.pyplot as plt


# create environment
env = gym.make('gym_grid:gridworld-v1')
plt.close()
# generate parameters for network from environment observation shape
params = nets.params(env)
# generate network
network = nets.ActorCritic(params)



agent = Agent(network, memory=None)

autoencoder, data, loss = rep_learning('onehot', env, n_samples=1000, training_cycles=500)

states = []

for i in range(env.nstates):
    s = np.zeros((1,env.nstates))
    s[0, i] = 1
    states.append(s)

actions = data[1][0:400]
latent_states, _, __ = autoencoder(states, actions)

state_distance, latent_distance = [], []
l_grid = np.zeros((20,20))
l_grid[:] = np.nan
for i in range(1):
    ref_state = states[i]
    ref_latent = latent_states[i].detach().numpy()

    for j in range(len(states)):
        test_state = states[j]
        test_latent = latent_states[j].detach().numpy()

        dist_state = np.linalg.norm(ref_state - test_state)
        dist_latent = np.linalg.norm(ref_latent - test_latent)

        state_distance.append(dist_state)
        cossim = np.dot(ref_latent, test_latent.T) / (np.linalg.norm(ref_latent) * np.linalg.norm(test_latent))
        s = np.where(test_state == 1)[1][0]
        print(f'{np.where(ref_state==1)[1][0]}:{np.where(test_state == 1)[1][0]}, {dist_state} / {1-cossim}')
        coord = env.oneD2twoD(s)
        l_grid[coord[0],coord[1]] = cossim
        latent_distance.append(dist_latent)

plt.figure()
plt.scatter(state_distance, latent_distance, alpha=0.3)

plt.figure()
plt.imshow(l_grid)
plt.show()



#latent_space_distance(autoencoder, data)


### JUNKYARD
#network = nets.FC(params)
#network = nets.CNN_AC(params)
#network = nets.CNN_2N(params)
#network = nets.FC2N()