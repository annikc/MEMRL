import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle
import torch

import sys
sys.path.append('../../modules')
from scipy.spatial.distance import pdist, squareform
from modules.Agents.RepresentationLearning.learned_representations import onehot, random, place_cell, sr, load_saved_latents

rep_types = {'random':random, 'onehot':onehot, 'latent':load_saved_latents, 'place cell':place_cell, 'sr':sr }

sys.path.append('../../../')


version = 3
env_name = f'gridworld:gridworld-v{version}'
representation_type = 'latent'


# make gym environment
env = gym.make(env_name)
plt.close()

def plot_all_reps():
    fig, ax = plt.subplots(3,len(list(rep_types.keys())), sharey='row',sharex='row')
    for i, representation_type in enumerate(rep_types):
        state_reps, representation_name, input_dims, _ = rep_types[representation_type](env)

        representation_matrix = np.zeros((env.nstates, env.nstates))
        for key, value in state_reps.items():
            representation_matrix[key] = value/np.max(value)

        sim_matrix = squareform(pdist(representation_matrix, metric='cosine'))
        ax[0,i].imshow(representation_matrix)

        ax[1,i].imshow(sim_matrix)
        ax[2,i].imshow(sim_matrix[90].reshape(env.shape))
        ax[0,i].set_title(representation_type)
    ax[0,0].set_ylabel('All States')
    ax[1,0].set_ylabel('Similarity Matrix')
    ax[2,0].set_ylabel('Distance \n from state (4,10)')
    plt.show()


state_reps, representation_name, input_dims, _ = rep_types[representation_type](env)
representation_matrix = np.zeros((env.nstates, env.nstates))
for key, value in state_reps.items():
    representation_matrix[key] = value/np.max(value)

if version==3:
    rooms = []
    states_by_room = []
    for i in np.arange(0,20,2)*10: #room1
        for j in np.arange(i,i+10):
            states_by_room.append(j)
    states_by_room.append(90)
    for i in (np.arange(0,20,2)*10): #room2
        for j in np.arange(i+11,i+20):
            states_by_room.append(j)
    states_by_room.append(215)
    for i in np.arange(22,40,2)*10: #room3
        for j in np.arange(i+11,i+20):
            states_by_room.append(j)
    states_by_room.append(270)
    for i in np.arange(22,40,2)*10: #room3
        for j in np.arange(i,i+10):
            states_by_room.append(j)
    states_by_room.append(220)

    idx = states_by_room + env.obstacle

    reordered_representation_matrix = np.zeros((env.nstates, env.nstates))
    reordered_representation_matrix[:] = np.nan
    for i, index in enumerate(states_by_room):
        value = state_reps[index]
        reordered_representation_matrix[i] = (value/np.max(value))[idx]

    plt.figure()
    a = plt.imshow(reordered_representation_matrix,interpolation='none')
    plt.colorbar(a)
    plt.show()




plot_all_reps()











'''
plt.figure(figsize=(6,0.75))
plt.imshow([representation_matrix[90]],aspect='auto')
plt.show()


cov = np.linalg.inv(np.cov(representation_matrix)) # for mahalanobis measure
distance_measure = squareform(pdist(representation_matrix, 'cosine'))



# plot total distance matrix
plt.figure()
s =env.nstates
a = plt.imshow(representation_matrix[0:s,0:s])
plt.colorbar(a)
plt.gca().get_xaxis().set_visible(False)
#plt.gca().get_yaxis().set_visible(False)
plt.show()

# plot single point distance to other points
plt.figure()
a = plt.imshow((distance_measure[310]/max(distance_measure[310])).reshape(env.shape))
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
plt.colorbar(a)
plt.show()'''