# import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import sys
import torch
import pickle
from scipy.spatial.distance import euclidean, mahalanobis, pdist, squareform
sys.path.insert(0, '../../../modules/')
from modules.Utils import running_mean as rm

# get environment
import gym

data_dir = '../../Data/'
filename = 'conv_mf_training.csv'

# what env representations will we look at
env_name = 'gridworld:gridworld-v1'
# generate env object
env = gym.make(env_name)
plt.close()

# load representations
with open(f'../../modules/Agents/RepresentationLearning/Learned_Rep_pickles/conv_{env_name[-12:]}.p', 'rb') as f:
    conv_representations = pickle.load(f)

with open(f'../../modules/Agents/RepresentationLearning/Learned_Rep_pickles/sr_{env_name[-12:]}.p', 'rb') as f:
    sr_representations = pickle.load(f)

with open(f'../../modules/Agents/RepresentationLearning/Learned_Rep_pickles/place_cell_{env_name[-12:]}.p', 'rb') as f:
    pc_representations = pickle.load(f)

def RSA_L1(representations):
    distance_matrix = np.zeros(representations.shape)
    for i in range(representations.shape[0]):
        r = np.nan_to_num(representations[i])
        reference_ = r/np.linalg.norm(r)
        for j in range(representations.shape[0]):
            q = np.nan_to_num(representations[j])
            comparison_ = q/np.linalg.norm(q)

            l1_dist = np.linalg.norm(reference_ - comparison_)
            distance_matrix[i,j] = l1_dist
    return distance_matrix

def RSA_L2(representations):
    distance_matrix = np.zeros(representations.shape)
    for i in range(representations.shape[0]):
        r = np.nan_to_num(representations[i])
        if np.linalg.norm(r) ==0:
            reference_ = r
        else:
            reference_ = r/np.linalg.norm(r)
        for j in range(representations.shape[0]):
            print(i,j)
            q = np.nan_to_num(representations[j])
            if np.linalg.norm(q) ==0:
                comparison_ = q
            else:
                comparison_ = q/np.linalg.norm(q)

            l2_dist = euclidean(reference_,comparison_)
            distance_matrix[i,j] = l2_dist
    return distance_matrix

def RSA_mahalanobis(representations):
    mean_vec = np.mean(representations, axis=0)
    inv_cov = np.linalg.inv(np.cov(representations))

    dist = pdist(representations,'mahalanobis', VI=inv_cov)
    return dist







''' distance_matrix = np.zeros(representations.shape)
    for i in range(representations.shape[0]):
        r = np.nan_to_num(representations[i])
        if np.linalg.norm(r) ==0:
            reference_ = r
        else:
            reference_ = r/np.linalg.norm(r)
        for j in range(representations.shape[0]):
            q = np.nan_to_num(representations[j])
            if np.linalg.norm(q) ==0:
                comparison_ = q
            else:
                comparison_ = q/np.linalg.norm(q)

            print(i,j)

            m_dist = mahalanobis(r,q,inv_cov)
            distance_matrix[i,j] = m_dist
    return distance_matrix
'''

def L1_norm(ref_index, representations):
    distance = []
    ref_rep = representations[ref_index]/np.linalg.norm(representations[ref_index])
    for i in range(representations.shape[0]):
        norm_rep = representations[i]/np.linalg.norm(representations[i])
        l1dist = euclidean(ref_rep,norm_rep)  #np.linalg.norm(ref_rep-norm_rep)
        distance.append(l1dist)

    return np.asarray(distance)

def euc_dist(ref_index, representations):
    xy = env.oneD2twoD(ref_index)
    print(xy)

    dist = []

    for i in range(representations.shape[0]):
        ab = env.oneD2twoD(i)
        dist.append(euclidean(xy,ab))
    return np.asarray(dist)

representations = pc_representations
l1_dist = squareform(pdist(representations, 'mahalanobis', VI=np.linalg.inv(np.cov(representations)))) # manhattan distance
#l2_dist = squareform(pdist(representations, 'minkowski', p=2.)) # euclidean distance
l2_dist = squareform(pdist(representations, 'cosine'))

fig, ax = plt.subplots(1,2, sharex=True, sharey=True)
a = ax[0].imshow(l1_dist)
b = ax[1].imshow(l2_dist/np.nanmax(l2_dist))
plt.colorbar(b)

fig1, ax1 = plt.subplots(1,2, sharex=True, sharey=True)
ax1[0].imshow(l1_dist[90].reshape(env.shape))
ax1[1].imshow(l2_dist[90].reshape(env.shape))
plt.show()


"""
dist = RSA_L2(pc_representations)
print(dist.shape)
fig, axes = plt.subplots(1,2)
axes[0].imshow(pc_representations)
ax = axes[1].imshow(dist, aspect='auto')
plt.colorbar(ax)
plt.show()



conv_dist = RSA_L2(conv_representations)
sr_dist = RSA_L2(SR_representations)

fix, ax = plt.subplots(1,2)
ax[0].imshow(conv_dist)
ax[1].imshow(sr_dist)
plt.show()"""

'''a_d = euc_dist(0, representations)
a_n = L1_norm(0, representations)

b_d = euc_dist(379, representations)
b_n = L1_norm(379, representations)

c_d = euc_dist(399, representations)
c_n = L1_norm(399, representations)

plt.figure()
plt.scatter(a_d, a_n)
plt.scatter(b_d, b_n)
plt.scatter(c_d, c_n)
plt.show()'''
