# import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import euclidean, mahalanobis, pdist, squareform
from gridworld_as_graph import make_env_graph, compute_distance_matrix
import gym
import pickle
import sys
sys.path.append('../../modules')
sys.path.append('../../../')

version = 1
latent_type = 'rwd_conv'
training_env_name = f'gridworld:gridworld-v{version}'
test_env_name = training_env_name+'1'

env = gym.make(test_env_name)
plt.close()

# compute distance_array from graph
G = make_env_graph(env)
shortest_dist_array = compute_distance_matrix(G,env)

print(shortest_dist_array.shape)


all_reps = []
all_sims = []

for latent_type in ['conv','rwd_conv']:
    with open(f'../../modules/Agents/RepresentationLearning/Learned_Rep_pickles/{latent_type}_{training_env_name[-12:]}.p', 'rb') as f:
        state_reps = pickle.load(f)

    representation_array = np.zeros((env.nstates,env.nstates))
    for state2d in env.useable:
        state1d = env.twoD2oneD(state2d)
        representation_array[state1d] = state_reps[state1d]
    all_reps.append(representation_array)

    similarity_array = squareform(pdist(representation_array, metric='chebyshev'))
    all_sims.append(similarity_array)


'''
fig, ax = plt.subplots(3,2)
for i in range(2):
    ax[0,i].imshow(all_reps[i])
    loc1 = 105
    (r,c) = env.oneD2twoD(loc1)
    sim_scaled_by_dist = np.true_divide(all_sims[i][loc1], shortest_dist_array[loc1])
    ax[1,i].imshow(sim_scaled_by_dist.reshape(env.shape))
    ax[1,i].add_patch(plt.Rectangle(np.add((c,r),(-0.5,-0.5)), 1, 1, edgecolor='w', fill=False, alpha=1))
    loc2 = 355
    (r,c) = env.oneD2twoD(loc2)
    sim_scaled_by_dist = np.true_divide(all_sims[i][loc2], shortest_dist_array[loc2])
    ax[2,i].imshow(sim_scaled_by_dist.reshape(env.shape))
    ax[2,i].add_patch(plt.Rectangle(np.add((c,r),(-0.5,-0.5)), 1, 1, edgecolor='w', fill=False, alpha=1))
plt.show()
'''

def plot_sim_matrices(list_of_envs, representation_type, scale=400, save=True, metric='cosine'):
    n_envs = len(list_of_envs)
    grids = []
    fig, ax = plt.subplots(n_envs, 5, figsize=(15,10),sharex='col')
    for i in range(n_envs):
        env = gym.make(list_of_envs[i])
        plt.close()

        #plot environment
        ax[i,0].pcolor(env.grid, cmap='bone_r', edgecolors='k', linewidths=0.1)
        (rwd_r, rwd_c) = list(env.rewards.keys())[0]
        ax[i,0].add_patch(plt.Rectangle((rwd_c,rwd_r),1,1,color='g',alpha=0.3))
        ax[i,0].axis(xmin=0, xmax=20, ymin=0,ymax=20)
        ax[i,0].set_aspect('equal')
        ax[i,0].get_xaxis().set_visible(False)
        ax[i,0].get_yaxis().set_visible(False)
        ax[i,0].invert_yaxis()

        # show distance in state space
        G = make_env_graph(env)
        shortest_dist_array = compute_distance_matrix(G,env)
        ax[i,1].imshow(shortest_dist_array[0:scale,0:scale])

        # show distance in latent space
        if representation_type in ['conv', 'rwd_conv']:
            latent_type = representation_type
            with open(f'../../modules/Agents/RepresentationLearning/Learned_Rep_pickles/{latent_type}_{list_of_envs[i][-13:-1]}.p', 'rb') as f:
                state_reps = pickle.load(f)

        representation_array = np.zeros((env.nstates,env.nstates))
        for state2d in env.useable:
            state1d = env.twoD2oneD(state2d)
            representation_array[state1d] = state_reps[state1d]

        similarity_array = squareform(pdist(representation_array, metric=metric))
        ax[i,2].imshow(similarity_array[0:scale,0:scale])

        # show a single state
        loc1 = 0 #25
        normalized_state_sim = similarity_array[loc1]/np.nanmax(similarity_array[loc1])
        ax[i,3].imshow(normalized_state_sim.reshape(env.shape))
        (r,c) = env.oneD2twoD(loc1)
        ax[i,3].add_patch(plt.Rectangle(np.add((c,r),(-0.5,-0.5)), 1, 1, edgecolor='w', fill=False, alpha=1))
        ax[i,3].set_aspect('equal')
        ax[i,3].get_xaxis().set_visible(False)
        ax[i,3].get_yaxis().set_visible(False)

        # show a single state
        loc2 = 375
        normalized_state_sim = similarity_array[loc2]/np.nanmax(similarity_array[loc2])
        ax[i,4].imshow(normalized_state_sim.reshape(env.shape))
        (r,c) = env.oneD2twoD(loc2)
        ax[i,4].add_patch(plt.Rectangle(np.add((c,r),(-0.5,-0.5)), 1, 1, edgecolor='w', fill=False, alpha=1))
        ax[i,4].set_aspect('equal')
        ax[i,4].get_xaxis().set_visible(False)
        ax[i,4].get_yaxis().set_visible(False)


        '''
        # show state sim scaled by true distance
        sim_scaled_by_dist = np.true_divide(normalized_state_sim, shortest_dist_array[loc])
        ax[i,4].imshow(sim_scaled_by_dist.reshape(env.shape))
        ax[i,4].add_patch(plt.Rectangle(np.add((c,r),(-0.5,-0.5)), 1, 1, edgecolor='w', fill=False, alpha=1))
        ax[i,4].set_aspect('equal')
        ax[i,4].get_xaxis().set_visible(False)
        ax[i,4].get_yaxis().set_visible(False)
        '''

    ax[0,1].set_title('Shortest Path \n On Graph')
    ax[0,2].set_title('Latent State \n Distance')
    ax[0,3].set_title(f'Distance from \n State {env.oneD2twoD(loc1)}')
    ax[0,4].set_title(f'Distance from \n State {env.oneD2twoD(loc2)}')

    if save:
        save_format = 'png'
        plt.savefig(f'../figures/CH1/representation_similarity/{latent_type}_{metric}_similarity.{save_format}',format=save_format)
    plt.show()


list_of_envs = [f'gridworld:gridworld-v{version}' for version in [11,41,31,51]]

for rep in ['conv', 'rwd_conv']:
    for metric in ['cosine','euclidean']:
        plot_sim_matrices(list_of_envs, representation_type=rep, save=False, scale=50, metric=metric)





















#########################################################
'''
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

'''



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

'''
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

'''
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
