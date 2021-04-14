# import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import sys
import torch
import pickle
sys.path.insert(0, '../../../modules/')
from modules.Utils import running_mean as rm
from modules.Utils import one_hot_state, onehot_state_collection, twoD_states

# import representation type
from modules.Agents.RepresentationLearning import PlaceCells

# get environment
import gym


# make env
env_name = 'gridworld:gridworld-v4'
env = gym.make(env_name)
plt.close()

input_dims = 400

data_dir = '../../Data/'
load_id = 'b6f51c73-ebc0-467a-b5e5-5b51a5a3208d'

### place cell representations
place_cells = PlaceCells(env.shape, input_dims, field_size=.1)

# load place cells
with open(data_dir+ f'results/{load_id}_data.p', 'rb') as f:
    place_cells = (pickle.load(f))['place_cells']

pc_state_reps = {}
oh_state_reps = {}
for state in env.useable:
	oh_state_reps[env.twoD2oneD(state)] = one_hot_state(env,env.twoD2oneD(state))
	pc_state_reps[env.twoD2oneD(state)] = place_cells.get_activities([state])[0]

def similarity_(state_reps):
    cos_sim = np.zeros((len(state_reps), len(state_reps)))
    distance = np.zeros((len(state_reps), len(state_reps)))

    for i, x in enumerate(state_reps.values()):
        for ii, y in enumerate(state_reps.values()):
            cosine_similarity = np.dot(x, y)/ (np.linalg.norm(x) * np.linalg.norm(y))
            l1_norm = np.linalg.norm(x-y)
            cos_sim[i,ii] = cosine_similarity
            distance[i,ii] = l1_norm
    return cos_sim, distance

def plot_state_sim():
    oh_cos_similarity, oh_distance_meas = similarity(oh_state_reps)
    pc_cos_similarity, pc_distance_meas = similarity(pc_state_reps)

    fig, ax = plt.subplots(2,2)
    ax[0,0].imshow(oh_cos_similarity)
    ax[0,1].imshow(oh_distance_meas)

    ax[1,0].imshow(pc_cos_similarity)
    ax[1,1].imshow(pc_distance_meas)
    plt.show()


def laplace_of_gaussian(x,y,cell_centers, field_size):
    sig_x = field_size
    sig_y = field_size
    laplace = np.zeros(len(cell_centers))
    for i, center in enumerate(cell_centers):
        mu_x, mu_y = center
        trans_x = (x-mu_x)/sig_x
        trans_y = (y-mu_y)/sig_y
        A = 0.5 * (trans_x**2 + trans_y**2)
        laplace[i] = (1/(2*np.pi*sig_x*sig_y)) * np.exp(-A) * ((1/sig_x**2)*(trans_x**2 - 1) + (1/sig_y**2)*(trans_y**2 - 1))

    return laplace


laplace_tensor = np.zeros((place_cells.num_cells, *env.shape))
print(laplace_tensor.shape)
for (r,c) in env.useable:
    y,x = r/env.shape[0], c/env.shape[0] # scale to 0-1 interval
    laplace_tensor[:,r,c] = laplace_of_gaussian(x,y,place_cells.cell_centres, place_cells.field_size).copy()

some_rand_inds = place_cells.plot_placefields(env.useable)

fig, axes = plt.subplots(3,3)
for i, ax in enumerate(axes.flat):
    ax1 = ax.imshow(laplace_tensor[some_rand_inds[i]], vmin=-80, vmax=10)
    fig.colorbar(ax1, ax=ax)
    ax.set_title(f'{np.round(env.shape*place_cells.cell_centres[some_rand_inds[i]],2)}')
    ax.axis('off')

plt.figure()
plt.imshow(laplace_tensor[0:10,10,:]) #first 10 place cells, 10th row, all columns
plt.yticks(np.arange(10), labels=[str(env.shape*np.round(lab,2)) for lab in place_cells.cell_centres[0:10]])

plt.show()

