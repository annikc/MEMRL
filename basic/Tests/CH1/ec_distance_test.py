import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('../../modules')
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from modules.Agents.RepresentationLearning.learned_representations import onehot, random, saved_latents, sr, place_cell
import gym
from scipy.spatial.distance import cdist, pdist, squareform


#parameters
env_name = 'gridworld:gridworld-v1'
example_dict = {'onehot':'db9fc8c4-6077-4740-b8a2-722ed762100c',
                'latents':'08ce120d-bf47-460f-9eb3-6b2f1293b50f',
                'sr': 'f1e484b4-3cf3-453b-a3d3-c8db0a1c73ca',
                'random': '3ca7b966-ae4e-481e-9ccf-93f904820921',
                'place_cell':'2caa8b36-adb4-46e4-8345-b94c68afabe6'}

rep_name = 'place_cell'
run_id = example_dict[rep_name]

# generate environment object
env = gym.make(env_name)
plt.close()

# load in episodic dictionary

with open(f'../../Data/ec_dicts/{run_id}_EC.p', 'rb') as f:
    cache_list = pickle.load(f)

memory = Memory(entry_size=env.action_space.n, cache_limit=400)
memory.cache_list = cache_list

# representations
rep_types = {'onehot':onehot, 'random':random, 'place_cell':place_cell, 'sr':sr, 'latents':saved_latents}
state_reps, name, dim, _ = rep_types[rep_name](env)



memory_ = np.zeros(env.shape)
i = 10
rep = state_reps[i]

for key in memory.cache_list.keys():
    state_id = memory.cache_list[key][-1]
    distance = np.linalg.norm(key-rep)
    memory_[env.oneD2twoD(state_id)]=distance

metric = 'canberra'
alt_dist = cdist([rep], list(state_reps.values()), metric=metric)


fig, ax = plt.subplots(6,5)
ax[0,0].axis('off')
ax[0,1].axis('off')
ax[0,2].imshow(memory_)
ax[0,2].set_title(f'EC dist \n(L1 Norm)')
ax[1,0].imshow(alt_dist[0,:].reshape(env.shape))
ax[1,0].set_title(f'Test Metric \n({metric})')
plt.suptitle(f'{rep_name}')
plt.show()

'''
index = 10
k, ec_l1, distance = memory.L1_norm(state_reps[index])
print(np.where(np.asarray(k)==1), distance, ec_l1)


sp_dist = cdist([state_reps[index]], list(memory.cache_list.keys()),metric='cityblock' )

'''