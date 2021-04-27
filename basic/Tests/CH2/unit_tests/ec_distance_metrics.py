# want to test different distance measures in EC module (comparing to linalg.norm() which is Frobenius norm
# 1. Cosine Distance
# 2. Chebyshev Distance
import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle
import sys

sys.path.append('../../../')
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from modules.Agents.RepresentationLearning.learned_representations import onehot, random, latents, sr, place_cell
from scipy.spatial.distance import cdist, pdist, squareform

parent_dir = '../../../Data/'
env_id     = 'gridworld:gridworld-v11'
rep_id     = 'analytic successor'
run_id     = 'eff8fc0c-3bc6-4005-b85f-06a9e49c7d2d'
filename   = f'{parent_dir}/ec_dicts/{run_id}_EC.p'

# make env
env = gym.make(env_id)
plt.close()


# make representations
representation_types = {'onehot':onehot, 'analytic successor':sr, 'conv_latents':latents, 'random':random, 'state-centred pc f0.05': place_cell}
state_reps = representation_types[rep_id](env)


# load an episodic cache
with open(filename, 'rb') as f:
    e_cache = pickle.load(f)
# try distance between probe key and keys in cache
print(e_cache[list(e_cache.keys())[0]])

# plot distance
ref_key = list(e_cache.keys())[0]
ref_state = e_cache[ref_key][2]
print("reference state: ", ref_state, env.oneD2twoD(ref_state))


entry     = np.asarray(ref_key)
mem_cache = np.asarray(list(e_cache.keys()))
sts_ids   = [e_cache[x][2] for x in list(e_cache.keys())]


distance_cos = cdist([entry], mem_cache, metric='chebyshev')[0]
print(min(distance_cos), "closest state is", e_cache[tuple(mem_cache[np.argmin(distance_cos)])][2])

dist = np.zeros(env.nstates)
dist[:] = np.nan
for ind,item in enumerate(sts_ids):
    dist[item] = distance_cos[ind]

fig, ax = plt.subplots(1,1)
ax.imshow(dist.reshape(env.shape))
plt.show()

memory = Memory(cache_limit=env.nstates, entry_size=env.action_space.n, distance='euclidean')
memory.cache_list = e_cache

closest_key, min_dist = memory.similarity_measure(ref_key)
print(f'ep recall closest dist = {min_dist}, state = {memory.cache_list[closest_key][2]}')

