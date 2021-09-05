import numpy as np
import matplotlib.pyplot as plt
import pickle
import gym
import pandas as pd

from Analysis.analysis_utils import make_env_graph, compute_graph_distance_matrix, LINCLAB_COLS
env_name = 'gridworld:gridworld-v4'
env = gym.make(env_name)
plt.close()
df = pd.read_csv('../../Data/conv_mf_training_with_occ.csv')
dfgb = df.groupby(['env_name'])['save_id']
id_list = list(dfgb.get_group((env_name)))
print(id_list)
state_occ_map = np.zeros((20,20))
all_occ = np.zeros(400)
all_v =[]

for id_num in id_list:
    with open(f'../../Data/results/{id_num}_data.p', 'rb') as f:
        data = pickle.load(f)

    state_occ_vec = data['occupancy']
    all_occ += state_occ_vec
    all_visits = np.sum(state_occ_vec)
    all_v.append(all_visits)
    print(all_occ)

state_occ_map[:] = np.nan
for r,c in env.useable:
    ind = env.twoD2oneD((r,c))
    state_occ_map[r,c] = all_occ[ind]

G = make_env_graph(env)
gd = compute_graph_distance_matrix(G, env)

dist_from_reward = gd[:,105]

#a = plt.imshow(state_occ_map/(all_visits/len(env.useable)),vmax=4)
#plt.colorbar(a)
#plt.show()
print(np.nanmax(dist_from_reward))
janky_histo = np.zeros(int(np.nanmax(dist_from_reward))+1)
num_times_for_dist = np.zeros(int(np.nanmax(dist_from_reward))+1)

for ind in range(len(state_occ_vec)):
    print(dist_from_reward[ind])
    if np.isnan(dist_from_reward[ind]):
        pass
    else:
        distance = int(dist_from_reward[ind])
        num_times_for_dist[distance] +=1
        janky_histo[distance] += all_occ[ind]

plt.figure()
scaled_histo = [janky_histo[x]/num_times_for_dist[x] for x in range(len(janky_histo))]
plt.bar(np.arange(len(janky_histo)), scaled_histo,color=LINCLAB_COLS['purple'])
#plt.savefig('../figures/CH1/state_occupancy_histo.svg')

plt.figure()
plt.imshow(all_occ.reshape(20,20))

plt.figure()
a = plt.imshow(dist_from_reward.reshape(20,20),cmap ='Spectral_r')
plt.colorbar(a)
#plt.savefig('../figures/CH1/env4_geodesic.svg')
plt.show()