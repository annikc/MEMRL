import numpy as np
import gym
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from matplotlib import cm

representation = 'sr'
version        = 3

run_id = '17ad33db-ed26-4db4-9091-d067ab43f72c' # sr, 1000 trials, env 31, dist euclidean, cache 100
with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
    data = pickle.load(f)

env = gym.make(f'gridworld:gridworld-v{version}1')


def make_env_graph(env):
    action_cols = ['orange','red','green','blue']
    G = nx.DiGraph() # why directed/undirected graph?

    for action in range(env.action_space.n):
        # down, up, right, left
        for state2d in env.useable:
            state1d = env.twoD2oneD(state2d)
            next_state = np.where(env.P[action,state1d,:]==1)[0]
            if len(next_state) ==0:
                pass
            else:
                for sprime in next_state:
                    edge_weight = env.P[action,state1d,sprime]
                    G.add_edge(state1d, sprime,color=action_cols[action],weight=edge_weight)

    return G

def compute_distance_matrix(G, env):
    x = nx.shortest_path(G)
    useable_1d = [env.twoD2oneD(x) for x in env.useable]
    shortest_dist_array = np.zeros((env.nstates,env.nstates))
    shortest_dist_array[:]=np.nan

    for start in useable_1d:
        for target in list(x[start].keys()):
            shortest_dist_array[start][target]= len(x[start][target])-1

    return shortest_dist_array


G= make_env_graph(env)
shortest_dist_array = compute_distance_matrix(G,env)
graphdist = nx.shortest_path(G)


d, r = [], []
viridis = cm.get_cmap('viridis', 500)
nums = np.linspace(0,1,500)
cs = [viridis(n) for n in nums]
print(cs)
for i in range(500):
    dist_returns = data['dist_rtn'][i]

    states_              = dist_returns[0]
    reconstructed_states = dist_returns[1]
    ec_distances         = dist_returns[2]
    computed_returns     = dist_returns[3]


    avg_dist = np.mean(ec_distances)
    avg_rtrn = np.mean(computed_returns)
    d.append(avg_dist)
    r.append(avg_rtrn)


plt.figure()
ax1 = plt.scatter(d,r,c=cs)
plt.colorbar(ax1)
plt.show()
