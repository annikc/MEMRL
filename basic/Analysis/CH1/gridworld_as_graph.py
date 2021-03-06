import numpy as np
import gym
import matplotlib.pyplot as plt
import networkx as nx

env_id = 'gridworld:gridworld-v41'
env = gym.make(env_id)
plt.close()
reward_state = env.twoD2oneD(list(env.rewards.keys())[0])
print(env.rewards,reward_state)

# plot transition graph

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
            shortest_dist_array[start][target]= len(x[start][target])

    return shortest_dist_array


G= make_env_graph(env)
shortest_dist_array = compute_distance_matrix(G,env)
print(10- 0.01*np.nanmean(shortest_dist_array[reward_state]))

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(shortest_dist_array[reward_state].reshape(env.shape))
plt.show()
'''
if env_id[-1]=='3':
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
    
    G= make_env_graph(env)
    shortest_dist_array = compute_distance_matrix(G,env)
    
    shortest_dist_by_room = np.zeros((env.nstates,env.nstates))
    shortest_dist_by_room[:] = np.nan
    idx = states_by_room+env.obstacle
    for ind in range(len(states_by_room)):
        shortest_dist_by_room[ind] = shortest_dist_array[states_by_room[ind]][idx]/(np.nanmax(shortest_dist_array[states_by_room[ind]]))

    plt.figure()
    a = plt.imshow(shortest_dist_by_room,interpolation='none')
    plt.colorbar(a)
    plt.show()


G= make_env_graph(env)
shortest_dist_array = compute_distance_matrix(G,env)

print(env.twoD2oneD((10,16)))
fig,ax = plt.subplots(3,1)
a= ax[0].imshow(shortest_dist_array)
ax[1].imshow(shortest_dist_array[90].reshape(env.shape))
ax[2].imshow(shortest_dist_array[:,env.twoD2oneD(list(env.rewards.keys())[0])].reshape(env.shape))
ax[0].get_xaxis().set_visible(False)
ax[0].get_yaxis().set_visible(False)
ax[1].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)
ax[2].get_xaxis().set_visible(False)
ax[2].get_yaxis().set_visible(False)
ax[0].set_ylabel('All State\nDistance')
ax[1].set_ylabel('Distance \nfrom (4,10)')
ax[2].set_ylabel('Distance \nto (5,5)')
plt.show()

'''