import gym
import pickle
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import colorsys

from basic.modules.Utils import running_mean as rm
from basic.modules.Utils.gridworld_plotting import plot_polmap, plot_pref_pol, plot_valmap, plot_world

run_id = '4544fdb3-98b0-4f31-8874-a6f50c4473b2' #'df092c0a-5478-41e1-a34a-43ecdb53262f'
env_id = 'gym_grid:gridworld-v1'

env = gym.make(env_id)
plt.close()

with open(f'../Data/results/{run_id}_data.p', 'rb') as f:
    data = pickle.load(f)
for key in data.keys():
    print(key, len(data[key]))
def plot_maps(start, stop, step=5, policy=False, pref_pol=False, value=False):
    for index in range(start, stop, step):
        mf_pol = data['P_snap'][index]
        mf_val = data['V_snap'][index]
        ec_pol = data['EC_snap'][index]

        if pref_pol:
            plot_pref_pol(env, mf_pol,save=True, directory='./figures/maps/pref_pol/', title=f'{run_id[0:8]}_{index}',show=False)
        if policy:
            plot_polmap(env, mf_pol,save=True, directory='./figures/maps/policy/', title=f'{run_id[0:8]}_{index}',show=False)
        if value:
            plot_valmap(env, mf_val,save=True, directory='./figures/maps/value/', title=f'{run_id[0:8]}_{index}',show=False, v_range=[-2.5,10])
        plt.close()

def trajectories(index):
    traj = data['trajectories'][index]

    map_plot = plot_world(env, plotNow=False)

    N = len(traj)
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))

    for event in range(len(traj)-1):
        coord1 = np.add(env.oneD2twoD(traj[event][0]), (0.5,0.5))
        coord2 = np.add(env.oneD2twoD(traj[event+1][0]), (0.5,0.5))
        #map_plot.gca().add_patch(plt.Circle(np.add(coord, (0.5,0.5)), radius=0.5, alpha=.5))
        # arrows args are x,y, coords are r,c
        map_plot.gca().add_patch(plt.Arrow(coord1[1],coord1[0], coord2[1]-coord1[1], coord2[0]-coord1[0], color=RGB_tuples[event], alpha = 0.2))
    plt.show()

def plot_pol_evol(coord):
    policy_ev = []
    for x in range(len(data['P_snap'])):
        policy = data['P_snap'][x][coord[0],coord[1]]
        policy_ev.append(policy)
    pol = np.array([list(x) for x in policy_ev]).T
    plt.figure()
    plt.stackplot(np.arange(1000), pol, labels=['D','U','R','L'])
    plt.legend(loc=0)
    plt.show()

fig, ax = plt.subplots(2,1, sharex=True)
smoothing=50
ax[0].plot(rm(data['total_reward'],smoothing), c='k', alpha=0.5)
ax[0].plot(rm(data['bootstrap_reward'],smoothing), c='r')
ax[1].plot(data['loss'][0], label='p')
ax[1].plot(data['loss'][1], label='v')
ax[1].legend(loc=0)
plt.show()

plot_maps(start=0,stop=999, step=10, policy=True)
#plot_pol_evol((5,6))
x = trajectories(995)
