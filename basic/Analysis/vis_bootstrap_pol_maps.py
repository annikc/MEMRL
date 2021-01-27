import gym
import pickle
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import colorsys
import os
from basic.modules.Utils import running_mean as rm
from basic.modules.Utils.gridworld_plotting import plot_polmap, plot_pref_pol, plot_valmap, plot_world

run_id = 'f96cadce-7d2c-4e5e-a6a4-ff075c49eb82' #'df092c0a-5478-41e1-a34a-43ecdb53262f' #'466ae988-ea1b-4d73-8d24-ca783f941153' #'df092c0a-5478-41e1-a34a-43ecdb53262f'
env_id = 'gym_grid:gridworld-v1'

env = gym.make(env_id)
plt.close()

with open(f'../Data/results/{run_id}_data.p', 'rb') as f:
    data = pickle.load(f)
for key in data.keys():
    print(key, len(data[key]))
def plot_maps(start, stop, step=5, policy=False, pref_pol=False, value=False):
    if policy:
        parent_dir = './figures/maps/policy/'
    if pref_pol:
        parent_dir = './figures/maps/pref_pol/'
    if value:
        parent_dir = './figures/maps/value/'
    dir = parent_dir + f'{run_id[0:8]}/'

    if not os.path.exists(dir):
        os.makedirs(dir)
    for index in range(start, stop, step):
        mf_pol = data['P_snap'][index]
        mf_val = data['V_snap'][index]
        #ec_pol = data['EC_snap'][index]

        if pref_pol:
            plot_pref_pol(env, mf_pol,save=True, directory=dir, title=f'{index}',show=False)
        if policy:
            plot_polmap(env, mf_pol,save=True, directory=dir, title=f'{index}',show=False)
        if value:
            plot_valmap(env, mf_val,save=True, directory=dir, title=f'{run_id[0:8]}_{index}',show=False, v_range=[-2.5,10])
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
    plt.stackplot(np.arange(len(data['P_snap'])), pol, labels=['D','U','R','L'])
    plt.legend(loc=0)
    plt.title(f'{coord}')
    plt.show()

def daves_idea():
    fig, ax = plt.subplots(4,1,sharex=True)
    num_spots = 3
    rand_spots = np.random.choice(np.arange(data['P_snap'][0].shape[0]*data['P_snap'][0].shape[0]),num_spots)
    rand_spots = [env.twoD2oneD(x) for x in [(0,1), (0,2), (0,3), (0,4), (0,5)]]
    dats = np.asarray(data['P_snap'])
    DURL = ['Down', 'Up', 'Right', 'Left']

    for ind in range(len(rand_spots)):
        coord = env.oneD2twoD(rand_spots[ind])
        print(ind, coord)
        for n, key in enumerate(DURL):
            print(n,key)
            action_change = np.array(dats[:,coord[0], coord[1]][key])
            ax[n].plot(action_change, label=f'{coord}')

    for n, key in enumerate(DURL):
        ax[n].set_ylabel(key)
        ax[n].set_ylim([0,1.1])
    ax[0].legend(loc=0)
    plt.show()
#daves_idea()

def plot_rewards():
    fig, ax = plt.subplots(4,1, sharex=True)
    smoothing=10
    ax[0].plot(rm(data['total_reward'],smoothing), c='k', alpha=0.5)
    if 'bootstrap_reward' in data.keys():
        ax[0].plot(rm(data['bootstrap_reward'],smoothing), c='r')

    ax[1].plot(rm(data['loss'][0], smoothing), label='ec_p')
    ax[1].plot(rm(data['mf_loss'][0], smoothing), label='mf_p')
    #ax[1].set_ylim([-1000,1000])
    #ax[1].set_yscale('log')
    ax[1].legend(loc=0)

    ax[2].plot(rm(data['loss'][1],smoothing), label='ec_v')
    ax[2].plot(rm(data['mf_loss'][1],smoothing), label='mf_v')
    #ax[2].set_yscale('log')
    ax[2].legend(loc=0)



    #ax[2].plot(np.asarray(data['weights']['h0'][0]), label='h0',c='r')
    #ax[2].plot(np.asarray(data['weights']['h1'][0]), label='h1',c='g')
    #ax[3].plot(np.vstack(data['weights']['out0'][0]), label='p',c='b')
    #ax[3].plot(np.vstack(data['weights']['out1'][0]), label='v',c='k')

    #ax[2].plot(data['weights']['h0'][1], ':', label='h0',c='r')
    #ax[2].plot(data['weights']['h1'][1], ':', label='h1',c='g')
    #ax[2].plot(data['weights']['out0'][1], ':', label='p',c='b')
    #ax[2].plot(data['weights']['out1'][1], ':', label='v',c='k')
    ax[2].legend(loc=0)
    plt.show()

plot_rewards()

'''
## show weights 
for index in range(0,999,5):
    plt.figure()
    weights = data['weights'][0][index]
    wt_plot = plt.imshow(weights.T, aspect='auto', interpolation='none', vmin=np.min(data['weights'][0]), vmax=np.max(data['weights'][0]))
    plt.colorbar(wt_plot)
    plt.savefig(f'../Analysis/figures/{index}.svg', format='svg')
    plt.close()
'''

'''
## violin plots of weights at topmost layer
dats = []
poss = []
i = 0
for x in range(0,500,50):
    poss.append(i)
    dats.append(data['weights'][0][x].flatten())
    i+=1

plt.figure()
plt.violinplot(dats, poss)
plt.show()
'''


'''
## show correlation between loss and reward
xdat = np.asarray(data['total_reward'][100:200])

yp = np.asarray(data['loss'][0][0:len(xdat)])
yv = np.asarray(data['loss'][1][0:len(xdat)])

N = len(xdat)
HSV_tuples = [(x*1.0/N, 0.7, 0.5) for x in range(N)]
RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))


plt.figure()
#plt.scatter(x=xdat,y=yp, label='pol', color=RGB_tuples)
#plt.scatter(x=np.arange(len(xdat))/80,y=700*np.ones(len(xdat)),color=RGB_tuples)
plt.scatter(xdat,yv, label='val', color=RGB_tuples)
plt.legend(loc=0)
plt.show()
'''




#plot_maps(start=0,stop=550, step=1, policy=True)
#plot_pol_evol((5,6))
#x = trajectories(995)
