import gym
import pickle
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt

from basic.modules.Utils import running_mean as rm
from basic.modules.Utils.gridworld_plotting import plot_polmap, plot_pref_pol, plot_valmap, plot_world
from basic.Analysis.vis_bootstrap_pol_maps import daves_idea, plot_pol_evol, trajectories, plot_maps, plot_rewards

filename = '../Data/linear_track.csv'
df = pd.read_csv(filename)
v_data = {}
v_norms = {'mf':{}, 'ecmf':{}}
for x in range(len(df)):
    run_id = df['run_id'].loc[x]
    lr = df['lr'].loc[x]
    if df['expt_type'].loc[x][0:9] == 'Bootstrap':
        dkey = 'ecmf'
    elif df['expt_type'].loc[x][0:9] == 'gridworld':
        dkey = 'mf'
    with open(f'../Data/results/{run_id}_data.p', 'rb') as f:
        data = pickle.load(f)
    v_data[run_id] = data

    norm = []
    print(len(data['V_snap']))
    for i in range(len(data['V_snap'])):
        norm.append(np.linalg.norm(data['opt_values']-data['V_snap'][i]))
    v_norms[dkey][str(lr)] = norm


fig, ax = plt.subplots(2,1,sharex=True)
for i in v_norms['mf'].keys():
    ax[0].plot(v_norms['mf'][i], label = i)
for i in v_norms['ecmf'].keys():
    ax[1].plot(v_norms['ecmf'][i], label = i)
ax[0].legend(loc=0)
ax[1].legend(loc=0)
plt.show()




'''
run_id = 'bfb88231-47b4-497a-aac9-8adcfd4d088d' # MF only = 'a691709c-e7c4-4e48-a32b-974c412c9426'
env_id = 'gym_grid:gridworld-v112'

env = gym.make(env_id)
plt.close()

with open(f'../Data/results/{run_id}_data.p', 'rb') as f:
    data = pickle.load(f)
for key in data.keys():
    print(key, len(data[key]))


def plot_vdif_norm(data,num=len(data['V_snap'])):
    norm = []
    smoothing = 1
    for i in range(num):
        norm.append(np.linalg.norm(data['opt_values']-data['V_snap'][i]))

    fig, ax = plt.subplots(4,1,sharex=True)
    ax[0].plot(rm(data['total_reward'],smoothing))
    ax[0].plot(rm(data['bootstrap_reward'],smoothing))
    ax[1].plot(rm(norm, smoothing))
    ax[2].plot(rm(data['loss'][1],smoothing), label='ecv')
    ax[2].plot(rm(data['mf_loss'][1],smoothing), label='mfv')
    ax[2].legend(loc=0)

    ax[3].plot(rm(data['weights']['h0'], smoothing),label='h0')
    ax[3].plot(rm(data['weights']['h1'], smoothing),label='h1')
    ax[3].plot(rm(data['weights']['p'], smoothing),label='p')
    ax[3].plot(rm(data['weights']['v'], smoothing),label='v')
    ax[3].legend(loc=0)
    

    plt.show()

print(data['opt_values'])

plot_vdif_norm(data)
#plot_rewards(data)
#plot_maps(env,data, 0,999,10,value=True)



'''