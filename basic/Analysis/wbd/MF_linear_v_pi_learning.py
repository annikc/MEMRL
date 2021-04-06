import gym
import pickle
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from scipy.stats import entropy

from basic.modules.Utils import running_mean as rm
from basic.modules.Utils.gridworld_plotting import plot_polmap, plot_pref_pol, plot_valmap, plot_world
from basic.Analysis.vis_bootstrap_pol_maps import daves_idea, plot_pol_evol, trajectories, plot_maps, plot_rewards

filename = '../Data/linear_track_vpi.csv'
df = pd.read_csv(filename)
val_norm = {}
KLD_norm = {}
for x in range(len(df)):
    run_id = df['run_id'].loc[x]
    lr = df['lr'].loc[x]
    with open(f'../Data/results/{run_id}_data.p', 'rb') as f:
        data = pickle.load(f)

    env_name = df['env_type'].loc[x]
    env = gym.make(env_name)
    plt.close()
    # only works for linear track:
    optimal_policy = np.zeros(env.shape, dtype=[(x, 'f8') for x in env.action_list])
    for x in range(env.shape[1]):
        optimal_policy[0, x] = (0.0000000001,0.0000000001,1-3*0.0000000001, 0.0000000001) ## need nonzero for KLD to work properly
        if x == env.shape[1]-1:
            optimal_policy[0,x] = (0.25, 0.25, 0.25, 0.25)

    # first calculate KLD of vectors:

    K_norm = []
    v_norm = []
    for i in range(len(data['P_snap'])):
        v_norm.append(np.linalg.norm(data['opt_values']-data['V_snap'][i]))
        val_norm[lr] = v_norm
        KLD = np.zeros(env.shape[1])
        for x in range(env.shape[1]):
            mf_pol = list(data['P_snap'][i][0,x])
            opt_pol = list(optimal_policy[0,x])
            KLD[x] = entropy(mf_pol,opt_pol)
        K_norm.append(np.linalg.norm(KLD))
    KLD_norm[lr] = K_norm


fig, ax = plt.subplots(2,1, sharex=True)
# norm of values
for id in val_norm.keys():
    ax[0].plot(val_norm[id], label=id)
# norm of policy KLD
    ax[1].plot(KLD_norm[id], label=id)

ax[0].legend(loc=0)
ax[1].legend(loc=1)
plt.show()

