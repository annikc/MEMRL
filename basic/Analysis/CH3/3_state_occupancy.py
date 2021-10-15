## import some things
import numpy as np
import pandas as pd
import gym
from Analysis.analysis_utils import structured_unstructured, LINCLAB_COLS, plot_specs, analysis_specs, linc_coolwarm
from Analysis.analysis_utils import get_grids
from modules.Utils import running_mean as rm
from Analysis.analysis_utils import avg_performance_over_envs, avg_perf_over_envs_lines, avg_performance_over_envs_violins
import pickle
import matplotlib.pyplot as plt

# import csv data summary
parent_path = '../../Data/'

# get baseline (MF retrain alone)
base_ = pd.read_csv(parent_path+'train_test_shallowAC.csv')
base_['representation'] = base_['representation'].apply(structured_unstructured)

# get EC with bootstrapped MF
df = pd.read_csv(parent_path+'bootstrapped_retrain_shallow_AC.csv')
df['representation'] = df['representation'].apply(structured_unstructured)

groups_to_split = ['env_name','representation']

gb_base = base_.groupby(groups_to_split+['num_trials'])["save_id"]
gb = df.groupby(groups_to_split+['EC_cache_limit','num_trials'])["save_id"]

colors = {100:LINCLAB_COLS['red'], 75: LINCLAB_COLS['orange'], 50:LINCLAB_COLS['green'], 25:LINCLAB_COLS['purple']}

def occupancy_plot(env_name, rep, pcts_to_plot):
    env = gym.make(env_name)
    plt.close()
    ec_group = df.groupby(groups_to_split+['EC_cache_limit','extra_info'])["save_id"]
    print(len(env.useable))

    fig, ax = plt.subplots(2,len(pcts_to_plot))

    max = 0.02
    for p, pct in enumerate(pcts_to_plot):

        EC = []
        MF = []
        reward_index = env.twoD2oneD(list(env.rewards.keys())[0])
        print(reward_index)
        id_list = list(ec_group.get_group((env_name,rep,int(cache_limits[env_name][100]*(pct/100)),'occ_map')))
        for id_num in id_list:
            with open(parent_path+f'results/{id_num}_data.p','rb') as f:
                dats = pickle.load(f)
            total_rwd_visits = np.count_nonzero(np.array(dats['total_reward']) >-2.49)
            EC_map = dats['EC_occupancy']
            EC_map[reward_index] += total_rwd_visits ## correct for dumb way of counting state occupancy
            EC_visits = np.nansum(EC_map)
            EC.append(EC_map/EC_visits)

            MF_map = dats['MF_occupancy']
            total_rwd_visits = np.count_nonzero(np.array(dats['bootstrap_reward']) >-2.49)
            MF_map[reward_index] += total_rwd_visits
            MF_visits = np.nansum(MF_map)
            MF.append(MF_map/MF_visits)

        EC_occ = np.nanmean(EC,axis=0).reshape(20,20)
        MF_occ = np.nanmean(MF,axis=0).reshape(20,20)

        for item in env.obstacle:
            coord = env.oneD2twoD(item)
            EC_occ[coord] = np.nan
            MF_occ[coord] = np.nan

        avg_visit_freq = 1/len(env.useable)

        cmap = 'RdBu_r'
        min_ = -4
        max_ = 4
        a= ax[0,p].imshow(np.log(EC_occ/avg_visit_freq),cmap=cmap, vmin=min_,vmax=max_)
        ax[0,p].get_xaxis().set_visible(False)
        ax[0,p].get_yaxis().set_visible(False)

        ax[0,p].set_title(f'{pct}')
        b = ax[1,p].imshow(np.log(MF_occ/avg_visit_freq),cmap=cmap,vmin=min_, vmax=max_)
        ax[1,p].get_xaxis().set_visible(False)
        ax[1,p].get_yaxis().set_visible(False)
    plt.colorbar(a, ax = ax[0,p])
    plt.colorbar(b, ax = ax[1])

    plt.savefig(f'../figures/CH3/state_occ_{env_name[-2:]}_{rep}_all.svg')
    plt.show()



####
envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
pcts_to_plot = [100,75,50,25]
grids = get_grids(envs_to_plot)
cache_limits = analysis_specs['cache_limits']
#plot_single_retraining(envs_to_plot[1],[25,50,75,100],'structured',index=1)
#plot_all_retraining(envs_to_plot[1],[25,50,75,100],['structured','unstructured'])

env = envs_to_plot[1]
rep = 'structured'
pct = 25
for env in envs_to_plot:
    e = gym.make(env)
    plt.close()
    occupancy_plot(env, rep, pcts_to_plot)

