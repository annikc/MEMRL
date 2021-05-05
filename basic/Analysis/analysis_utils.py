import numpy as np
import pickle
import sys
import gym
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../modules/')
from modules.Utils import running_mean as rm

### plotting settings
labels_for_plot = {'analytic successor':'SR',
                   'onehot':'onehot',
                   'random':'random',
                   'place_cell':'PC',
                   'conv_latents':'latent'}

convert_rep_to_color = {'analytic successor':'C0',
                        'onehot':'C1',
                        'random':'C2',
                        'place_cell':'C4',
                        'conv_latents':'C3'}




def get_env_rep_id_dict(df, **kwargs):
    master_dict = {}
    envs = kwargs.get('envs', df.env_name.unique())
    reps = kwargs.get('reps',df.representation.unique())
    for env in envs:
        master_dict[env] = {}
        for rep in reps:
            id_list = list(df.loc[(df['env_name']==env)
             & (df['representation']==rep)]['save_id'])

            master_dict[env][rep]=id_list
    return master_dict

def get_avg_std(list_of_ids, cutoff=5000, smoothing=500):
    data_dir='../../Data/results/'
    results = []
    for id_num in list_of_ids:
        with open(data_dir+ f'{id_num}_data.p', 'rb') as f:
            dats = pickle.load(f)
            reward_info = dats['total_reward'][0:cutoff]
            results.append(reward_info)

    pp = np.vstack(results)
    avg_ = rm(np.mean(pp,axis=0),smoothing)
    std_ = rm(np.std(pp, axis=0), smoothing)

    return avg_, std_

def get_detailed_avg_std(list_of_ids, cutoff=5000, smoothing=500):
    data_dir='../../Data/results/'
    results = []
    for id_num in list_of_ids:
        with open(data_dir+ f'{id_num}_data.p', 'rb') as f:
            dats = pickle.load(f)
            reward_info = dats['total_reward'][0:cutoff]
            results.append(reward_info)

    pp = np.vstack(results)
    avg_ = rm(np.mean(pp,axis=0),smoothing)
    std_ = rm(np.std(pp, axis=0), smoothing)
    a_s, s_s = [], []
    for xx in range(len(pp)):
        rr = pp[xx]
        smoothed_rr = rm(rr, smoothing)
        a_s.append(np.mean(smoothed_rr))
        s_s.append(np.std(smoothed_rr))

    return avg_, std_, np.asarray(a_s), np.asarray(s_s)

def get_grids(gw_versions):
    for version in gw_versions:
        env_names = [f'gridworld:gridworld-v{version}' for version in gw_versions]

    grids = []
    for ind, environment_to_plot in enumerate(env_names):
        env = gym.make(environment_to_plot)
        plt.close()
        grids.append(env.grid)

    return grids

def plot_each(list_of_ids,data_dir,cutoff=25000, smoothing=500):
    plt.figure()
    for id_num in list_of_ids:
        with open(data_dir+ f'results/{id_num}_data.p', 'rb') as f:
            dats = pickle.load(f)
            reward_info = dats['total_reward'][0:cutoff]
        processed_rwd = rm(reward_info, smoothing)
        plt.plot(processed_rwd, label=id_num[0:8])
    plt.legend(loc='upper center', bbox_to_anchor=(0.1,1.1))
    plt.ylim([-4,12])
    plt.show()