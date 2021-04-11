import numpy as np
import pickle
import sys
import gym
import pandas as pd

sys.path.append('../modules/')
from Utils import running_mean as rm

def get_id_dict(df):
    master_dict = {}
    envs = df.env_name.unique()
    reps = df.representation.unique()
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
