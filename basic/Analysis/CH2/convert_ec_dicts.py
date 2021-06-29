import gym
import pickle
import numpy as np
import pandas as pd

from modules.Utils.gridworld_plotting import plot_pref_pol, plot_polmap
from modules.Agents.EpisodicMemory import EpisodicMemory as Memory
from modules.Agents.RepresentationLearning.learned_representations import sr, onehot
from Analysis.analysis_utils import analysis_specs,linc_coolwarm,make_env_graph,compute_graph_distance_matrix, LINCLAB_COLS
from scipy.special import rel_entr
from scipy.stats import entropy


# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'ec_track_pols.csv')
groups_to_split = ['env_name','representation','EC_cache_limit']
gb = df.groupby(groups_to_split)["save_id"]

def get_xy_maps(env, state_reps, data,trial_num=-1):
    blank_mem = Memory(cache_limit=400, entry_size=4)
    blank_mem.cache_list = data['ec_dicts'][trial_num]
    xy_pol_grid = np.zeros((2,*env.shape))
    polar_grid = np.zeros(env.shape)
    polar_grid[:]=np.nan
    dxdy = np.array([(0,1),(0,-1),(1,0),(-1,0)]) #D U R L

    for key, value in state_reps.items():
        twoD = env.oneD2twoD(key)
        sr_rep = value
        pol = blank_mem.recall_mem(sr_rep)
        xy = np.dot(pol,dxdy)
        xy_pol_grid[0,twoD] = xy[0]
        xy_pol_grid[1,twoD] = xy[1]
        '''rads = np.arctan(xy[1]/xy[0])
        degs = rads*(180/np.pi)
        if xy[0]>=0 and xy[1]>=0: #Q1
            theta = degs
        elif xy[0]<0: #Q2 and Q3
            theta = degs+180
        elif xy[0]>=0 and xy[1]<=0:
            theta = degs+360
        else:
            theta = -1

        polar_grid[twoD] = theta'''

    return xy_pol_grid,polar_grid

def save_processed_data(env_name,cache_limits,pcts_to_plot,reps_to_plot,save='polar'):
    rep_dict = {'analytic successor': sr, 'onehot':onehot}
    env = gym.make(env_name)

    for pct in pcts_to_plot:
        for rep in reps_to_plot:
            print(env_name, rep,pct)
            run_id = list(gb.get_group((env_name,rep,cache_limits[env_name][pct])))[0]
            print(run_id)

            with open(f'../../Data/results/{run_id}_data.p', 'rb') as f:
                data = pickle.load(f)

            state_reps, _, __, ___ = rep_dict[rep](env)
            if rep == 'analytic successor':
                for s1 in env.obstacle:
                    state_reps.pop(s1)

            polar_pref = []
            for i in range(800,1000):
                print(i)
                xy_map, polar_map = get_xy_maps(env,state_reps,data,i)
                if save == 'polar':
                    polar_pref.append(polar_map)
                elif save == 'xy':
                    polar_pref.append(xy_map)

            with open(f'../../Data/ec_dicts/lifetime_dicts/{run_id}_{save}coord.p', 'wb') as savedata:
                pickle.dump(np.asarray(polar_pref), savedata)

def get_ec_policy_map(env, state_reps, data, trial_num, full=True):
    ec_pol_grid = np.zeros(env.shape, dtype=[(x, 'f8') for x in env.action_list])

    blank_mem = Memory(cache_limit=400, entry_size=4)
    blank_mem.cache_list = data['ec_dicts'][trial_num]

    if full:
        for k in state_reps.keys():
            twoD = env.oneD2twoD(k)
            sr_rep = state_reps[k]
            pol = blank_mem.recall_mem(sr_rep)

            ec_pol_grid[twoD] = tuple(pol)
    else:
        for k in blank_mem.cache_list.keys():
            oneDstate = blank_mem.cache_list[k][2]
            twoD = env.oneD2twoD(oneDstate)
            sr_rep = state_reps[oneDstate]
            pol = blank_mem.recall_mem(sr_rep)

            ec_pol_grid[twoD] = tuple(pol)

    return ec_pol_grid


env_name = 'gridworld:gridworld-v41'
reps_to_plot = ['analytic successor', 'onehot']
pcts_to_plot = [100,75,50,25]
cache_limits = analysis_specs['cache_limits']

save_processed_data(env_name,cache_limits,pcts_to_plot,reps_to_plot,save='xy')