import numpy as np
import pandas as pd
import gym
from Analysis.analysis_utils import structured_unstructured, LINCLAB_COLS, plot_specs, analysis_specs
from Analysis.analysis_utils import get_grids
from modules.Utils.gridworld_plotting import plot_polmap, plot_valmap, plot_pref_pol
from modules.Agents.RepresentationLearning.learned_representations import latents, sr, onehot
from optimal_p_v_maps import attempt_opt_pol
import pickle
import matplotlib.pyplot as plt
cache_limits = analysis_specs['cache_limits']

# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'shallowAC_withPVmaps.csv')
groups_to_split = ['env_name','representation']
df_gb = df.groupby(groups_to_split)["save_id"]

envs_to_plot = ['gridworld:gridworld-v1','gridworld:gridworld-v4','gridworld:gridworld-v3','gridworld:gridworld-v5']
pcts_to_plot = [100,75,50,25]
reps_to_plot = ['unstructured','structured']
grids = get_grids(envs_to_plot)

def get_agent(df_gb, env, rep):
    id_num = list(df_gb.get_group((env,rep)))[0]
    with open(parent_path+ f'agents/{id_num}_data.p', 'rb') as f:
        dats = pickle.load(f)
        p_maps = dats['P_snap']
        v_maps = dats['V_snap']
        print(len(p_maps), len(dats['total_reward']))

