import numpy as np
import pandas as pd
import gym
from Analysis.analysis_utils import structured_unstructured, LINCLAB_COLS, plot_specs, analysis_specs
from Analysis.analysis_utils import get_grids
from modules.Utils.gridworld_plotting import plot_polmap, plot_valmap, plot_pref_pol
from modules.Utils import running_mean as rm
from Analysis.analysis_utils import avg_performance_over_envs, avg_perf_over_envs_lines, avg_performance_over_envs_violins
import pickle
import matplotlib.pyplot as plt
cache_limits = analysis_specs['cache_limits']

# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'shallowAC_withPVmaps.csv')

df['representation'] = df['representation'].apply(structured_unstructured)
groups_to_split = ['env_name','representation']
df_gb = df.groupby(groups_to_split)["save_id"]

envs_to_plot = ['gridworld:gridworld-v1','gridworld:gridworld-v4','gridworld:gridworld-v3','gridworld:gridworld-v5']
pcts_to_plot = [100,75,50,25]
reps_to_plot = ['unstructured','structured']
grids = get_grids(envs_to_plot)

env = envs_to_plot[2]
env_obj = gym.make(env)
plt.close()
pct = 100
rep = 'structured'

id_list = list(df_gb.get_group((env,rep)))
print(env, rep, len(id_list))
for i, id_num in enumerate(id_list):
    with open(parent_path+ f'results/{id_num}_data.p', 'rb') as f:
        dats = pickle.load(f)
        p_maps = dats['P_snap']
        v_maps = dats['V_snap']
        print(len(p_maps), len(dats['total_reward']))


plot_pref_pol(env_obj, p_maps[2999])