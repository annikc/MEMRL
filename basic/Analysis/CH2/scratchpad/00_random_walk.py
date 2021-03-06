import pandas as pd
import sys
sys.path.append('../../modules/')
from Analysis.analysis_utils import get_grids, avg_performance_over_envs, avg_perf_over_envs_lines
from Analysis.analysis_utils import no_rm_avg_std
import matplotlib.pyplot as plt

# import csv data summary
parent_path = '../../Data/'
#df = pd.read_csv(parent_path+'random_forget_ec.csv')
df = pd.read_csv(parent_path+'throttled_ec_allreps_chebyshev.csv')
ref_df = pd.read_csv(parent_path+'random_walk.csv')
gb_ref = ref_df.groupby(['env_name'])['save_id']

groups_to_split = ['env_name','representation','EC_cache_limit']
gb = df.groupby(groups_to_split)["save_id"]


envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
pcts_to_plot = [100,75,50,25]
reps_to_plot = ['random', 'onehot','conv_latents','place_cell','analytic successor'] # df.representation.unique()
grids        = get_grids(envs_to_plot)

avg_performance_over_envs(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,legend='pcts',savename='oldest_forgetting_randomwalk_ref',save=True,ref_gb=gb_ref)#
avg_perf_over_envs_lines(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,legend='reps',savename='oldest_forgetting_lines',save=True,ref_gb=gb_ref)

''':
env = envs_to_plot[0]
pct = 50
rep = 'random'
cache_limits = analysis_specs['cache_limits']
list_of_ids = list(gb.get_group((env, rep, cache_limits[env][pct])))
plot_each(list_of_ids,data_dir=parent_path, cutoff=5000,smoothing=100)
'''