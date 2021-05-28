import pandas as pd
from Analysis.analysis_utils import get_grids
from Analysis.analysis_utils import avg_performance_over_envs, avg_performance_over_envs_violins, avg_perf_over_envs_lines

# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'throttled_ec_allreps_chebyshev.csv')

groups_to_split = ['env_name','representation','EC_cache_limit']
gb = df.groupby(groups_to_split)["save_id"]

envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
reps_to_plot = ['random','onehot','conv_latents','place_cell', 'analytic successor']
pcts_to_plot = [100]
grids = get_grids(envs_to_plot)
avg_performance_over_envs(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,savename='unrestricted_allreps_bars',save=True)
