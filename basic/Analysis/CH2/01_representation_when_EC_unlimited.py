import pandas as pd
import sys
sys.path.append('../../modules/')
from Analysis.analysis_utils import get_avg_std, get_grids, avg_performance_over_envs

# import csv data summary
parent_path = '../../Data/'
df = pd.read_csv(parent_path+'throttled_ec_allreps_chebyshev.csv')

groups_to_split = ['env_name','representation','EC_cache_limit']
gb = df.groupby(groups_to_split)["save_id"]


envs_to_plot = ['gridworld:gridworld-v11','gridworld:gridworld-v41','gridworld:gridworld-v31','gridworld:gridworld-v51']
pcts_to_plot = [100]
reps_to_plot = ['random', 'onehot','conv_latents','place_cell','analytic successor'] # df.representation.unique()
grids        = get_grids(envs_to_plot)

avg_performance_over_envs(gb,envs_to_plot,reps_to_plot,pcts_to_plot,grids,save=True,savename='unbounded_mem')